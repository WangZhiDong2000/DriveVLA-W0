"""
Rollout collector for DriveVLA-W0 GRPO (Task 1.6).

Assembles Tasks 1.1-1.5 modules into a complete RolloutBatch for Phase 4
two-pass GRPO training.

Corresponds to DD-v2 forward_train_rl (model_rl.py:800-939), adapted for
Euler ODE flow matching instead of DDPM DDIM:
  - Stochasticity source: per-step multiplicative noise eps_step ~ N(0, σ²)
    (vs. DDIM stochastic transition in DD-v2)
  - logp_old = log N(eps_step; 0, σ²), differentiable in Pass 2 through
    implied_eps = x_prev_phys / x_prev_mean_phys_new - 1

Interface:
  collect_rollouts(denoise_fn, gt_trajectory, anchor_centers, scorer,
                   metric_cache_paths, q01, q99, ...) -> RolloutBatch

  denoise_fn(x_t, t) -> v_t  (callable, decouples rollout from model arch)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List

import torch

from models.policy_head.multiplicative_noise import apply_multiplicative_noise
from utils.rl_modules.intra_anchor_advantage import compute_intra_anchor_advantages
from utils.rl_modules.inter_anchor_truncated import apply_inter_anchor_truncation


# ---------------------------------------------------------------------------
# RolloutBatch: Pass 1 snapshot consumed by Phase 4 Pass 2
# ---------------------------------------------------------------------------

@dataclass
class RolloutBatch:
    """Complete Pass 1 snapshot for GRPO two-pass training.

    All tensors on the device passed to collect_rollouts.
    K = N_anchor × G (total candidates per scene).
    """
    anchors:        torch.Tensor            # (B, N_anchor, N_f, 3)  physical
    x_t_per_step:   torch.Tensor            # (B, K, T, N_f, 3)      norm [-1,1], model input
    eps_step:       torch.Tensor            # (B, K, T, 2)            per-step mult noise
    logp_old:       torch.Tensor            # (B, K, T)               detached log-π
    rewards:        torch.Tensor            # (B, K)
    sub_rewards:    Dict[str, torch.Tensor] # 7 keys, each (B, K)
    advantages:     torch.Tensor            # (B, K, T)               truncated + discounted
    trajectories:   torch.Tensor            # (B, K, N_f, 3)          physical, final
    gt_trajectory:  torch.Tensor            # (B, N_f, 3)             physical


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _denorm(x: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → physical (m / rad)."""
    return (x + 1.0) / 2.0 * (q99 - q01) + q01


def _norm(x_phys: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    """Physical (m / rad) → [-1, 1]."""
    return (x_phys - q01) / (q99 - q01) * 2.0 - 1.0


def _logp_gauss(eps: torch.Tensor, sigma: float) -> torch.Tensor:
    """Log probability of 2D isotropic Gaussian N(0, σ²I).

    Args:
        eps: (..., 2) raw noise samples.
        sigma: standard deviation.

    Returns:
        (...,) log-probability.
    """
    var = sigma ** 2
    return (-0.5 * (eps ** 2).sum(dim=-1) / var) - math.log(2.0 * math.pi * var)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def collect_rollouts(
    denoise_fn: Callable[[torch.Tensor, float], torch.Tensor],
    gt_trajectory: torch.Tensor,
    anchor_centers: torch.Tensor,
    scorer,
    metric_cache_paths: List[str],
    q01: torch.Tensor,
    q99: torch.Tensor,
    num_groups: int = 8,
    num_steps: int = 10,
    sigma_init: float = 0.5,
    sigma_step: float = 0.04,
    discount: float = 0.8,
    device: torch.device = torch.device("cpu"),
) -> RolloutBatch:
    """Collect GRPO rollouts for one training batch.

    Args:
        denoise_fn:          (B*K, N_f, 3) norm → velocity (B*K, N_f, 3) norm.
                             Caller controls @torch.no_grad() context.
        gt_trajectory:       (B, N_f, 3) physical space (meters/rad).
        anchor_centers:      (N_anchor, N_f, 3) physical space (Task 1.1 output).
        scorer:              PDMRewardWrapper instance (Task 1.5).
        metric_cache_paths:  List[str] of length B, paths to .lzma files.
        q01:                 (3,) 1st-percentile normalization stats.
        q99:                 (3,) 99th-percentile normalization stats.
        num_groups:          G, candidates per anchor (default 8).
        num_steps:           T, truncated Euler denoising steps (default 10).
        sigma_init:          σ for initial anchor multiplicative noise.
        sigma_step:          σ for per-step multiplicative noise (= min_clip).
        discount:            γ for temporal discounting of advantages.
        device:              Target device for returned tensors.

    Returns:
        RolloutBatch with all 9 fields populated; K = N_anchor × G.
    """
    B = gt_trajectory.shape[0]
    N_anchor, N_f, _ = anchor_centers.shape
    G = num_groups
    K = N_anchor * G
    T = num_steps
    q01_d = q01.to(device)
    q99_d = q99.to(device)

    # ------------------------------------------------------------------
    # 1. Expand anchors → (B, K, N_f, 3)
    #    Memory layout: k = anchor_idx * G + group_idx
    #    This matches DD-v2's view(bs, num_groups, ego_fut_mode) order.
    # ------------------------------------------------------------------
    anchor_centers_d = anchor_centers.to(device)
    anchors_k = (
        anchor_centers_d.unsqueeze(0)              # (1, N_anchor, N_f, 3)
        .expand(B, -1, -1, -1)                     # (B, N_anchor, N_f, 3)
        .unsqueeze(2).expand(-1, -1, G, -1, -1)    # (B, N_anchor, G, N_f, 3)
        .reshape(B, K, N_f, 3)                     # (B, K, N_f, 3)
    )

    # ------------------------------------------------------------------
    # 2. Initial multiplicative noise → x_1 (diverse starting points)
    #    Each of the B*K candidates gets independent eps_init.
    # ------------------------------------------------------------------
    anchors_flat = anchors_k.reshape(B * K, N_f, 3)   # (B*K, N_f, 3)
    x1_phys, _ = apply_multiplicative_noise(
        anchors_flat, sigma=sigma_init, min_clip=0.04)  # (B*K, N_f, 3)
    x_t = _norm(x1_phys, q01_d, q99_d)                 # (B*K, N_f, 3) normalized

    # ------------------------------------------------------------------
    # 3. T-step Euler ODE + per-step multiplicative noise
    # ------------------------------------------------------------------
    all_x_t:   List[torch.Tensor] = []   # each (B*K, N_f, 3)
    all_eps:   List[torch.Tensor] = []   # each (B*K, 2)
    all_logp:  List[torch.Tensor] = []   # each (B*K,)

    dt = -1.0 / T          # negative: t walks from 1.0 → 0.0
    for step_idx in range(T):
        t = 1.0 + step_idx * dt           # 1.0, 0.9, ..., 0.1

        all_x_t.append(x_t.clone())        # record model input (normalized)

        v_t = denoise_fn(x_t, t)           # (B*K, N_f, 3) normalized velocity
        x_prev_mean = x_t + dt * v_t       # Euler step (deterministic, normalized)

        # Apply multiplicative noise in physical space
        x_prev_mean_phys = _denorm(x_prev_mean, q01_d, q99_d)
        x_prev_phys, eps_mul = apply_multiplicative_noise(
            x_prev_mean_phys, sigma=sigma_step, min_clip=0.04)
        # eps_mul: (B*K, 1, 2)
        eps_2d = eps_mul.squeeze(-2)                         # (B*K, 2)

        all_eps.append(eps_2d)
        all_logp.append(_logp_gauss(eps_2d, sigma_step))    # (B*K,)

        x_t = _norm(x_prev_phys, q01_d, q99_d)              # next step input

    # ------------------------------------------------------------------
    # 4. Stack and reshape collected tensors
    # ------------------------------------------------------------------
    x_t_per_step = (
        torch.stack(all_x_t, dim=1)       # (B*K, T, N_f, 3)
        .reshape(B, K, T, N_f, 3)
    )
    eps_step = (
        torch.stack(all_eps, dim=1)        # (B*K, T, 2)
        .reshape(B, K, T, 2)
    )
    logp_old = (
        torch.stack(all_logp, dim=1)       # (B*K, T)
        .reshape(B, K, T)
        .detach()
    )

    # Final trajectories (physical space) after T Euler steps
    trajectories = _denorm(x_t, q01_d, q99_d).reshape(B, K, N_f, 3)

    # ------------------------------------------------------------------
    # 5. PDM scoring (Task 1.5)
    # ------------------------------------------------------------------
    rewards, reward_gt, sub_rewards = scorer.score(
        trajectories.cpu().numpy(),       # (B, K, N_f, 3)
        gt_trajectory.cpu().numpy(),      # (B, N_f, 3)
        metric_cache_paths,
        device=device,
    )
    # rewards: (B, K)  reward_gt: (B,)  sub_rewards: dict[str, (B, K)]

    # ------------------------------------------------------------------
    # 6. Intra-anchor advantage normalization (Task 1.3)
    # ------------------------------------------------------------------
    rewards_grouped = rewards.reshape(B, N_anchor, G)           # (B, N_anchor, G)
    adv_intra = compute_intra_anchor_advantages(rewards_grouped) # (B, N_anchor, G)

    # ------------------------------------------------------------------
    # 7. Inter-anchor truncation (Task 1.4)
    # ------------------------------------------------------------------
    nc = sub_rewards["no_collision"].reshape(B, N_anchor, G)    # (B, N_anchor, G)
    da = sub_rewards["drivable_area"].reshape(B, N_anchor, G)   # (B, N_anchor, G)
    adv_trunc = apply_inter_anchor_truncation(
        adv_intra, rewards_grouped, reward_gt, nc, da)          # (B, N_anchor, G)

    # ------------------------------------------------------------------
    # 8. Temporal discounting (DD-v2 model_rl.py:926-932)
    #    discount_vec[i] = γ^(T-i-1):  last step = γ^0 = 1, first = γ^(T-1)
    # ------------------------------------------------------------------
    adv_k = adv_trunc.reshape(B, K, 1).expand(-1, -1, T)        # (B, K, T)
    discount_vec = torch.tensor(
        [discount ** (T - i - 1) for i in range(T)],
        dtype=rewards.dtype, device=device,
    )
    advantages = adv_k * discount_vec                            # (B, K, T)

    # ------------------------------------------------------------------
    # 9. Assemble RolloutBatch
    # ------------------------------------------------------------------
    return RolloutBatch(
        anchors       = anchor_centers_d.unsqueeze(0).expand(B, -1, -1, -1),
        x_t_per_step  = x_t_per_step,
        eps_step      = eps_step,
        logp_old      = logp_old,
        rewards       = rewards,
        sub_rewards   = sub_rewards,
        advantages    = advantages,
        trajectories  = trajectories,
        gt_trajectory = gt_trajectory.to(device),
    )
