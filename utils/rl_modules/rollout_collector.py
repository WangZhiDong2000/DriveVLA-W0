"""
Rollout collector for DriveVLA-W0 GRPO (Task 4.1 rewrite).

Thin wrapper over Emu3Pi0.sample_actions_stochastic — all denoising math,
stochastic ODE step, and log_prob computation live in
models/policy_head/stochastic_ode_sampler._gaussian_log_prob_z (σ_lp_min=0.10).
This guarantees that Pass-1 logp_old and Pass-2 recompute_log_prob are
bit-identical (Plan_3 §Task 2.6 / §Task 5.5).

Differences from Phase-1 version:
- Removed internal denoise loop and _logp_gauss (ε-form, wrong σ); these
  conflicted with the DD-v2 z-form log_prob in stochastic_ode_sampler.
- Anchor-chunk: K = N_anchor × G candidates split into ceil(N_anchor / chunk)
  sub-batches to avoid VLM prefill OOM on a single GPU.
- RolloutBatch gains z_t_per_step and z_next_phys_per_step for Pass-2
  recompute_log_prob (Task 4.3).
- metric_cache_paths is now a List[Optional[str]] — None entries allowed when
  using MockPDMRewardWrapper.

Interface:
  collect_rollouts(model, batch, anchors_norm, scorer, *,
                   metric_cache_paths, num_groups, num_steps, sigma_init,
                   sigma_step, sigma_logprob_min, discount, anchor_chunk)
  -> RolloutBatch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from models.policy_head.anchored_flow_path import normalize, denormalize
from models.policy_head.multiplicative_noise import apply_multiplicative_noise
from utils.rl_modules.intra_anchor_advantage import compute_intra_anchor_advantages
from utils.rl_modules.inter_anchor_truncated import apply_inter_anchor_truncation


# ---------------------------------------------------------------------------
# RolloutBatch: Pass-1 snapshot consumed by Phase-4 Pass-2
# ---------------------------------------------------------------------------

@dataclass
class RolloutBatch:
    """Complete Pass-1 snapshot for GRPO two-pass training.

    Shape conventions:
      B = per-device batch size
      K = N_anchor × G  (total candidates per scene)
      T = num_truncated_steps
      N_F = action_frames

    All tensor fields are on the device of the incoming model.
    logp_old is detached; z_t_per_step / z_next_phys_per_step are detached
    normalized / physical respectively — ready for recompute_log_prob in Pass-2.
    """
    # Anchor that seeded each trajectory (broadcast, not per-group)
    anchors:               torch.Tensor  # (B, N_anchor, N_F, 3) physical

    # Per-step ODE state (Pass-2 inputs for recompute_log_prob)
    z_t_per_step:          torch.Tensor  # (B, K, T, N_F, 3) norm [-1,1], detached
    z_next_phys_per_step:  torch.Tensor  # (B, K, T, N_F, 3) physical, detached

    # Log-π from Pass-1 (detached; used as logp_old in IS ratio)
    logp_old:              torch.Tensor  # (B, K, T)  detached

    # PDM scores
    rewards:               torch.Tensor  # (B, K)
    sub_rewards:           Dict[str, torch.Tensor]  # 7 keys, each (B, K)

    # GRPO advantages (intra-anchor normed + inter-anchor truncated + discounted)
    advantages:            torch.Tensor  # (B, K, T)

    # Final trajectories after T Euler steps
    trajectories:          torch.Tensor  # (B, K, N_F, 3) physical

    # GT trajectory for IL loss
    gt_trajectory:         torch.Tensor  # (B, N_F, 3) physical


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_rollouts(
    model,
    batch: Dict[str, torch.Tensor],
    anchors_norm: torch.Tensor,
    scorer,
    *,
    metric_cache_paths: List[Optional[str]],
    num_groups: int = 8,
    num_steps: int = 10,
    sigma_init: float = 0.5,
    sigma_step: float = 0.04,
    sigma_logprob_min: float = 0.10,
    discount: float = 0.8,
    anchor_chunk: int = 20,
) -> RolloutBatch:
    """Collect GRPO rollouts for one training batch.

    Args:
        model:              Emu3Pi0 instance (in train() mode on device).
        batch:              Collated dataloader dict containing at minimum:
                              input_ids, attention_mask, pre_action, cmd,
                              action  (B, N_F, 3) normalized GT.
        anchors_norm:       (N_anchor, N_F, 3) normalized [-1,1] anchor centers.
        scorer:             PDMRewardWrapper or MockPDMRewardWrapper instance.
        metric_cache_paths: len=B paths to .lzma files, or None for mock scorer.
        num_groups:         G, stochastic rollouts per anchor.
        num_steps:          T, truncated Euler ODE steps.
        sigma_init:         σ for initial anchor multiplicative noise (anchor → z_1).
        sigma_step:         σ for per-step stochastic Euler noise.
        sigma_logprob_min:  σ floor for log-prob (≥ 0.10, Plan §Task 5.5).
        discount:           γ for temporal advantage discounting.
        anchor_chunk:       N_anchor sub-batch size to avoid VLM OOM.

    Returns:
        RolloutBatch with all fields populated (all tensors on model device).
    """
    device = next(model.parameters()).device
    _dtype = next(model.parameters()).dtype

    gt_action_norm = batch["action"].to(device, dtype=_dtype)   # (B, N_F, 3)
    B, N_F, _ = gt_action_norm.shape
    N_anchor = anchors_norm.shape[0]
    G = num_groups
    K = N_anchor * G
    T = num_steps

    anchors_norm_d = anchors_norm.to(device, dtype=_dtype)      # (N_anchor, N_F, 3)
    q01 = model.action_q01.to(device, dtype=_dtype)             # (3,)
    q99 = model.action_q99.to(device, dtype=_dtype)

    # GT physical for scoring
    gt_phys = denormalize(gt_action_norm, q01, q99)             # (B, N_F, 3)

    # ------------------------------------------------------------------
    # Pre-compute VLM input tensors once (shared across anchor chunks)
    # ------------------------------------------------------------------
    input_ids      = batch.get("input_ids",      None)
    attention_mask = batch.get("attention_mask", None)
    pre_action     = batch["pre_action"].to(device, dtype=_dtype)
    cmd            = batch["cmd"].to(device, dtype=_dtype)

    if input_ids is not None:
        input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # ------------------------------------------------------------------
    # Anchor-chunked rollout (avoids B*K VLM OOM)
    # ------------------------------------------------------------------
    # Accumulate per-chunk outputs; we'll cat along the K dimension.
    all_z_t:        List[torch.Tensor] = []   # each (B, K_c, T, N_F, 3)
    all_z_next_phys: List[torch.Tensor] = []  # each (B, K_c, T, N_F, 3)
    all_logp:        List[torch.Tensor] = []  # each (B, K_c, T)
    all_trajs:       List[torch.Tensor] = []  # each (B, K_c, N_F, 3)

    n_chunks = (N_anchor + anchor_chunk - 1) // anchor_chunk

    for chunk_idx in range(n_chunks):
        a_start = chunk_idx * anchor_chunk
        a_end   = min(N_anchor, a_start + anchor_chunk)
        chunk_anc_norm = anchors_norm_d[a_start:a_end]   # (N_c, N_F, 3)
        N_c = chunk_anc_norm.shape[0]
        K_c = N_c * G

        # Expand: each anchor → G identical copies → (N_c*G, N_F, 3)
        chunk_anc_norm_g = (
            chunk_anc_norm.unsqueeze(1)           # (N_c, 1, N_F, 3)
            .expand(-1, G, -1, -1)                # (N_c, G, N_F, 3)
            .reshape(N_c * G, N_F, 3)             # (K_c, N_F, 3)
        )
        # Broadcast over B → (B*K_c, N_F, 3)
        chunk_anc_bk = (
            chunk_anc_norm_g.unsqueeze(0)         # (1, K_c, N_F, 3)
            .expand(B, -1, -1, -1)                # (B, K_c, N_F, 3)
            .reshape(B * K_c, N_F, 3)
        )

        # Build z_init from anchor + sigma_init multiplicative noise (physical space)
        chunk_anc_phys_bk = denormalize(chunk_anc_bk, q01, q99)
        z_init_phys, _ = apply_multiplicative_noise(
            chunk_anc_phys_bk, sigma=sigma_init, min_clip=0.04)
        z_init_norm = normalize(z_init_phys, q01, q99)     # (B*K_c, N_F, 3)

        # Tile VLM inputs from B → B*K_c
        ids_bk  = input_ids.repeat_interleave(K_c, dim=0) if input_ids is not None else None
        mask_bk = attention_mask.repeat_interleave(K_c, dim=0) if attention_mask is not None else None
        pa_bk   = pre_action.repeat_interleave(K_c, dim=0)
        cmd_bk  = cmd.repeat_interleave(K_c, dim=0)

        result = model.sample_actions_stochastic(
            input_ids=ids_bk,
            pre_action=pa_bk,
            cmd=cmd_bk,
            attention_mask=mask_bk,
            num_steps=T,
            anchor=chunk_anc_bk,
            sigma_step=sigma_step,
            sigma_logprob_min=sigma_logprob_min,
            z_init=z_init_norm,
        )
        # result["z_traj"]           : (T+1, B*K_c, N_F, 3) norm
        # result["log_prob_traj"]    : (T,   B*K_c)
        # result["z_next_phys_traj"] : (T,   B*K_c, N_F, 3) physical, pre-renormalize
        # result["z_final"]          : (B*K_c, N_F, 3) norm

        z_traj         = result["z_traj"]              # (T+1, B*K_c, N_F, 3)
        lp_traj        = result["log_prob_traj"]       # (T,   B*K_c)
        z_next_phys_tr = result["z_next_phys_traj"]    # (T, B*K_c, N_F, 3) physical
        z_fin_norm     = result["z_final"]             # (B*K_c, N_F, 3)

        # z_t_per_step: inputs to each ODE step = z_traj[0..T-1]
        z_t_chunk = (
            z_traj[:-1]                              # (T, B*K_c, N_F, 3)
            .permute(1, 0, 2, 3)                     # (B*K_c, T, N_F, 3)
            .reshape(B, K_c, T, N_F, 3)
            .detach()
        )
        # z_next_phys_per_step: take the physical sample directly from the
        # stochastic step. denormalize(normalize(z_next_phys)) drifts ~1e-3 per
        # dim in bf16; over 16 (x,y) dims this puts log_prob ~2 off and breaks
        # the bit-identical IS-ratio guarantee in Pass-2.
        z_next_phys_chunk = (
            z_next_phys_tr                           # (T, B*K_c, N_F, 3)
            .permute(1, 0, 2, 3)                     # (B*K_c, T, N_F, 3)
            .reshape(B, K_c, T, N_F, 3)
            .detach()
        )
        logp_chunk = (
            lp_traj                                  # (T, B*K_c)
            .permute(1, 0)                           # (B*K_c, T)
            .reshape(B, K_c, T)
            .detach()
        )
        traj_chunk = denormalize(z_fin_norm, q01, q99).reshape(B, K_c, N_F, 3)

        all_z_t.append(z_t_chunk)
        all_z_next_phys.append(z_next_phys_chunk)
        all_logp.append(logp_chunk)
        all_trajs.append(traj_chunk)

    # ------------------------------------------------------------------
    # Concatenate chunks along K dimension
    # ------------------------------------------------------------------
    z_t_all        = torch.cat(all_z_t,         dim=1)   # (B, K, T, N_F, 3)
    z_next_phys_all= torch.cat(all_z_next_phys, dim=1)   # (B, K, T, N_F, 3)
    logp_old_all   = torch.cat(all_logp,        dim=1)   # (B, K, T)
    trajectories   = torch.cat(all_trajs,       dim=1)   # (B, K, N_F, 3)

    # ------------------------------------------------------------------
    # PDM scoring (real or mock)
    # ------------------------------------------------------------------
    rewards, reward_gt, sub_rewards = scorer.score(
        trajectories.float().cpu().numpy(),    # (B, K, N_F, 3)
        gt_phys.float().cpu().numpy(),         # (B, N_F, 3)
        metric_cache_paths,
        device=device,
    )
    # rewards: (B, K)  reward_gt: (B,)  sub_rewards: dict[str, (B, K)]

    # ------------------------------------------------------------------
    # Advantage computation
    # ------------------------------------------------------------------
    rewards_grouped = rewards.reshape(B, N_anchor, G)              # (B, N_anchor, G)
    adv_intra = compute_intra_anchor_advantages(rewards_grouped)   # (B, N_anchor, G)

    nc = sub_rewards["no_collision"].reshape(B, N_anchor, G)
    da = sub_rewards["drivable_area"].reshape(B, N_anchor, G)
    adv_trunc = apply_inter_anchor_truncation(
        adv_intra, rewards_grouped, reward_gt, nc, da)             # (B, N_anchor, G)

    # Temporal discounting: discount_vec[i] = γ^(T-i-1); last step = γ^0 = 1
    adv_k = adv_trunc.reshape(B, K, 1).expand(-1, -1, T)
    discount_vec = torch.tensor(
        [discount ** (T - i - 1) for i in range(T)],
        dtype=rewards.dtype, device=device,
    )
    advantages = adv_k * discount_vec                              # (B, K, T)

    # ------------------------------------------------------------------
    # Assemble RolloutBatch
    # ------------------------------------------------------------------
    anchors_phys = denormalize(anchors_norm_d, q01, q99)           # (N_anchor, N_F, 3)
    return RolloutBatch(
        anchors              = anchors_phys.unsqueeze(0).expand(B, -1, -1, -1),
        z_t_per_step         = z_t_all,
        z_next_phys_per_step = z_next_phys_all,
        logp_old             = logp_old_all,
        rewards              = rewards,
        sub_rewards          = sub_rewards,
        advantages           = advantages,
        trajectories         = trajectories,
        gt_trajectory        = gt_phys,
    )
