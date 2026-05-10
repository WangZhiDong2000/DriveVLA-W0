"""
Stochastic ODE Sampler for DriveVLA-W0 GRPO (Task 2.5).

Single-step API decoupled from Emu3Pi0 so that the math is unit-testable
in isolation without instantiating the 7B VLA. Emu3Pi0.sample_actions_stochastic
wraps this around the velocity-prediction forward chain.

Step semantics (Plan_3 v2.2 §306-340, with §337 log_prob revised to DD-v2 z form):

    # 1) Euler mean in normalized space
    z_mean_norm = z_norm + dt * velo_pred                 # dt < 0  (t: 1 -> 0)

    # 2) Denorm -> multiplicative noise on (x, y) only -> renorm
    z_mean_phys = denormalize(z_mean_norm, q01, q99)
    z_next_phys, eps_xy = apply_multiplicative_noise(
        z_mean_phys, sigma=sigma_step, min_clip=0.04
    )                                                      # eps_xy: (*leading, 1, 2)
    eps_xy = eps_xy.clamp(-0.5, 0.5)
    z_next_norm = normalize(z_next_phys, q01, q99)

    # 3) DD-v2 form log_prob (per-step), summed over (waypoint, x/y) dims
    sigma_lp = max(sigma_step, 0.10)                       # min clip for log-prob stability
    diff = z_next_phys[..., :2].detach() - z_mean_phys[..., :2]
    log_prob = (-(diff**2) / (2*sigma_lp**2) - log(sigma_lp) - 0.5*log(2π)).sum(dim=(-2,-1))

Returns: (z_next_norm, log_prob, z_mean_norm, eps_xy)
  z_next_norm  : (*, N_F, 3)   — perturbed sample in normalized space; fed to next step
  log_prob     : (*,)          — per-sample summed log-density (DD-v2 z form)
  z_mean_norm  : (*, N_F, 3)   — deterministic Euler mean in normalized space
  eps_xy       : (*, 1, 2)     — clipped multiplicative noise |ε| ≤ 0.5

Design decisions (Plan_3 v2.2 §306-340):
- sigma_step=0.04 (noise sampling std) is separated from sigma_logprob_min=0.10
  (log-prob numerical stability floor). Mirrors DD-v2 reference lines 639/668.
- Log_prob computed only on (x, y) channels; heading is unchanged by noise.
- Single-step returns 4 tensors; multi-step trajectory is stacked by the caller.
- q01/q99 must be passed explicitly so this function is isolation-testable without
  instantiating Emu3Pi0. Callers on Emu3Pi0 pass self.action_q01 / self.action_q99.
"""

from __future__ import annotations
import math
import torch

from models.policy_head.anchored_flow_path import normalize, denormalize
from models.policy_head.multiplicative_noise import apply_multiplicative_noise


def _gaussian_log_prob_z(
    z_next_phys: torch.Tensor,
    z_mean_phys: torch.Tensor,
    sigma_lp: float,
) -> torch.Tensor:
    """Per-batch summed log N(z_next.detach()[..., :2] | z_mean[..., :2], σ_lp²·I).

    Shared by stochastic_euler_step (rollout) and recompute_log_prob (Pass-2) to
    guarantee bit-identical values, so IS ratio = exp(lp_new - lp_old.detach()) = 1
    exactly at unchanged θ.
    """
    diff = z_next_phys[..., :2].detach() - z_mean_phys[..., :2]  # (*, N_F, 2)
    return (
        -(diff ** 2) / (2.0 * sigma_lp ** 2)
        - math.log(sigma_lp)
        - 0.5 * math.log(2.0 * math.pi)
    ).sum(dim=(-2, -1))  # (*,)


def stochastic_euler_step(
    z_norm: torch.Tensor,
    velo_pred: torch.Tensor,
    dt: float,
    q01: torch.Tensor,
    q99: torch.Tensor,
    sigma_step: float = 0.04,
    sigma_logprob_min: float = 0.10,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single stochastic Euler step with physical-space multiplicative noise.

    Args:
        z_norm:           (*, N_F, 3) current trajectory in normalized [-1, 1] space.
        velo_pred:        (*, N_F, 3) velocity predicted by the action expert.
        dt:               negative scalar step size, e.g. -1/T_trunc.
        q01:              (3,) physical lower-bound normalization stats.
        q99:              (3,) physical upper-bound normalization stats.
        sigma_step:       multiplicative noise std for sampling; min_clip=0.04 enforced.
        sigma_logprob_min: minimum σ used for log-prob to avoid numerical blow-up (≥ 0.10).
        generator:        optional RNG for reproducibility.

    Returns:
        z_next_norm  : (*, N_F, 3)  — perturbed normalized sample (input to next step).
        log_prob     : (*,)         — per-sample summed log π (DD-v2 form, x/y channels only).
        z_mean_norm  : (*, N_F, 3)  — deterministic Euler mean in normalized space.
        eps_xy       : (*, 1, 2)    — clipped multiplicative noise ε ∈ [-0.5, 0.5].
    """
    # 1) Euler mean in normalized space
    z_mean_norm = z_norm + dt * velo_pred

    # 2) Denorm → multiplicative noise on (x, y) only → renorm
    _q01 = q01.to(dtype=z_mean_norm.dtype, device=z_mean_norm.device)
    _q99 = q99.to(dtype=z_mean_norm.dtype, device=z_mean_norm.device)
    z_mean_phys = denormalize(z_mean_norm, _q01, _q99)
    z_next_phys, eps_xy = apply_multiplicative_noise(
        z_mean_phys, sigma=sigma_step, min_clip=0.04, generator=generator,
    )
    eps_xy = eps_xy.clamp(-0.5, 0.5)
    # Re-apply clamp to z_next_phys with the clamped eps (eps was clamp-modified after sampling)
    z_next_phys = torch.cat(
        [z_mean_phys[..., :2] * (1.0 + eps_xy), z_mean_phys[..., 2:3]], dim=-1
    )
    z_next_norm = normalize(z_next_phys, _q01, _q99)

    # 3) DD-v2 z form log_prob on (x, y) channels only
    sigma_lp = max(float(sigma_step), float(sigma_logprob_min))
    log_prob = _gaussian_log_prob_z(z_next_phys, z_mean_phys, sigma_lp)

    return z_next_norm, log_prob, z_mean_norm, eps_xy


def recompute_log_prob(
    z_t_norm: torch.Tensor,
    velo_pred: torch.Tensor,
    z_next_phys_stored: torch.Tensor,
    dt: float,
    q01: torch.Tensor,
    q99: torch.Tensor,
    sigma_step: float = 0.04,
    sigma_logprob_min: float = 0.10,
) -> torch.Tensor:
    """Pass-2 recompute: log π_θ(z_{t-1} | z_t) on stored z_{t-1} with fresh velo_pred(θ).

    For GRPO REINFORCE, Pass 1 (rollout) stores (z_t, z_{t-1}_phys) under no_grad.
    Pass 2 (loss) calls this function with a fresh forward's velo_pred(θ) to get
    log_p_new that carries gradient, so the IS ratio exp(log_p_new - log_p_old.detach())
    is non-trivial after a θ update.

    Args:
        z_t_norm:           (*, N_F, 3) stored z_t in normalized space (input to that step).
        velo_pred:          (*, N_F, 3) FRESH velocity from current θ forward pass.
        z_next_phys_stored: (*, N_F, 3) stored z_{t-1} in physical space (detached inside helper).
        dt:                 same negative scalar used in rollout.
        q01, q99:           (3,) normalization bounds (same as Emu3Pi0.action_q01/q99).
        sigma_step:         sampling σ (determines σ_lp floor with sigma_logprob_min).
        sigma_logprob_min:  log-prob σ floor for numerical stability (≥ 0.10).

    Returns:
        log_prob : (*,) — per-batch summed log π, gradient flows through velo_pred only.

    Numerical guarantee: bit-identical to stochastic_euler_step's log_prob when called
    with the rollout's stored tensors and same velo_pred, because both use _gaussian_log_prob_z.
    """
    _q01 = q01.to(dtype=z_t_norm.dtype, device=z_t_norm.device)
    _q99 = q99.to(dtype=z_t_norm.dtype, device=z_t_norm.device)
    z_mean_norm = z_t_norm + dt * velo_pred
    z_mean_phys = denormalize(z_mean_norm, _q01, _q99)
    sigma_lp = max(float(sigma_step), float(sigma_logprob_min))
    return _gaussian_log_prob_z(z_next_phys_stored, z_mean_phys, sigma_lp)
