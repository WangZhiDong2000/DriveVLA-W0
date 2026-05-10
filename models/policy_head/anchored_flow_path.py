"""
Anchored Flow Path for DriveVLA-W0 GRPO (Task 2.3).

Replaces the pure-Gaussian flow endpoint z0 ~ N(0, I) with an
anchor-perturbed endpoint x1 = renorm(denorm(anchor) ⊙ (1 + ε_mul)),
with multiplicative noise applied in PHYSICAL space on (x, y) only
(heading passes through unchanged) — Plan_3 v2.2 §0.2 #10.

Convention (matches Emu3Pi0.forward + noise_schedulers.py):
    x_t    = (1 - t) · tau_GT + t · x1    # t=0 → tau_GT (data); t=1 → x1 (prior)
    target = x1 − tau_GT                  # velocity field; same sign as legacy
                                          # `noise − action` so MSE drop-in.

x1 replaces what was `noise = randn_like(action)` in the pure-noise path.
The perturbed anchor endpoint is constructed in physical space and renormalized,
so the FM model learns to predict the velocity from the anchor neighborhood to GT.

Note on norm_stats: configs/normalizer_navsim_trainval/norm_stats.json
top-level data key is "libero" by historical naming convention; the
numeric values are NAVSIM dataset stats. Caller is responsible for
loading q01/q99 — this module only consumes tensors.
"""

from __future__ import annotations

import torch

from models.policy_head.multiplicative_noise import apply_multiplicative_noise


def denormalize(x_norm: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    """[-1, 1] normalized → physical (m / rad)."""
    return (x_norm + 1.0) / 2.0 * (q99 - q01) + q01


def normalize(x_phys: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    """Physical (m / rad) → [-1, 1] normalized."""
    return (x_phys - q01) / (q99 - q01) * 2.0 - 1.0


def build_anchored_flow_path(
    anchor: torch.Tensor,
    tau_GT: torch.Tensor,
    t: torch.Tensor,
    q01: torch.Tensor,
    q99: torch.Tensor,
    sigma_anchor: float = 0.04,
    eps_abs_clip: float | None = 0.5,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct anchored linear-interp flow path.

    Args:
        anchor:       (B, N_f, 3) normalized [-1, 1] — cluster center anchor trajectory.
        tau_GT:       (B, N_f, 3) normalized [-1, 1] — ground-truth trajectory.
        t:            (B,) ∈ [0, 1] — flow timesteps from rf.sample_t().
        q01:          (3,) physical lower bound for denorm/renorm.
        q99:          (3,) physical upper bound for denorm/renorm.
        sigma_anchor: Std of multiplicative noise N(0, σ²) on (x, y). Default 0.04
                      (DD-v2 §3.1). Caller controls schedule; pass 0.0 for pure-interp.
        eps_abs_clip: Hard clip on ε values to prevent (1+ε) sign flip. None = no clip.
                      Needed when sigma_anchor is large (e.g. Stage 2-A init 0.5).
                      Distinct from apply_multiplicative_noise's min_clip (a σ floor).
        generator:    Optional torch.Generator for reproducible sampling in tests.

    Returns:
        x_t:     (B, N_f, 3) normalized — interpolated input for action_projector.
        target:  (B, N_f, 3) normalized — MSE target for v_φ (= x1 − tau_GT).
        x1_norm: (B, N_f, 3) normalized — perturbed anchor endpoint (Task 2.5 reuse).
    """
    q01 = q01.to(anchor.dtype)
    q99 = q99.to(anchor.dtype)

    # 1) denorm anchor to physical space
    anchor_phys = denormalize(anchor, q01, q99)

    # 2) multiplicative noise in physical space, (x, y) only; heading passes through
    x1_phys, eps_xy = apply_multiplicative_noise(
        anchor_phys, sigma=sigma_anchor, min_clip=0.0, generator=generator,
    )

    # 2b) ε hard-clip: safety net against sign flip for large σ_anchor (Stage 2-A)
    # Distinct from min_clip (σ floor in apply_multiplicative_noise).
    if eps_abs_clip is not None:
        eps_xy_clipped = eps_xy.clamp(-eps_abs_clip, eps_abs_clip)
        xy_noisy = anchor_phys[..., :2] * (1.0 + eps_xy_clipped)
        x1_phys = torch.cat([xy_noisy, anchor_phys[..., 2:3]], dim=-1)

    # 3) renorm back to [-1, 1]
    x1 = normalize(x1_phys, q01, q99)

    # 4) linear interpolation in normalized space (same convention as noise_schedulers.py)
    t_b = t.view(-1, 1, 1).to(x1.dtype)
    x_t = (1.0 - t_b) * tau_GT + t_b * x1
    target = x1 - tau_GT
    return x_t, target, x1
