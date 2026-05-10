"""
Task 2.6 — Log Policy recompute tests (Pass-2 GRPO building block).

Tests cover:
  1. On-policy bit-identity: Pass-1 stochastic_euler_step log_prob == Pass-2
     recompute_log_prob(z_t, velo_pred_same, z_next_phys_same).
  2. Grad flows through velo_pred only; z_next_phys_stored.grad stays None/zero.
  3. IS ratio at θ unchanged: exp(lp_new - lp_old.detach()) ≈ 1.
  4. Sensitivity: shifting velo_pred away from rollout → log_prob decreases (Gaussian).
  5. Sigma floor: sigma_step=0.0 → sigma_lp = max(0, 0.10) = 0.10 → same as sigma_step=0.10.

Run:
    conda run -n drivevla python -m pytest tests/test_log_policy_recompute.py -v
"""

import torch

from models.policy_head.stochastic_ode_sampler import (
    stochastic_euler_step,
    recompute_log_prob,
)
from models.policy_head.anchored_flow_path import denormalize

Q01 = torch.tensor([-0.0179, -0.1909, -0.1892], dtype=torch.float32)
Q99 = torch.tensor([6.1996,  0.2426,  0.1805], dtype=torch.float32)
B, N_F = 4, 8
DT = -1.0 / 10


def _rollout(seed):
    g_in = torch.Generator().manual_seed(seed)
    z = torch.empty(B, N_F, 3).uniform_(-0.5, 0.5, generator=g_in)
    v = torch.empty(B, N_F, 3).uniform_(-0.5, 0.5, generator=g_in)
    g_s = torch.Generator().manual_seed(seed + 1000)
    z_next_norm, lp, z_mean_norm, _ = stochastic_euler_step(
        z, v, DT, Q01, Q99, generator=g_s,
    )
    z_next_phys = denormalize(z_next_norm, Q01, Q99)
    return z, v, z_next_norm, lp, z_mean_norm, z_next_phys


def test_recompute_matches_rollout_on_policy():
    """Pass-1 lp == Pass-2 lp when called with same (z_t, velo_pred, z_next_phys)."""
    z, v, _, lp_rollout, _, z_next_phys = _rollout(seed=0)
    lp_re = recompute_log_prob(z, v, z_next_phys, DT, Q01, Q99)
    assert torch.allclose(lp_rollout, lp_re, atol=1e-6), (
        f"max diff {(lp_rollout - lp_re).abs().max():.2e}"
    )


def test_recompute_grad_flows_through_velo_pred_only():
    """Grad reaches velo_pred; z_next_phys_stored.grad stays None or zero."""
    z, _, _, _, _, z_next_phys = _rollout(seed=1)
    v = torch.randn(B, N_F, 3, requires_grad=True)
    z_next_leaf = z_next_phys.detach().clone().requires_grad_(True)
    lp = recompute_log_prob(z, v, z_next_leaf, DT, Q01, Q99)
    lp.sum().backward()
    assert v.grad is not None and v.grad.abs().max() > 0, "no grad through velo_pred"
    assert torch.isfinite(v.grad).all(), "NaN/Inf in velo_pred.grad"
    assert z_next_leaf.grad is None or z_next_leaf.grad.abs().max() == 0, (
        "z_next_phys_stored must not receive gradient"
    )


def test_is_ratio_one_at_init():
    """At θ unchanged: exp(lp_new - lp_old.detach()) ≈ 1.0 (bit-identical by construction)."""
    z, v, _, lp_rollout, _, z_next_phys = _rollout(seed=2)
    lp_new = recompute_log_prob(z, v, z_next_phys, DT, Q01, Q99)
    is_ratio = torch.exp(lp_new - lp_rollout.detach())
    assert torch.allclose(is_ratio, torch.ones_like(is_ratio), atol=1e-5), (
        f"IS ratio drift: max |1 - r| = {(is_ratio - 1).abs().max():.2e}"
    )


def test_log_prob_decreases_when_velo_drifts():
    """Larger residual (z_next - z_mean) → smaller log_prob (Gaussian density is penalized)."""
    z, v, _, _, _, z_next_phys = _rollout(seed=3)
    lp_base = recompute_log_prob(z, v, z_next_phys, DT, Q01, Q99)
    v_jit = v + 0.5 * torch.ones_like(v)
    lp_jit = recompute_log_prob(z, v_jit, z_next_phys, DT, Q01, Q99)
    assert (lp_jit <= lp_base + 1e-4).all(), (
        f"log_prob should not increase when velo drifts; diff max={( lp_jit - lp_base).max():.4f}"
    )


def test_sigma_floor_enforced():
    """sigma_step=0.0 → sigma_lp = max(0.0, 0.10) = 0.10 → same result as sigma_step=0.10."""
    z, v, _, _, _, z_next_phys = _rollout(seed=4)
    lp_zero = recompute_log_prob(z, v, z_next_phys, DT, Q01, Q99, sigma_step=0.0)
    lp_010  = recompute_log_prob(z, v, z_next_phys, DT, Q01, Q99, sigma_step=0.10)
    assert torch.isfinite(lp_zero).all(), "sigma floor not protecting against σ_step=0"
    assert torch.allclose(lp_zero, lp_010, atol=1e-6), (
        f"sigma floor mismatch: max diff {(lp_zero - lp_010).abs().max():.2e}"
    )
