"""
Task 2.5 — Stochastic ODE Sampler isolation tests (single-step).

Tests cover Plan_3 v2.2 §331 Pass criteria + log_prob sanity:
  1. eps_xy clamp: at min noise, |eps_xy| <= 0.5 is enforced.
  2. Reproducibility: same generator seed → bit-identical outputs.
  3. Monte Carlo mean convergence: 50-trial mean of z_next → z_mean (loose tol 1e-1).
  4. Heading channel unchanged: z_next[..., 2] == z_mean[..., 2] (atol 1e-5).
  5. eps_xy shape (B, 1, 2); |eps_xy| <= 0.5 even with large sigma_step.
  6. log_prob shape (B,); finite for 100 random batches.
  7. Gradient flows from log_prob through velo_pred into z_mean (REINFORCE path).

Run:
    conda run -n drivevla python -m pytest tests/test_stochastic_ode_sampler.py -v
"""

import math
import torch

from models.policy_head.stochastic_ode_sampler import stochastic_euler_step

# NAVSIM stats from configs/normalizer_navsim_trainval/norm_stats.json
Q01 = torch.tensor([-0.0179, -0.1909, -0.1892], dtype=torch.float32)
Q99 = torch.tensor([6.1996,  0.2426,  0.1805], dtype=torch.float32)
B, N_F = 4, 8
DT = -1.0 / 10  # T_trunc=10


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.empty(B, N_F, 3).uniform_(-0.5, 0.5, generator=g)
    v = torch.empty(B, N_F, 3).uniform_(-0.5, 0.5, generator=g)
    return z, v


def test_eps_clamp_enforced():
    """At default sigma_step=0.04 min noise, |eps_xy| <= 0.5 must hold."""
    z, v = _inputs(0)
    g = torch.Generator().manual_seed(7)
    _, _, _, eps = stochastic_euler_step(z, v, DT, Q01, Q99, sigma_step=0.04, generator=g)
    assert (eps.abs() <= 0.5).all(), f"eps clamp violated: max={eps.abs().max():.4f}"


def test_reproducibility_same_seed():
    """Same generator seed must produce bit-identical outputs."""
    z, v = _inputs(0)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    z1, lp1, zm1, e1 = stochastic_euler_step(z, v, DT, Q01, Q99, generator=g1)
    z2, lp2, zm2, e2 = stochastic_euler_step(z, v, DT, Q01, Q99, generator=g2)
    assert torch.equal(z1, z2), "z_next not bit-identical for same seed"
    assert torch.equal(lp1, lp2), "log_prob not bit-identical for same seed"
    assert torch.equal(e1, e2), "eps_xy not bit-identical for same seed"


def test_monte_carlo_mean_converges_to_z_mean():
    """Monte Carlo mean of 50 z_next samples should be close to z_mean (tol 1e-1)."""
    z, v = _inputs(1)
    N = 50
    accum = torch.zeros(B, N_F, 3)
    z_mean_ref = None
    for s in range(N):
        g = torch.Generator().manual_seed(s)
        z_next, _, z_mean, _ = stochastic_euler_step(z, v, DT, Q01, Q99, generator=g)
        accum += z_next
        z_mean_ref = z_mean
    mc_mean = accum / N
    diff = (mc_mean - z_mean_ref).abs().max().item()
    assert diff < 1e-1, f"MC mean diverges from z_mean: {diff:.3e}"


def test_heading_unchanged():
    """Heading channel (idx 2) of z_next must equal z_mean heading (multiplicative
    noise only affects (x, y), and heading renorm round-trip is identity)."""
    z, v = _inputs(2)
    g = torch.Generator().manual_seed(3)
    z_next, _, z_mean, _ = stochastic_euler_step(z, v, DT, Q01, Q99, generator=g)
    assert torch.allclose(z_next[..., 2], z_mean[..., 2], atol=1e-5), (
        f"Heading drifted: max diff {(z_next[..., 2] - z_mean[..., 2]).abs().max():.2e}"
    )


def test_eps_shape_and_clamp_large_sigma():
    """eps_xy shape must be (B, 1, 2) and |eps_xy| <= 0.5 even for large sigma_step."""
    z, v = _inputs(3)
    g = torch.Generator().manual_seed(4)
    _, _, _, eps = stochastic_euler_step(z, v, DT, Q01, Q99, sigma_step=0.5, generator=g)
    assert eps.shape == (B, 1, 2), f"eps shape {eps.shape} != ({B}, 1, 2)"
    assert (eps.abs() <= 0.5).all(), f"eps clamp violated: max={eps.abs().max():.4f}"


def test_log_prob_finite_100_batches():
    """log_prob must be shape (B,) and finite for 100 random batches."""
    for s in range(100):
        z, v = _inputs(s)
        g = torch.Generator().manual_seed(s + 100)
        _, lp, _, _ = stochastic_euler_step(z, v, DT, Q01, Q99, generator=g)
        assert lp.shape == (B,), f"log_prob shape {lp.shape} at seed {s}"
        assert torch.isfinite(lp).all(), f"log_prob has NaN/Inf at seed {s}"


def test_grad_flow_through_z_mean():
    """Gradient must flow from log_prob through velo_pred (REINFORCE path via z_mean)."""
    z = torch.randn(B, N_F, 3)
    v = torch.randn(B, N_F, 3, requires_grad=True)
    g = torch.Generator().manual_seed(11)
    _, lp, _, _ = stochastic_euler_step(z, v, DT, Q01, Q99, generator=g)
    lp.sum().backward()
    assert v.grad is not None, "No gradient on velo_pred"
    assert torch.isfinite(v.grad).all(), "NaN/Inf in velo_pred gradient"
    assert v.grad.abs().max() > 0, "Gradient is zero — REINFORCE path broken"
