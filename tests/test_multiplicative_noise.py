"""
Unit tests for models/policy_head/multiplicative_noise.py (Task 1.2).

Run:
    python -m pytest tests/test_multiplicative_noise.py -v

Pass criteria (v2.2 §1.2):
  1. σ=0 + min_clip=0  →  output identical to input, ε == 0
  2. Heading channel is bit-exact unchanged even at large σ
  3. eps_mul shape is (*, 1, 2) and broadcasts correctly; different (b,k) get independent ε
  4. min_clip overrides small σ — effective std ≈ min_clip, not the raw σ
  5. Same generator seed → identical eps_mul (reproducibility)
"""

import math

import torch
import pytest

from models.policy_head.multiplicative_noise import apply_multiplicative_noise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_traj(*leading, N_f=8, seed=42) -> torch.Tensor:
    """Random physical-space trajectory tensor with given leading dims."""
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(*leading, N_f, 3, generator=g)
    # Make (x, y) positive and reasonably large (meters) to stress-test multiplicative effect
    t[..., :2] = t[..., :2].abs() * 5.0 + 0.5
    t[..., 2] = t[..., 2] * 0.3   # heading in radians
    return t


# ---------------------------------------------------------------------------
# Test 1: σ=0 with min_clip=0 is a strict identity
# ---------------------------------------------------------------------------

def test_sigma_zero_is_identity():
    traj = _make_traj(4, N_f=8)   # leading=(4,), traj shape (4, 8, 3)
    noisy, eps = apply_multiplicative_noise(traj, sigma=0.0, min_clip=0.0)

    assert noisy.shape == traj.shape
    assert eps.shape == (4, 1, 2)
    assert torch.allclose(eps, torch.zeros_like(eps), atol=1e-7), \
        f"Expected eps==0 but got max|eps|={eps.abs().max().item():.2e}"
    assert torch.allclose(noisy, traj, atol=1e-7), \
        f"Expected noisy==traj but max diff={( noisy - traj).abs().max().item():.2e}"


# ---------------------------------------------------------------------------
# Test 2: Heading channel is bit-exact unchanged at large σ
# ---------------------------------------------------------------------------

def test_heading_channel_unchanged():
    traj = _make_traj(3, N_f=8)   # leading dim (3,), traj shape (3, 8, 3)
    noisy, _ = apply_multiplicative_noise(traj, sigma=0.16, min_clip=0.04)

    # Heading (channel index 2) must survive unchanged — it's a split-cat, no arithmetic
    assert torch.equal(noisy[..., 2], traj[..., 2]), \
        "Heading channel was modified by multiplicative noise"


# ---------------------------------------------------------------------------
# Test 3: eps_mul shape, broadcast consistency, inter-sample independence
# ---------------------------------------------------------------------------

def test_eps_shape_and_broadcast():
    B, K, N_f = 4, 20, 8
    sigma = 0.04
    traj = _make_traj(B, K, N_f=N_f)

    noisy, eps = apply_multiplicative_noise(traj, sigma=sigma, min_clip=0.0)

    # Shape check
    assert eps.shape == (B, K, 1, 2), \
        f"Expected eps shape ({B}, {K}, 1, 2), got {tuple(eps.shape)}"
    assert noisy.shape == (B, K, N_f, 3)

    # All waypoints in a given (b, k) slice share the same ε — verify via reconstruction
    # noisy[b, k, :, 0] == traj[b, k, :, 0] * (1 + eps[b, k, 0, 0])
    for b in range(B):
        for k in range(K):
            scale_x = 1.0 + eps[b, k, 0, 0]
            expected_x = traj[b, k, :, 0] * scale_x
            assert torch.allclose(noisy[b, k, :, 0], expected_x, atol=1e-6), \
                f"Waypoints do not share ε at (b={b}, k={k})"

    # Different (b, k) pairs have independent ε — std across all (B*K) samples ≈ sigma
    eps_flat = eps.view(-1)   # B*K*2 scalars
    std = eps_flat.std().item()
    assert abs(std - sigma) < 0.02, \
        f"eps std={std:.4f} too far from sigma={sigma} (N={eps_flat.numel()})"


# ---------------------------------------------------------------------------
# Test 4: min_clip overrides small σ
# ---------------------------------------------------------------------------

def test_min_clip_applied():
    sigma_raw = 0.001
    min_clip = 0.04
    B = 2000   # large sample to measure std accurately

    traj = _make_traj(B, N_f=8)   # leading=(B,), shape (B, 8, 3)
    _, eps = apply_multiplicative_noise(traj, sigma=sigma_raw, min_clip=min_clip)

    # eps shape: (B, 1, 2); flatten to measure std
    eps_flat = eps.view(-1).float()
    std = eps_flat.std().item()

    # Effective σ should be min_clip (0.04), not sigma_raw (0.001)
    assert abs(std - min_clip) < 0.005, \
        f"Expected effective std≈{min_clip} (clip applied), got std={std:.4f}"

    # Confirm raw sigma would have been distinguishably smaller
    assert abs(std - sigma_raw) > 0.02, \
        "Effective std is too close to raw sigma — clip may not have activated"


# ---------------------------------------------------------------------------
# Test 5: Same generator seed → identical eps_mul
# ---------------------------------------------------------------------------

def test_generator_reproducibility():
    traj = _make_traj(4, N_f=8)   # leading=(4,)
    seed = 12345

    g1 = torch.Generator().manual_seed(seed)
    _, eps1 = apply_multiplicative_noise(traj, sigma=0.04, generator=g1)

    g2 = torch.Generator().manual_seed(seed)
    _, eps2 = apply_multiplicative_noise(traj, sigma=0.04, generator=g2)

    assert torch.equal(eps1, eps2), "Same generator seed produced different eps_mul"

    # Sanity: different seed → different eps
    g3 = torch.Generator().manual_seed(seed + 1)
    _, eps3 = apply_multiplicative_noise(traj, sigma=0.04, generator=g3)
    assert not torch.equal(eps1, eps3), "Different seeds produced identical eps_mul (unexpected)"
