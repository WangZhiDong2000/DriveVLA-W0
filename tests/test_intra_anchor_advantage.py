"""
Unit tests for utils/rl_modules/intra_anchor_advantage.py (Task 1.3).

Run:
    python -m pytest tests/test_intra_anchor_advantage.py -v

Pass criteria (v2.2 §1.3):
  1. Random rewards → per-anchor mean≈0, std≈1 after normalization
  2. Constant rewards within an anchor → advantages≈0, no NaN/Inf
  3. Output shape and dtype preserved
  4. Normalization is per-anchor (changes in one anchor don't affect others)
"""

import torch
import pytest

from utils.rl_modules.intra_anchor_advantage import compute_intra_anchor_advantages


def _make_rewards(B=4, N_anchor=20, G=8, seed=0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, N_anchor, G, generator=g)


# ---------------------------------------------------------------------------
# Test 1: mean≈0, std≈1 after normalization
# ---------------------------------------------------------------------------

def test_mean_zero_std_one():
    rewards = _make_rewards(B=4, N_anchor=20, G=8)
    adv = compute_intra_anchor_advantages(rewards, epsilon=1e-4)

    mean_per_anchor = adv.mean(dim=-1)   # (B, N_anchor)
    std_per_anchor  = adv.std(dim=-1)    # (B, N_anchor)

    assert torch.allclose(mean_per_anchor, torch.zeros_like(mean_per_anchor), atol=1e-5), \
        f"Per-anchor mean not ≈0; max|mean|={mean_per_anchor.abs().max().item():.2e}"
    # epsilon=1e-4 slightly deflates std: std_out ≈ std_actual/(std_actual+ε) < 1
    # For G=8 uniform rewards (std≈0.3), max deflation ≈ 1e-4/0.3 ≈ 3e-4 → use atol=2e-3
    assert torch.allclose(std_per_anchor, torch.ones_like(std_per_anchor), atol=2e-3), \
        f"Per-anchor std not ≈1; max|std-1|={(std_per_anchor - 1).abs().max().item():.2e}"


# ---------------------------------------------------------------------------
# Test 2: constant rewards within anchor → advantages≈0, no NaN/Inf
# ---------------------------------------------------------------------------

def test_zero_std_stability():
    B, N_anchor, G = 3, 5, 8
    rewards = torch.ones(B, N_anchor, G)   # all identical within each anchor

    adv = compute_intra_anchor_advantages(rewards, epsilon=1e-4)

    assert torch.isfinite(adv).all(), "NaN or Inf detected with zero-std rewards"
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6), \
        f"Expected advantages≈0 for constant rewards; max|adv|={adv.abs().max().item():.2e}"


# ---------------------------------------------------------------------------
# Test 3: shape and dtype preserved
# ---------------------------------------------------------------------------

def test_shape_preserved():
    for shape in [(2, 5, 8), (1, 20, 8), (8, 1, 8)]:
        rewards = torch.randn(*shape)
        adv = compute_intra_anchor_advantages(rewards)
        assert adv.shape == rewards.shape, f"Shape mismatch for input {shape}"
        assert adv.dtype == rewards.dtype, f"dtype changed for input {shape}"


# ---------------------------------------------------------------------------
# Test 4: per-anchor independence — modifying one anchor does not affect others
# ---------------------------------------------------------------------------

def test_per_anchor_independence():
    B, N_anchor, G = 2, 6, 8
    rewards = _make_rewards(B=B, N_anchor=N_anchor, G=G, seed=7)
    adv_ref = compute_intra_anchor_advantages(rewards.clone(), epsilon=1e-4)

    # Perturb anchor index 0 only
    rewards_perturbed = rewards.clone()
    rewards_perturbed[:, 0, :] += 10.0
    adv_perturbed = compute_intra_anchor_advantages(rewards_perturbed, epsilon=1e-4)

    # Anchors 1..N-1 should be unaffected
    assert torch.allclose(adv_ref[:, 1:, :], adv_perturbed[:, 1:, :], atol=1e-6), \
        "Modifying anchor 0 changed advantages at other anchors"
