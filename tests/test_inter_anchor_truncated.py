"""
Unit tests for utils/rl_modules/inter_anchor_truncated.py (Task 1.4).

Run:
    python -m pytest tests/test_inter_anchor_truncated.py -v

Pass criteria (v2.2 §1.4):
  1. Positive advantage + reward > GT → kept unchanged
  2. Negative advantage + reward > GT → clamped to 0
  3. Any advantage + reward ≤ GT → output 0
  4. Collision (no_collision=0) → output -1 regardless
  5. Off-road (drivable_area=0) → output -1 regardless
  6. Collision + reward < GT → output -1 (collision overrides GT-mask zeroing)
"""

import torch
import pytest

from utils.rl_modules.inter_anchor_truncated import apply_inter_anchor_truncation


def _ones(B=1, N=3, G=4):
    """Helper: all-ones float tensor of given shape."""
    return torch.ones(B, N, G)


def _call(adv, rew, gt, nc=None, da=None):
    """Convenience wrapper with default safe masks."""
    B = adv.shape[0]
    if nc is None:
        nc = _ones(*adv.shape)
    if da is None:
        da = _ones(*adv.shape)
    return apply_inter_anchor_truncation(adv, rew, gt, nc, da)


# ---------------------------------------------------------------------------
# Test 1: positive adv + reward > GT → kept
# ---------------------------------------------------------------------------

def test_positive_better_than_gt():
    adv = torch.tensor([[[0.5, 1.2, 0.3]]])   # (1, 1, 3) all positive
    rew = torch.tensor([[[0.8, 0.9, 0.7]]])
    gt  = torch.tensor([0.5])                  # all rewards > 0.5

    out = _call(adv, rew, gt)
    assert torch.allclose(out, adv, atol=1e-7), \
        f"Positive adv better than GT should be unchanged; got {out}"


# ---------------------------------------------------------------------------
# Test 2: negative adv + reward > GT → clamped to 0
# ---------------------------------------------------------------------------

def test_negative_clamped_to_zero():
    adv = torch.tensor([[[-0.5, -1.2]]])
    rew = torch.tensor([[[0.8, 0.9]]])
    gt  = torch.tensor([0.5])

    out = _call(adv, rew, gt)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7), \
        f"Negative adv should be clamped to 0; got {out}"


# ---------------------------------------------------------------------------
# Test 3: reward ≤ GT → output 0 even if adv is positive
# ---------------------------------------------------------------------------

def test_worse_than_gt_zeroed():
    # gt_threshold=1e-6: reward qualifies if reward > gt - 1e-6 = 0.499999.
    # Use rewards clearly below this threshold.
    adv = torch.tensor([[[1.0, 0.8]]])
    rew = torch.tensor([[[0.3, 0.49]]])  # both < 0.499999 → masked out
    gt  = torch.tensor([0.5])

    out = _call(adv, rew, gt)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7), \
        f"Rewards ≤ GT should yield 0; got {out}"


# ---------------------------------------------------------------------------
# Test 4: collision (no_collision=0) → -1
# ---------------------------------------------------------------------------

def test_collision_gives_minus_one():
    adv = torch.tensor([[[0.5, -0.3, 0.0]]])
    rew = torch.tensor([[[0.9,  0.8, 0.7]]])
    gt  = torch.tensor([0.5])
    nc  = torch.zeros(1, 1, 3)   # all collision

    out = apply_inter_anchor_truncation(adv, rew, gt, nc, _ones(1, 1, 3))
    assert torch.allclose(out, torch.full_like(out, -1.0), atol=1e-7), \
        f"All collision → expected all -1; got {out}"


# ---------------------------------------------------------------------------
# Test 5: off-road (drivable_area=0) → -1
# ---------------------------------------------------------------------------

def test_drivable_area_gives_minus_one():
    adv = torch.tensor([[[0.7, 0.2]]])
    rew = torch.tensor([[[0.9, 0.8]]])
    gt  = torch.tensor([0.5])
    da  = torch.zeros(1, 1, 2)   # all off-road

    out = apply_inter_anchor_truncation(adv, rew, gt, _ones(1, 1, 2), da)
    assert torch.allclose(out, torch.full_like(out, -1.0), atol=1e-7), \
        f"All off-road → expected all -1; got {out}"


# ---------------------------------------------------------------------------
# Test 6: collision + reward < GT → -1 (not 0 from GT mask)
# ---------------------------------------------------------------------------

def test_collision_overrides_clamp():
    # reward=0.3 < gt=0.5, so GT-mask alone would → 0
    # but no_collision=0 (collision) should override to -1
    adv = torch.tensor([[[0.5]]])
    rew = torch.tensor([[[0.3]]])
    gt  = torch.tensor([0.5])
    nc  = torch.zeros(1, 1, 1)   # collision
    da  = _ones(1, 1, 1)

    out = apply_inter_anchor_truncation(adv, rew, gt, nc, da)
    assert torch.allclose(out, torch.full_like(out, -1.0), atol=1e-7), \
        f"Collision should override GT-mask zeroing to -1; got {out}"
