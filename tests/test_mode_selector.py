"""
Unit tests for models/policy_head/mode_selector.py (Task 1.7).

Phase 1 skeleton pass criteria (v2.2 §1.7):
  1. forward returns (B, K) float32
  2. Output is finite (no NaN/Inf)
  3. Gradients flow to both trajectories and vlm_h inputs
  4. K can be any positive integer (1 / 8 / 20 / 160)

All tests use random tensors — no real model, no data files required.

Run:
    conda run -n drivevla python -m pytest tests/test_mode_selector.py -v
"""

import torch
import pytest

from models.policy_head.mode_selector import ModeSelector


B, N_F, H_VLM, S = 2, 8, 4096, 16   # S=16 keeps CPU tests fast


# ---------------------------------------------------------------------------
# Test 1: forward output shape
# ---------------------------------------------------------------------------

def test_forward_shape():
    """ModeSelector.forward must return (B, K) float32 logits."""
    K = 20
    sel = ModeSelector(n_waypoints=N_F, vlm_hidden=H_VLM)
    traj  = torch.randn(B, K, N_F, 3)
    vlm_h = torch.randn(B, S, H_VLM)

    logits = sel(traj, vlm_h)

    assert logits.shape == (B, K), \
        f"Expected ({B}, {K}), got {logits.shape}"
    assert logits.dtype == torch.float32, \
        f"Expected float32, got {logits.dtype}"


# ---------------------------------------------------------------------------
# Test 2: finite values
# ---------------------------------------------------------------------------

def test_finite():
    """logits must not contain NaN or Inf."""
    K = 20
    sel   = ModeSelector(n_waypoints=N_F, vlm_hidden=H_VLM)
    traj  = torch.randn(B, K, N_F, 3)
    vlm_h = torch.randn(B, S, H_VLM)

    logits = sel(traj, vlm_h)

    bad = ~torch.isfinite(logits)
    assert not bad.any(), \
        f"logits has {bad.sum()} non-finite values at {bad.nonzero()[:5]}"


# ---------------------------------------------------------------------------
# Test 3: gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow():
    """Gradient must flow back to both trajectories and vlm_h."""
    K = 20
    sel   = ModeSelector(n_waypoints=N_F, vlm_hidden=H_VLM)
    traj  = torch.randn(B, K, N_F, 3, requires_grad=True)
    vlm_h = torch.randn(B, S, H_VLM, requires_grad=True)

    logits = sel(traj, vlm_h)
    logits.sum().backward()

    assert traj.grad is not None, "No gradient to trajectories"
    assert torch.isfinite(traj.grad).all(), "NaN/Inf in trajectories.grad"

    assert vlm_h.grad is not None, "No gradient to vlm_h"
    assert torch.isfinite(vlm_h.grad).all(), "NaN/Inf in vlm_h.grad"


# ---------------------------------------------------------------------------
# Test 4: arbitrary K (inference=20 vs training=160 vs edge cases)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", [1, 8, 20, 160])
def test_arbitrary_K(K):
    """K must not be hard-coded: selector works for any positive integer."""
    sel   = ModeSelector(n_waypoints=N_F, vlm_hidden=H_VLM)
    traj  = torch.randn(B, K, N_F, 3)
    vlm_h = torch.randn(B, S, H_VLM)

    logits = sel(traj, vlm_h)

    assert logits.shape == (B, K), \
        f"K={K}: expected ({B}, {K}), got {logits.shape}"
