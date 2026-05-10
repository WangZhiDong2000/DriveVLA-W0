"""
Task 2.3 — Anchored Flow Path isolation tests.

Tests cover (Plan_3 v2.2 §291 Pass criteria + grad flow):
  1. σ_anchor=0 → x_t == (1-t)·tau_GT + t·anchor (renorm round-trip < 1e-5).
  2. t=0 → x_t == tau_GT exactly; t=1 → x_t == x1 exactly (< 1e-6).
  3. Heading channel at t=1 equals anchor heading in normalized space.
  4. Lateral channel at σ=0.04 produces non-zero normalized perturbation.
  5. Gradient flows through tau_GT into x_t and target.

Run:
    conda run -n drivevla python -m pytest tests/test_anchored_flow_path.py -v
"""

import torch
import pytest

from models.policy_head.anchored_flow_path import (
    build_anchored_flow_path,
    denormalize,
    normalize,
)

# NAVSIM stats from configs/normalizer_navsim_trainval/norm_stats.json
# Key "libero" is historical naming; values are NAVSIM dataset stats.
Q01 = torch.tensor([-0.0179, -0.1909, -0.1892], dtype=torch.float32)
Q99 = torch.tensor([6.1996,  0.2426,  0.1805], dtype=torch.float32)
B, N_F = 4, 8


def _dummy_inputs(seed=0):
    torch.manual_seed(seed)
    anchor = torch.empty(B, N_F, 3).uniform_(-1, 1)
    tau_GT = torch.empty(B, N_F, 3).uniform_(-1, 1)
    return anchor, tau_GT


# ---------------------------------------------------------------------------
# Test 1: σ=0 → pure linear interp (renorm round-trip identity)
# ---------------------------------------------------------------------------

def test_sigma_zero_matches_pure_interp():
    """σ=0 → x_t == (1-t)·tau_GT + t·anchor (renorm round-trip < 1e-5)."""
    anchor, tau_GT = _dummy_inputs(0)
    t = torch.rand(B)
    x_t, target, x1 = build_anchored_flow_path(
        anchor, tau_GT, t, Q01, Q99, sigma_anchor=0.0, eps_abs_clip=None,
    )
    assert torch.allclose(x1, anchor, atol=1e-5), \
        f"renorm round-trip failed: max diff {(x1 - anchor).abs().max():.2e}"
    expected = (1 - t.view(-1, 1, 1)) * tau_GT + t.view(-1, 1, 1) * anchor
    assert torch.allclose(x_t, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 2: endpoint conditions (t=0 and t=1)
# ---------------------------------------------------------------------------

def test_endpoint_t0_t1():
    """t=0 → x_t == tau_GT; t=1 → x_t == x1."""
    anchor, tau_GT = _dummy_inputs(1)
    t0 = torch.zeros(B)
    t1 = torch.ones(B)
    x_t0, _, _  = build_anchored_flow_path(anchor, tau_GT, t0, Q01, Q99, sigma_anchor=0.04)
    x_t1, _, x1 = build_anchored_flow_path(anchor, tau_GT, t1, Q01, Q99, sigma_anchor=0.04)
    assert torch.allclose(x_t0, tau_GT, atol=1e-6)
    assert torch.allclose(x_t1, x1, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 3: heading channel unchanged at t=1
# ---------------------------------------------------------------------------

def test_heading_unchanged_at_t1():
    """At t=1, heading (channel 2) of x_t equals heading of anchor in norm space."""
    anchor, tau_GT = _dummy_inputs(2)
    t1 = torch.ones(B)
    x_t1, _, x1 = build_anchored_flow_path(anchor, tau_GT, t1, Q01, Q99, sigma_anchor=0.04)
    assert torch.allclose(x_t1[..., 2], anchor[..., 2], atol=1e-5), \
        f"heading drifted: max diff {(x_t1[..., 2] - anchor[..., 2]).abs().max():.2e}"
    assert torch.allclose(x1[..., 2], anchor[..., 2], atol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: lateral perturbation is non-zero at σ=0.04
# ---------------------------------------------------------------------------

def test_lateral_perturbation_nonzero():
    """σ=0.04 should produce a non-zero normalized lateral perturbation."""
    torch.manual_seed(123)
    # Anchor with lateral ≈ 0.05 m → normalized ≈ (0.05+0.19)/(0.24+0.19)*2-1
    anchor_phys = torch.tensor([[2.0, 0.05, 0.01]], dtype=torch.float32).expand(B, N_F, 3).contiguous()
    anchor = normalize(anchor_phys, Q01, Q99)
    tau_GT = anchor.clone()
    t = torch.full((B,), 1.0)
    x_t, _, x1 = build_anchored_flow_path(anchor, tau_GT, t, Q01, Q99, sigma_anchor=0.04)
    lat_diff = (x1[..., 1] - anchor[..., 1]).abs().max().item()
    assert lat_diff > 1e-3, f"lateral perturbation too small: {lat_diff:.2e}"


# ---------------------------------------------------------------------------
# Test 5: gradient flows through tau_GT
# ---------------------------------------------------------------------------

def test_grad_flow():
    """target & x_t are differentiable w.r.t. tau_GT."""
    anchor, _ = _dummy_inputs(7)
    tau_GT = torch.empty(B, N_F, 3).uniform_(-1, 1).requires_grad_(True)
    t = torch.full((B,), 0.5)
    x_t, target, _ = build_anchored_flow_path(anchor, tau_GT, t, Q01, Q99, sigma_anchor=0.04)
    (x_t.sum() + target.sum()).backward()
    assert tau_GT.grad is not None
    assert torch.isfinite(tau_GT.grad).all()
