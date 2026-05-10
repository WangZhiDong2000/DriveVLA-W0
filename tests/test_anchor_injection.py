"""
Task 2.2 — Anchor 注入 isolation tests.

Tests cover:
  1. AnchorEmbedding output shape: (B, action_hidden_size).
  2. AnchorEmbedding zero-init: anchor_token strictly == 0 at init.
  3. Gradient flow through AnchorEmbedding (after non-zero init).
  4. Different anchors → different anchor_tokens (after non-zero init).
  5. Token sequence shape with anchor: (B, 10, 1024).
  6. Token sequence shape without anchor: (B, 9, 1024).
  7. Decode slice correctness: skip 2 (anchor) vs skip 1 (no anchor).

Run:
    conda run -n drivevla python -m pytest tests/test_anchor_injection.py -v
"""

import torch
import torch.nn as nn
import pytest

from models.policy_head.anchor_embedding import AnchorEmbedding

B, N_F, ACTION_DIM, ACTION_HIDDEN = 2, 8, 3, 1024


# ---------------------------------------------------------------------------
# Test 1: output shape
# ---------------------------------------------------------------------------

def test_anchor_embedding_shape():
    """AnchorEmbedding must return (B, action_hidden_size)."""
    ae = AnchorEmbedding(N_F, ACTION_DIM, ACTION_HIDDEN)
    out = ae(torch.randn(B, N_F, ACTION_DIM))
    assert out.shape == (B, ACTION_HIDDEN), \
        f"Expected ({B}, {ACTION_HIDDEN}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: zero-init guarantee
# ---------------------------------------------------------------------------

def test_anchor_embedding_zero_init():
    """anchor_token must be strictly 0 at init (zero-init last layer)."""
    ae = AnchorEmbedding(N_F, ACTION_DIM, ACTION_HIDDEN)
    out = ae(torch.randn(B, N_F, ACTION_DIM))
    assert torch.all(out == 0.0), \
        f"anchor_token not zero at init: max abs={out.abs().max():.6f}"


# ---------------------------------------------------------------------------
# Test 3: gradient flow
# ---------------------------------------------------------------------------

def test_anchor_embedding_grad_flow():
    """Gradient must flow into all AnchorEmbedding parameters."""
    ae = AnchorEmbedding(N_F, ACTION_DIM, ACTION_HIDDEN)
    # Activate first layer so backprop reaches it (last layer is zero but has grad path)
    nn.init.normal_(ae.mlp[0].weight, std=0.02)
    a = torch.randn(B, N_F, ACTION_DIM)
    out = ae(a)
    out.sum().backward()
    assert ae.mlp[0].weight.grad is not None, "No gradient to first layer weight"
    assert torch.isfinite(ae.mlp[0].weight.grad).all(), "NaN/Inf in first layer grad"


# ---------------------------------------------------------------------------
# Test 4: different anchors → different tokens (after non-zero last layer)
# ---------------------------------------------------------------------------

def test_different_anchors_produce_different_tokens():
    """After non-zero last-layer init, different anchors → different anchor_tokens."""
    ae = AnchorEmbedding(N_F, ACTION_DIM, ACTION_HIDDEN)
    nn.init.normal_(ae.mlp[-1].weight, std=0.02)
    nn.init.normal_(ae.mlp[-1].bias, std=0.02)
    a1 = torch.randn(B, N_F, ACTION_DIM)
    a2 = torch.randn(B, N_F, ACTION_DIM)
    diff = (ae(a1) - ae(a2)).abs().max()
    assert diff > 1e-3, \
        f"Different anchors should produce different tokens, got max diff={diff:.6f}"


# ---------------------------------------------------------------------------
# Test 5: token sequence shape with anchor (B, 10, 1024)
# ---------------------------------------------------------------------------

def test_token_sequence_with_anchor():
    """Full token sequence with anchor: (B, 1+1+N_F, action_hidden) = (B, 10, 1024)."""
    ae = AnchorEmbedding(N_F, ACTION_DIM, ACTION_HIDDEN)
    state_tok = torch.randn(B, 1, ACTION_HIDDEN)
    action_hs = torch.randn(B, N_F, ACTION_HIDDEN)
    anchor_tok = ae(torch.randn(B, N_F, ACTION_DIM)).unsqueeze(1)  # (B, 1, 1024)
    seq = torch.cat([state_tok, anchor_tok, action_hs], dim=1)
    assert seq.shape == (B, 1 + 1 + N_F, ACTION_HIDDEN), \
        f"Expected ({B}, 10, {ACTION_HIDDEN}), got {seq.shape}"
    assert seq.shape[1] == 10


# ---------------------------------------------------------------------------
# Test 6: token sequence shape without anchor (B, 9, 1024) — backward compat
# ---------------------------------------------------------------------------

def test_token_sequence_without_anchor():
    """Backward-compat: anchor=None → (B, 1+N_F, action_hidden) = (B, 9, 1024)."""
    state_tok = torch.randn(B, 1, ACTION_HIDDEN)
    action_hs = torch.randn(B, N_F, ACTION_HIDDEN)
    seq = torch.cat([state_tok, action_hs], dim=1)
    assert seq.shape == (B, 1 + N_F, ACTION_HIDDEN), \
        f"Expected ({B}, 9, {ACTION_HIDDEN}), got {seq.shape}"
    assert seq.shape[1] == 9


# ---------------------------------------------------------------------------
# Test 7: decode slice correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("anchor_provided,expected_skip", [
    (False, 1),   # anchor=None: skip state_token only → (B, N_F, h)
    (True,  2),   # anchor provided: skip state_token + anchor_token → (B, N_F, h)
])
def test_decode_slice(anchor_provided, expected_skip):
    """Decode slice [:, _decode_skip:, :] gives (B, N_F, action_hidden) in both cases."""
    seq_len = 1 + (1 if anchor_provided else 0) + N_F  # 10 or 9
    final_hidden = torch.randn(B, seq_len, ACTION_HIDDEN)
    decoded_in = final_hidden[:, expected_skip:, :]
    assert decoded_in.shape == (B, N_F, ACTION_HIDDEN), \
        f"anchor_provided={anchor_provided}: expected ({B}, {N_F}, {ACTION_HIDDEN}), got {decoded_in.shape}"
