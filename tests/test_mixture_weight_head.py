"""
Task 2.4 — Mixture Weight Head isolation tests.

Tests cover (Plan_3 v2.2 §301 Pass criteria, with §300 misleading
"softmax sums to 1" replaced by per-anchor sigmoid/BCE semantics):
  1. Output shape: (B,).
  2. Zero-init: logit strictly == 0 at init -> sigmoid == 0.5.
  3. Sigmoid range: sigmoid(logit) ∈ (0, 1) for arbitrary inputs.
  4. Different anchor hiddens -> different logits (after non-zero init).
  5. Gradient flows into all MixtureWeightHead parameters.

Run:
    conda run -n drivevla python -m pytest tests/test_mixture_weight_head.py -v
"""

import torch
import torch.nn as nn

from models.policy_head.mixture_weight_head import MixtureWeightHead

B, ACTION_HIDDEN = 4, 1024


def test_output_shape():
    """Output shape must be (B,)."""
    head = MixtureWeightHead(ACTION_HIDDEN)
    out = head(torch.randn(B, ACTION_HIDDEN))
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


def test_zero_init_logit_is_zero():
    """At init, logit must be strictly 0 (zero-init last layer)."""
    head = MixtureWeightHead(ACTION_HIDDEN)
    out = head(torch.randn(B, ACTION_HIDDEN))
    assert torch.all(out == 0.0), \
        f"logit not zero at init: max abs = {out.abs().max():.6f}"
    # sigmoid(0) = 0.5 (uniform anchor prior)
    assert torch.allclose(torch.sigmoid(out), torch.full((B,), 0.5))


def test_sigmoid_range():
    """sigmoid(logit) ∈ (0, 1) for non-zero last layer init."""
    head = MixtureWeightHead(ACTION_HIDDEN)
    nn.init.normal_(head.mlp[-1].weight, std=0.1)
    nn.init.normal_(head.mlp[-1].bias, std=0.1)
    p = torch.sigmoid(head(torch.randn(B, ACTION_HIDDEN)))
    assert (p > 0).all() and (p < 1).all(), \
        f"sigmoid out of (0,1): min={p.min():.6f}, max={p.max():.6f}"


def test_different_inputs_produce_different_logits():
    """After non-zero last-layer init, distinct anchor hiddens -> distinct logits."""
    head = MixtureWeightHead(ACTION_HIDDEN)
    nn.init.normal_(head.mlp[-1].weight, std=0.02)
    nn.init.normal_(head.mlp[-1].bias, std=0.02)
    h1 = torch.randn(B, ACTION_HIDDEN)
    h2 = torch.randn(B, ACTION_HIDDEN)
    diff = (head(h1) - head(h2)).abs().max()
    assert diff > 1e-3, \
        f"Different inputs should produce different logits, got max diff={diff:.6f}"


def test_grad_flow():
    """Gradient must flow into all MixtureWeightHead parameters."""
    head = MixtureWeightHead(ACTION_HIDDEN)
    # Activate first layer so backprop reaches it (last layer zero-init still
    # has grad path through input, but we want non-zero output for first layer).
    nn.init.normal_(head.mlp[0].weight, std=0.02)
    nn.init.normal_(head.mlp[-1].weight, std=0.02)
    h = torch.randn(B, ACTION_HIDDEN)
    head(h).sum().backward()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert torch.isfinite(p.grad).all(), f"NaN/Inf in {name} grad"
