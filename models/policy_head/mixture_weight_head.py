"""
Mixture Weight Head for DriveVLA-W0 GRPO (Task 2.4).

Reads the anchor_token position of Emu3Pi0's action_expert final hidden state
and outputs a scalar logit s_hat^k per (sample, anchor) pair. Used by
Plan_3 v2.2 §1.1 for anchor classification:

    L_BCE = sum_k BCE(s_hat^k, 1[k = k_positive])

Architecture mirrors AnchorEmbedding: 2-layer SiLU MLP, zero-init last
layer so logit == 0 at init -> sigmoid == 0.5 (uniform prior over anchors).
Combined with AnchorEmbedding's zero-init anchor_token, guarantees Phase 3
startup is bit-compatible with the 87.2 PDMS ckpt.

Multi-anchor batching (GRPO Stage 2-B):
    Caller batches N_anchor anchors into B via repeat_interleave; head
    sees (B*N_anchor, h) -> (B*N_anchor,). Outer trainer reshape into
    (B, N_anchor) and run BCE.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MixtureWeightHead(nn.Module):
    """Anchor-token hidden state -> scalar mixture logit.

    Args:
        action_hidden_size: must equal action_config.hidden_size = 1024.
    """

    def __init__(self, action_hidden_size: int = 1024) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(action_hidden_size, action_hidden_size),
            nn.SiLU(),
            nn.Linear(action_hidden_size, 1),
        )
        # Zero-init last layer: logit == 0 at init -> sigmoid == 0.5
        # (uniform prior over anchors). Combined with AnchorEmbedding
        # zero-init, preserves 87.2 PDMS ckpt bit-compat at step 0.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, anchor_hidden: torch.Tensor) -> torch.Tensor:
        """anchor_hidden: (B, action_hidden_size) -> logit (B,)."""
        return self.mlp(anchor_hidden).squeeze(-1)
