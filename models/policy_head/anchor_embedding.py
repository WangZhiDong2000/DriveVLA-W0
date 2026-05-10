"""
Anchor embedding for DriveVLA-W0 anchored Flow Matching (Task 2.2).

Maps an anchor trajectory (B, N_f, action_dim) to a single token of
shape (B, action_hidden_size) that gets prepended to the action
expert's token sequence:

    action_h = [state_token, anchor_token, action_projector(noisy_action, tau_emb)]

The last Linear layer is zero-initialized so that anchor_token is
exactly 0 at initialization. This guarantees the joint forward
pass produces numerically identical outputs to the original 87.2
PDMS ckpt at training step 0 (Pass criteria §Task 2.2).

Multi-anchor batching (GRPO Stage 2-B):
    anchor shape is (B, N_f, action_dim) — one anchor per sample.
    For N_anchor anchors per scene, the caller expands the batch:
        anchor_BK = anchors.repeat_interleave(N_anchor, dim=0)
    Emu3Pi0 needs no special K-dim logic; all axes are flattened into B.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AnchorEmbedding(nn.Module):
    """Project anchor (B, N_f, action_dim) → token (B, action_hidden_size).

    Args:
        n_waypoints: number of waypoints per anchor (N_f, default 8).
        action_dim:  channels per waypoint (x, y, heading = 3).
        action_hidden_size: must equal action_config.hidden_size = 1024.
    """

    def __init__(
        self,
        n_waypoints: int = 8,
        action_dim: int = 3,
        action_hidden_size: int = 1024,
    ) -> None:
        super().__init__()
        in_dim = n_waypoints * action_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, action_hidden_size),
            nn.SiLU(),
            nn.Linear(action_hidden_size, action_hidden_size),
        )
        # Zero-init last layer so anchor_token == 0 at init.
        # Guarantees forward output is bit-compatible with original ckpt
        # before any training updates (Plan_3 v2.2 §Task 2.2 Pass criteria).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        """anchor: (B, N_f, action_dim) in normalized [-1, 1] space → (B, action_hidden_size)."""
        return self.mlp(anchor.flatten(1))
