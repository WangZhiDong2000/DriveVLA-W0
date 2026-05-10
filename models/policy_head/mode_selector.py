"""
VLM-conditioned mode selector for DriveVLA-W0 GRPO (Task 1.7 skeleton).

Phase 1: interface and shape validation only.
Phase 6 Task 6.1: train on Stage 2-B rollout data with BCE + Margin-Rank loss.

Interface:
    ModeSelector(trajectories, vlm_h) -> logits
    trajectories: (B, K, N_f, 3)  physical or normalized (caller decides)
    vlm_h:        (B, S, H_vlm)   VLM decoder last-layer hidden states
    logits:       (B, K)          unnormalized selection scores

Architecture (§6.1):
    1. Trajectory MLP encoder  → query embeddings (B, K, d_sel)
    2. Cross-attention (Q=traj, K=V=vlm_h)  — attend to scene context
    3. Self-attention over K candidates      — comparative ranking
    4. Scalar score MLP                      → (B, K) logits

DD-v2 analogue: DiffMotionPlanningRefinementModule (model_rl.py:309-354),
but BEV cross-attention is replaced by VLM hidden state conditioning since
this repo has no BEV encoder (v2.2 §0.2 #6).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModeSelector(nn.Module):
    """VLM-conditioned trajectory mode selector.

    Args:
        n_waypoints: Number of waypoints per trajectory (N_f, default 8).
        action_dim:  Channels per waypoint (x, y, heading = 3).
        d_sel:       Internal feature dimension (default 256).
        num_heads:   MHA heads; must divide d_sel evenly (default 4).
        vlm_hidden:  VLM hidden state dimension H_vlm (default 4096 for Emu3).
    """

    def __init__(
        self,
        n_waypoints: int = 8,
        action_dim: int = 3,
        d_sel: int = 256,
        num_heads: int = 4,
        vlm_hidden: int = 4096,
    ) -> None:
        super().__init__()
        assert d_sel % num_heads == 0, \
            f"d_sel={d_sel} must be divisible by num_heads={num_heads}"

        traj_in = n_waypoints * action_dim
        self.traj_encoder = nn.Sequential(
            nn.Linear(traj_in, d_sel),
            nn.SiLU(),
            nn.Linear(d_sel, d_sel),
        )
        self.vlm_proj  = nn.Linear(vlm_hidden, d_sel)
        self.cross_attn = nn.MultiheadAttention(d_sel, num_heads, batch_first=True)
        self.self_attn  = nn.MultiheadAttention(d_sel, num_heads, batch_first=True)
        self.score_head = nn.Sequential(
            nn.Linear(d_sel, d_sel // 2),
            nn.SiLU(),
            nn.Linear(d_sel // 2, 1),
        )

    def forward(
        self,
        trajectories: torch.Tensor,  # (B, K, N_f, 3)
        vlm_h: torch.Tensor,         # (B, S, H_vlm)
    ) -> torch.Tensor:               # (B, K)
        """Score K candidate trajectories conditioned on VLM hidden states.

        Args:
            trajectories: (B, K, N_f, 3) candidate trajectories.
            vlm_h:        (B, S, H_vlm) VLM decoder hidden states.

        Returns:
            Tensor of shape (B, K) with unnormalized logits.
        """
        # 1. Trajectory encoding: (B, K, N_f*3) → (B, K, d_sel)
        q = self.traj_encoder(trajectories.flatten(2))

        # 2. Cross-attention: trajectory queries attend to VLM scene context
        kv = self.vlm_proj(vlm_h)              # (B, S, d_sel)
        q_cross, _ = self.cross_attn(q, kv, kv)
        q_cross = q + q_cross                  # residual

        # 3. Self-attention: candidates compare each other
        q_self, _ = self.self_attn(q_cross, q_cross, q_cross)
        q_self = q_cross + q_self              # residual

        # 4. Scalar score per candidate
        return self.score_head(q_self).squeeze(-1)  # (B, K)
