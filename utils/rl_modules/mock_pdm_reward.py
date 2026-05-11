"""
Mock PDM reward wrapper for local smoke testing (Task 4.1).

Implements the same interface as PDMRewardWrapper.score without any navsim
dependencies or metric_cache .lzma files — usable on machines where the full
NAVSIM environment is not available.

Reward heuristic:
    rewards[b, k] = exp(-0.5 * mean_squared_xy_error(traj[b,k], gt[b]))
so rewards ∈ (0, 1] and vary across K candidates, giving non-degenerate
intra-anchor advantage distributions for smoke testing.

sub_rewards: all 7 required keys filled with the same rewards tensor so that
inter_anchor_truncation (which reads 'no_collision' and 'drivable_area')
does not error.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class MockPDMRewardWrapper:
    """Drop-in replacement for PDMRewardWrapper for local smoke tests.

    Usage::

        scorer = MockPDMRewardWrapper()
        rewards, reward_gt, sub_rewards = scorer.score(
            trajectories_phys,      # (B, K, N_f, 3) float32
            gt_trajectories_phys,   # (B, N_f, 3)    float32
            metric_cache_paths,     # ignored (List[Optional[str]])
            device=device,
        )
    """

    _SUB_KEYS = (
        "no_collision", "drivable_area", "progress",
        "ttc", "comfort", "dir_weighted", "final",
    )

    def score(
        self,
        trajectories_phys: np.ndarray,
        gt_trajectories_phys: np.ndarray,
        metric_cache_paths: List[Optional[str]],  # ignored
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Score K candidate trajectories per scene via a simple XY MSE heuristic.

        Args:
            trajectories_phys:    (B, K, N_f, 3) float32, physical (m/rad).
            gt_trajectories_phys: (B, N_f, 3)    float32, physical.
            metric_cache_paths:   Ignored; present for interface parity.
            device:               Target device for returned tensors.

        Returns:
            rewards:    (B, K) ∈ (0, 1].
            reward_gt:  (B,)   = 1.0 (GT gets perfect score).
            sub_rewards: dict with 7 keys, each (B, K).
        """
        B, K = trajectories_phys.shape[:2]

        gt_xy = gt_trajectories_phys[:, :, :2]          # (B, N_f, 2)
        traj_xy = trajectories_phys[:, :, :, :2]        # (B, K, N_f, 2)

        # Mean squared XY error per candidate, converted to reward
        mse = ((traj_xy - gt_xy[:, None, :, :]) ** 2).mean(axis=(-2, -1))  # (B, K)
        rewards_np = np.exp(-0.5 * mse).astype(np.float32)

        rewards   = torch.tensor(rewards_np, device=device, dtype=torch.float32)
        reward_gt = torch.ones(B, device=device, dtype=torch.float32)

        sub_rewards: Dict[str, torch.Tensor] = {
            k: rewards.clone() for k in self._SUB_KEYS
        }
        # Ensure no_collision and drivable_area ∈ {0, 1} (they are binary in real PDM).
        # Binarise by thresholding at the per-batch median so roughly half pass.
        median_val = rewards.median(dim=1, keepdim=True).values      # (B, 1)
        binary = (rewards >= median_val).to(torch.float32)
        sub_rewards["no_collision"]  = binary
        sub_rewards["drivable_area"] = binary

        return rewards, reward_gt, sub_rewards

    def shutdown(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass
