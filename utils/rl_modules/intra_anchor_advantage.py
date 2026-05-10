"""
Intra-anchor advantage normalization for DriveVLA-W0 GRPO (Task 1.3).

Ports DiffusionDriveV2's group advantage computation to this repo.
Reference: diffusiondrivev2_model_rl.py:885-889.

For each anchor k, rewards across G candidates are standardized to zero
mean and unit std. This keeps the policy gradient scale consistent across
anchors regardless of the absolute reward magnitude.
"""

import torch


def compute_intra_anchor_advantages(
    rewards: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Standardize rewards within each anchor's G candidates.

    Mirrors DD-v2 diffusiondrivev2_model_rl.py:887-889:
      mean = reward_group.mean(dim=1)   [mean over G candidates per anchor]
      std  = reward_group.std(dim=1)
      A    = (rewards - mean) / (std + 1e-4)

    Our tensor layout is (B, N_anchor, G) instead of DD-v2's (B, G, N_anchor),
    so we normalize along dim=-1 (G axis) rather than dim=1.

    Args:
        rewards: Shape (B, N_anchor, G), per-candidate PDM rewards in [−1, 1].
                 G must be ≥ 2 for Bessel-corrected std to be well-defined;
                 G=1 yields nan which is replaced with 0 via nan_to_num.
        epsilon: Denominator floor for numerical stability (DD-v2 default 1e-4).

    Returns:
        advantages: Same shape/dtype/device as rewards.
                    Per-anchor mean ≈ 0, std ≈ 1 (up to finite-sample error).
    """
    mean = rewards.mean(dim=-1, keepdim=True)   # (B, N_anchor, 1)
    std  = rewards.std(dim=-1, keepdim=True)    # (B, N_anchor, 1), correction=1
    advantages = (rewards - mean) / (std + epsilon)
    return torch.nan_to_num(advantages, nan=0.0)
