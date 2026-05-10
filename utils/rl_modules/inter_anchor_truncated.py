"""
Inter-anchor truncation for DriveVLA-W0 GRPO (Task 1.4).

Ports DiffusionDriveV2's truncation rules to this repo.
Reference: diffusiondrivev2_model_rl.py:891-904.

Truncation logic (applied in order; later steps override earlier ones):
  1. Clamp negative advantages to 0.
  2. Zero out advantages for candidates worse than GT (below reward_gt − threshold).
  3. Force −1 for collision violations (no_collision != 1).
  4. Force −1 for drivable-area violations (drivable_area != 1).

Steps 3–4 always win: a collision trajectory gets −1 even if it was already
zeroed by step 1 or 2. This matches DD-v2's torch.where override pattern.
"""

import torch


def apply_inter_anchor_truncation(
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    reward_gt: torch.Tensor,
    no_collision: torch.Tensor,
    drivable_area: torch.Tensor,
    gt_threshold: float = 1e-6,
) -> torch.Tensor:
    """Apply DD-v2 inter-anchor truncation rules.

    Mirrors DD-v2 diffusiondrivev2_model_rl.py:891-904.

    Args:
        advantages:    Shape (B, N_anchor, G), intra-anchor normalized (Task 1.3 output).
        rewards:       Shape (B, N_anchor, G), original PDM rewards used for GT comparison.
        reward_gt:     Shape (B,), GT trajectory reward per scene. Broadcast to (B, 1, 1).
        no_collision:  Shape (B, N_anchor, G), float. 1.0 = no collision, 0.0 = collision.
                       Must be exactly 0.0 or 1.0 (as returned by PDM scorer).
        drivable_area: Shape (B, N_anchor, G), float. 1.0 = on-road, 0.0 = off-road.
                       Must be exactly 0.0 or 1.0 (as returned by PDM scorer).
        gt_threshold:  Margin for the "better than GT" mask (DD-v2 uses 1e-6).

    Returns:
        truncated: Same shape/dtype/device as advantages.
                   Values are in {−1} ∪ {0} ∪ ℝ⁺ (no NaN).
    """
    gt = reward_gt.view(-1, 1, 1)                             # (B, 1, 1)

    # Step 1+2: clamp negatives to 0; zero out worse-than-GT
    mask_gt = (rewards > gt - gt_threshold)                   # (B, N_anchor, G) bool
    adv = advantages.clamp(min=0.0) * mask_gt.float()

    # Steps 3+4: safety violations override to -1
    neg_one = torch.full_like(adv, -1.0)
    adv = torch.where(no_collision  != 1.0, neg_one, adv)
    adv = torch.where(drivable_area != 1.0, neg_one, adv)
    return adv
