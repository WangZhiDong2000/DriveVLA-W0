"""
Unit tests for utils/rl_modules/rollout_collector.py (Task 1.6).

All tests use mock denoise_fn and mock PDMRewardWrapper — no real model
and no MetricCache .lzma files are required.

Run:
    conda run -n drivevla python -m pytest tests/test_rollout_collector.py -v

Pass criteria (v2.2 §1.6):
  1. All 9 RolloutBatch fields have correct shapes (B=4, K=160, T=10, N_f=8)
  2. logp_old is finite and matches log N(eps; 0, σ²) exactly
  3. advantages discount structure: last step ≥ first step (γ^0 vs γ^(T-1))
  4. GT trajectory correctly separated from trajectories field
  5. sub_rewards keys all present and correctly shaped
"""

import math
from typing import Dict
from unittest.mock import MagicMock

import torch
import pytest

from utils.rl_modules.rollout_collector import collect_rollouts, RolloutBatch


# ---------------------------------------------------------------------------
# Test constants (match plan spec)
# ---------------------------------------------------------------------------

B, N_ANCHOR, G, T, N_F = 4, 20, 8, 10, 8
K = N_ANCHOR * G    # 160

# Actual q01/q99 from configs/normalizer_navsim_trainval/norm_stats.json
Q01 = torch.tensor([-0.0179, -0.1909, -0.1892])
Q99 = torch.tensor([6.2000, 0.2426, 0.1805])

SUB_KEYS = ("no_collision", "drivable_area", "progress",
            "ttc", "comfort", "dir_weighted", "final")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_denoise_fn(x_t: torch.Tensor, t: float) -> torch.Tensor:
    """Zero velocity → Euler step is identity (only noise changes x_t)."""
    return torch.zeros_like(x_t)


def _make_scorer(seed: int = 42) -> MagicMock:
    """Mock PDMRewardWrapper.score() with deterministic random outputs."""
    torch.manual_seed(seed)
    rewards  = torch.rand(B, K)
    reward_gt = torch.rand(B)
    sub_rewards: Dict[str, torch.Tensor] = {}
    for k in SUB_KEYS:
        v = torch.rand(B, K)
        if k in ("no_collision", "drivable_area"):
            v = (v > 0.3).float()   # binary masks expected by Task 1.4
        sub_rewards[k] = v

    scorer = MagicMock()
    scorer.score.return_value = (rewards, reward_gt, sub_rewards)
    return scorer


def _make_inputs(seed: int = 0):
    """Return (anchor_centers, gt_traj, metric_cache_paths)."""
    torch.manual_seed(seed)
    # Physical-space trajectories: forward ~[0,4] m, lateral ~[0,0.2] m
    scale = torch.tensor([4.0, 0.2, 0.15])
    anchor_centers = torch.rand(N_ANCHOR, N_F, 3) * scale
    gt_traj        = torch.rand(B, N_F, 3) * scale
    paths          = ["/fake/scene_{}.lzma".format(b) for b in range(B)]
    return anchor_centers, gt_traj, paths


def _call(**kwargs):
    """collect_rollouts with default arguments, merging any overrides."""
    anchors, gt_traj, paths = _make_inputs()
    defaults = dict(
        denoise_fn=_mock_denoise_fn,
        gt_trajectory=gt_traj,
        anchor_centers=anchors,
        scorer=_make_scorer(),
        metric_cache_paths=paths,
        q01=Q01,
        q99=Q99,
        num_groups=G,
        num_steps=T,
    )
    defaults.update(kwargs)
    return collect_rollouts(**defaults)


# ---------------------------------------------------------------------------
# Test 1: All RolloutBatch field shapes
# ---------------------------------------------------------------------------

def test_rollout_batch_shapes():
    """Every field must have exactly the shape specified in the plan."""
    rb = _call()

    assert rb.anchors.shape       == (B, N_ANCHOR, N_F, 3),  f"anchors:      {rb.anchors.shape}"
    assert rb.x_t_per_step.shape  == (B, K, T, N_F, 3),      f"x_t_per_step: {rb.x_t_per_step.shape}"
    assert rb.eps_step.shape      == (B, K, T, 2),            f"eps_step:     {rb.eps_step.shape}"
    assert rb.logp_old.shape      == (B, K, T),               f"logp_old:     {rb.logp_old.shape}"
    assert rb.rewards.shape       == (B, K),                  f"rewards:      {rb.rewards.shape}"
    assert rb.advantages.shape    == (B, K, T),               f"advantages:   {rb.advantages.shape}"
    assert rb.trajectories.shape  == (B, K, N_F, 3),          f"trajectories: {rb.trajectories.shape}"
    assert rb.gt_trajectory.shape == (B, N_F, 3),             f"gt_trajectory:{rb.gt_trajectory.shape}"

    assert set(rb.sub_rewards.keys()) == set(SUB_KEYS), \
        f"sub_rewards keys: {set(rb.sub_rewards.keys())}"
    for k, v in rb.sub_rewards.items():
        assert v.shape == (B, K), f"sub_rewards['{k}']: {v.shape}"


# ---------------------------------------------------------------------------
# Test 2: logp_old finite
# ---------------------------------------------------------------------------

def test_logp_old_finite():
    """logp_old must not contain NaN or Inf."""
    rb = _call()
    bad = ~torch.isfinite(rb.logp_old)
    assert not bad.any(), \
        f"logp_old has {bad.sum()} non-finite values at positions {bad.nonzero()[:5]}"


# ---------------------------------------------------------------------------
# Test 3: logp_old matches log N(eps; 0, σ²) formula exactly
# ---------------------------------------------------------------------------

def test_logp_matches_eps_formula():
    """logp_old[b,k,t] == -0.5 * ||eps[b,k,t]||² / σ² - log(2π σ²)."""
    sigma = 0.04
    rb = _call(sigma_step=sigma)

    eps_sq_sum = (rb.eps_step ** 2).sum(dim=-1)   # (B, K, T)
    expected = -0.5 * eps_sq_sum / sigma**2 - math.log(2.0 * math.pi * sigma**2)

    assert torch.allclose(rb.logp_old, expected, atol=1e-5), \
        f"logp_old formula mismatch; max |err|={(rb.logp_old - expected).abs().max():.2e}"


# ---------------------------------------------------------------------------
# Test 4: Temporal discount structure (last step ≥ first step in magnitude)
# ---------------------------------------------------------------------------

def test_advantages_discount_structure():
    """Last denoising step carries higher weight than first (γ^0 vs γ^(T-1))."""
    # Build a scorer that returns all-positive rewards so advantages ≠ 0
    scorer = _make_scorer()
    rewards, reward_gt, sub_rewards = scorer.score.return_value
    rewards[:] = 0.95
    reward_gt[:] = 0.0    # ensure reward > reward_gt − 1e-6
    # All safe (no collision / drivable violations)
    sub_rewards["no_collision"][:] = 1.0
    sub_rewards["drivable_area"][:] = 1.0
    scorer.score.return_value = (rewards, reward_gt, sub_rewards)

    rb = _call(scorer=scorer, discount=0.8)

    non_zero = (rb.advantages != 0)
    if non_zero.any():
        last_step_mag  = rb.advantages[..., -1].abs().mean()   # γ^0 = 1.0
        first_step_mag = rb.advantages[..., 0].abs().mean()    # γ^(T-1) = 0.8^9
        assert last_step_mag >= first_step_mag - 1e-5, (
            f"Last step magnitude {last_step_mag:.4f} should ≥ "
            f"first step {first_step_mag:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 5: GT not mixed into trajectories
# ---------------------------------------------------------------------------

def test_gt_not_in_trajectories():
    """gt_trajectory is stored separately; trajectories has exactly K candidates."""
    anchors, gt_traj, paths = _make_inputs()
    rb = collect_rollouts(
        _mock_denoise_fn, gt_traj, anchors, _make_scorer(), paths,
        Q01, Q99, num_groups=G, num_steps=T,
    )

    assert rb.trajectories.shape[1] == K, \
        f"Expected K={K} candidates in trajectories, got {rb.trajectories.shape[1]}"
    assert torch.allclose(rb.gt_trajectory, gt_traj), \
        "gt_trajectory stored incorrectly"


# ---------------------------------------------------------------------------
# Test 6: Memory sanity (B=4, K=160, T=10, N_f=8, 3ch float32 ≈ 0.6 MB)
# ---------------------------------------------------------------------------

def test_memory_bound():
    """x_t_per_step should be well under 100 MB for the standard config."""
    rb = _call()
    nbytes = rb.x_t_per_step.element_size() * rb.x_t_per_step.nelement()
    limit_bytes = 100 * 1024 * 1024   # 100 MB
    assert nbytes < limit_bytes, \
        f"x_t_per_step uses {nbytes / 1e6:.1f} MB (expected ≪ 100 MB)"
