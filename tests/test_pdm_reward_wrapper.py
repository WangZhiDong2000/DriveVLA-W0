"""
Unit tests for utils/rl_modules/pdm_reward_wrapper.py (Task 1.5).

Level 1 tests (no MetricCache data needed, always run):
  1. _pairwise_scores formula — mock SCORER, hand-computed expected values
  2. _pairwise_subscores keys — 7 expected keys present, shapes correct
  3. PDMRewardWrapper.score output shapes — mock pool, (B,K)/(B,)/(B,K) dict
  4. GT index split — scores[-1] → reward_gt, scores[:-1] → rewards

Level 2 tests (require NAVSIM_CACHE_DIR env var, skipped otherwise):
  5. GT trajectory → R ≥ 0.95
  6. Collision trajectory → no_collision = 0 → final = 0
  7. Off-road trajectory → drivable_area = 0 → final = 0

Run all Level 1:
    conda run -n drivevla python -m pytest tests/test_pdm_reward_wrapper.py -v -m "not integration"

Run with data:
    NAVSIM_CACHE_DIR=/path/to/cache \\
    conda run -n drivevla python -m pytest tests/test_pdm_reward_wrapper.py -v
"""

import os
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from utils.rl_modules.pdm_reward_wrapper import (
    _pairwise_scores,
    _pairwise_subscores,
    PDMRewardWrapper,
)


# ---------------------------------------------------------------------------
# Helper: build a synthetic mock SCORER
#
# Trajectory layout: [PDM_ref | cand_0 | cand_1 | GT]  → N = 4
#   - PDM_ref: index 0 (progress reference, not returned in scores)
#   - cand_0:  index 1, no_collision=1, drivable_area=1
#   - cand_1:  index 2, no_collision=0 (collision)  → final = 0
#   - GT:      index 3, no_collision=1, drivable_area=1
# ---------------------------------------------------------------------------

def _build_mock_scorer():
    """Return a MagicMock mirroring PDMScorer after score_proposals()."""
    N = 4   # PDM ref + 2 candidates + GT

    # MultiMetricIndex: NO_COLLISION=0, DRIVABLE_AREA=1
    mm = np.array([
        [1.0, 1.0, 0.0, 1.0],   # NO_COLLISION: cand_1 collision
        [1.0, 1.0, 1.0, 1.0],   # DRIVABLE_AREA: all on road
    ], dtype=np.float64)         # (2, N)

    # WeightedMetricIndex: PROGRESS=0, TTC=1, COMFORTABLE=2, DRIVING_DIRECTION=3
    wm = np.array([
        [0.5, 0.8, 0.7, 0.9],   # PROGRESS (will be overwritten for [1:])
        [1.0, 0.9, 0.8, 1.0],   # TTC
        [1.0, 0.7, 0.6, 0.8],   # COMFORTABLE
        [1.0, 1.0, 1.0, 1.0],   # DRIVING_DIRECTION (weight=0, no contribution)
    ], dtype=np.float64)          # (4, N)

    # progress_raw — keep small so max_pair stays below threshold for cand_1 (collision)
    # thresh=5.0; set raw_prog=1.0 for all → below thresh → norm_prog by rule
    prog_raw = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    # Config mock
    config = MagicMock()
    # weighted_metrics_array shape (4,): progress=5, ttc=5, comfortable=2, dir=0
    config.weighted_metrics_array = np.array([5.0, 5.0, 2.0, 0.0], dtype=np.float64)
    config.progress_distance_threshold = 5.0

    scorer = MagicMock()
    scorer._multi_metrics = mm
    scorer._weighted_metrics = wm
    scorer._progress_raw = prog_raw
    scorer._config = config
    return scorer


# ---------------------------------------------------------------------------
# Test 1: _pairwise_scores formula
# ---------------------------------------------------------------------------

def test_pairwise_scores_formula():
    """Verify _pairwise_scores computes correct values with a mock SCORER.

    Expected logic (thresh=5.0, all raw_prog=1.0 → max_pair=1.0 < 5.0):
      For each index i in [1, 2, 3] (candidates + GT):
        - multi_prod[i] = NO_COLLISION[i] * DRIVABLE_AREA[i]
        - max_pair = max(raw_prog_ref * multi_prod[0], raw_prog[i] * multi_prod[i])
        - norm_prog: if max_pair > 5 → ratio; else 0 if collision, 1 otherwise
        - weighted_score = (5*norm_prog + 5*ttc + 2*comfort + 0*dir) / 12
        - final = multi_prod[i] * weighted_score

    cand_0 (index 1): multi_prod=1.0, norm_prog=1.0
      weighted = (5*1 + 5*0.9 + 2*0.7) / 12 = (5+4.5+1.4)/12 = 10.9/12 ≈ 0.9083
      final = 1.0 * 0.9083

    cand_1 (index 2): multi_prod=0.0 (collision) → norm_prog=0.0
      weighted = (5*0 + 5*0.8 + 2*0.6) / 12 = (0+4.0+1.2)/12 = 5.2/12 ≈ 0.4333
      final = 0.0 * 0.4333 = 0.0

    GT (index 3): multi_prod=1.0, norm_prog=1.0
      weighted = (5*1 + 5*1.0 + 2*0.8) / 12 = (5+5+1.6)/12 = 11.6/12 ≈ 0.9667
      final = 1.0 * 0.9667
    """
    scorer = _build_mock_scorer()
    scores = _pairwise_scores(scorer)

    assert scores.shape == (3,), f"Expected shape (3,), got {scores.shape}"
    assert scores.dtype == np.float32

    # cand_0
    expected_cand0 = (5 * 1.0 + 5 * 0.9 + 2 * 0.7 + 0 * 1.0) / 12.0
    assert abs(scores[0] - expected_cand0) < 1e-5, \
        f"cand_0 score mismatch: expected {expected_cand0:.6f}, got {scores[0]:.6f}"

    # cand_1 (collision → final = 0)
    assert abs(scores[1] - 0.0) < 1e-7, \
        f"cand_1 (collision) score should be 0.0, got {scores[1]}"

    # GT
    expected_gt = (5 * 1.0 + 5 * 1.0 + 2 * 0.8 + 0 * 1.0) / 12.0
    assert abs(scores[2] - expected_gt) < 1e-5, \
        f"GT score mismatch: expected {expected_gt:.6f}, got {scores[2]:.6f}"


# ---------------------------------------------------------------------------
# Test 2: _pairwise_subscores keys and shapes
# ---------------------------------------------------------------------------

def test_pairwise_subscores_keys():
    """_pairwise_subscores must return exactly 7 keys, each shape (N-1,)."""
    expected_keys = {
        "no_collision", "drivable_area", "progress",
        "ttc", "comfort", "dir_weighted", "final",
    }
    scorer = _build_mock_scorer()
    sub = _pairwise_subscores(scorer)

    assert set(sub.keys()) == expected_keys, \
        f"Key mismatch: got {set(sub.keys())}"

    for k, v in sub.items():
        assert v.shape == (3,), f"Key '{k}' shape should be (3,), got {v.shape}"


def test_pairwise_subscores_no_collision_values():
    """no_collision sub-score matches the mock mm values for [1:]."""
    scorer = _build_mock_scorer()
    sub = _pairwise_subscores(scorer)

    expected_nc = np.array([1.0, 0.0, 1.0])   # cand_0 safe, cand_1 collision, GT safe
    assert np.allclose(sub["no_collision"], expected_nc, atol=1e-7), \
        f"no_collision mismatch: {sub['no_collision']}"


def test_pairwise_subscores_final_zero_on_collision():
    """final sub-score is 0 for collision trajectory."""
    scorer = _build_mock_scorer()
    sub = _pairwise_subscores(scorer)

    assert abs(sub["final"][1] - 0.0) < 1e-7, \
        f"final score for collision candidate should be 0, got {sub['final'][1]}"


# ---------------------------------------------------------------------------
# Test 3: PDMRewardWrapper.score output shapes (mocked pool)
# ---------------------------------------------------------------------------

def _make_fake_scores(K: int) -> tuple:
    """Return fake (K+1,) scores and subscores dict for one scene."""
    scores = np.random.rand(K + 1).astype(np.float32)
    sub = {
        k: np.random.rand(K + 1).astype(np.float64)
        for k in ("no_collision", "drivable_area", "progress",
                  "ttc", "comfort", "dir_weighted", "final")
    }
    return scores, sub


def _make_wrapper_with_mock_pool(fake_results):
    """Return a PDMRewardWrapper whose pool is replaced by a mock that returns fake_results."""
    from unittest.mock import patch as _patch

    call_count = [0]

    def fake_submit(fn, args):
        idx = call_count[0]
        call_count[0] += 1
        f = MagicMock()
        f.result.return_value = fake_results[idx]
        return f

    mock_pool = MagicMock()
    mock_pool.submit.side_effect = fake_submit

    # Bypass __init__ (which needs OmegaConf/navsim) and inject mock pool
    with patch.object(PDMRewardWrapper, "__init__", lambda self, *a, **kw: None):
        wrapper = PDMRewardWrapper()
    wrapper._pool = mock_pool
    return wrapper


def test_score_output_shapes():
    """PDMRewardWrapper.score() returns (B,K)/(B,)/dict[str,(B,K)] shapes."""
    B, K = 3, 5
    fake_results = [_make_fake_scores(K) for _ in range(B)]

    wrapper = _make_wrapper_with_mock_pool(fake_results)
    traj = np.zeros((B, K, 8, 3), dtype=np.float32)
    gt = np.zeros((B, 8, 3), dtype=np.float32)
    paths = ["/fake/cache.lzma"] * B

    rewards, reward_gt, sub_rewards = wrapper.score(traj, gt, paths)

    assert rewards.shape == (B, K), f"rewards shape: {rewards.shape}"
    assert reward_gt.shape == (B,), f"reward_gt shape: {reward_gt.shape}"
    assert isinstance(sub_rewards, dict)
    for k, v in sub_rewards.items():
        assert v.shape == (B, K), f"sub_rewards['{k}'] shape: {v.shape}"


# ---------------------------------------------------------------------------
# Test 4: GT index split — scores[-1] → reward_gt, scores[:-1] → rewards
# ---------------------------------------------------------------------------

def test_score_gt_index_split():
    """Verify GT score ends up in reward_gt, not in rewards."""
    B, K = 2, 4
    gt_score_value = 0.99   # distinctive value for GT

    fake_results = []
    for b in range(B):
        scores = np.full(K + 1, 0.5, dtype=np.float32)
        scores[-1] = gt_score_value   # GT is last
        sub = {k: np.ones(K + 1, dtype=np.float64)
               for k in ("no_collision", "drivable_area", "progress",
                         "ttc", "comfort", "dir_weighted", "final")}
        fake_results.append((scores, sub))

    wrapper = _make_wrapper_with_mock_pool(fake_results)
    traj = np.zeros((B, K, 8, 3), dtype=np.float32)
    gt = np.zeros((B, 8, 3), dtype=np.float32)
    paths = ["/fake/cache.lzma"] * B

    rewards, reward_gt, sub_rewards = wrapper.score(traj, gt, paths)

    # GT value must appear in reward_gt, not in rewards
    assert torch.allclose(reward_gt, torch.full((B,), gt_score_value)), \
        f"reward_gt should be {gt_score_value}, got {reward_gt}"
    assert not torch.any(rewards == gt_score_value), \
        f"GT score should not appear in rewards tensor"

    # rewards should all be 0.5
    assert torch.allclose(rewards, torch.full((B, K), 0.5)), \
        f"rewards should all be 0.5, got {rewards}"


# ---------------------------------------------------------------------------
# Level 2: Integration tests (require MetricCache data)
# ---------------------------------------------------------------------------

CACHE_DIR = os.environ.get("NAVSIM_CACHE_DIR", "")


@pytest.mark.integration
@pytest.mark.skipif(not CACHE_DIR, reason="NAVSIM_CACHE_DIR env var not set")
def test_gt_trajectory_high_score():
    """GT trajectory should receive R ≥ 0.95 per v2.2 §1.5 spec."""
    import glob
    cache_files = glob.glob(os.path.join(CACHE_DIR, "**/*.lzma"), recursive=True)
    assert cache_files, f"No .lzma files found under {CACHE_DIR}"

    cache_path = cache_files[0]
    with lzma.open(cache_path, "rb") as f:
        metric_cache = pickle.load(f)

    gt_traj = metric_cache.trajectory  # navsim Trajectory object
    gt_poses = np.array(gt_traj.poses, dtype=np.float32)   # (N_f, 3)

    with PDMRewardWrapper(num_workers=1) as wrapper:
        candidates = gt_poses[None]   # (1, N_f, 3) — use GT as the single candidate
        rewards, reward_gt, _ = wrapper.score(
            candidates[None],             # (B=1, K=1, N_f, 3)
            gt_poses[None],               # (B=1, N_f, 3)
            [cache_path],
        )

    assert reward_gt[0].item() >= 0.95, \
        f"GT trajectory should score ≥ 0.95, got {reward_gt[0].item():.4f}"


@pytest.mark.integration
@pytest.mark.skipif(not CACHE_DIR, reason="NAVSIM_CACHE_DIR env var not set")
def test_collision_trajectory_score():
    """Trajectory with no_collision=0 should have final sub-score = 0."""
    import glob
    cache_files = glob.glob(os.path.join(CACHE_DIR, "**/*.lzma"), recursive=True)
    assert cache_files

    cache_path = cache_files[0]
    with lzma.open(cache_path, "rb") as f:
        metric_cache = pickle.load(f)

    gt_poses = np.array(metric_cache.trajectory.poses, dtype=np.float32)

    # Collision trajectory: move sideways into obstacles
    collision_poses = gt_poses.copy()
    collision_poses[:, 1] += 5.0   # +5m lateral = off into obstacles

    with PDMRewardWrapper(num_workers=1) as wrapper:
        rewards, reward_gt, sub_rewards = wrapper.score(
            collision_poses[None, None],   # (B=1, K=1, N_f, 3)
            gt_poses[None],
            [cache_path],
        )

    nc = sub_rewards["no_collision"][0, 0].item()
    assert nc == 0.0 or rewards[0, 0].item() == 0.0, \
        f"Lateral-offset trajectory should collide; no_collision={nc:.2f}, reward={rewards[0,0]:.4f}"


@pytest.mark.integration
@pytest.mark.skipif(not CACHE_DIR, reason="NAVSIM_CACHE_DIR env var not set")
def test_offroad_trajectory_score():
    """Off-road trajectory (drivable_area=0) should have final sub-score = 0."""
    import glob
    cache_files = glob.glob(os.path.join(CACHE_DIR, "**/*.lzma"), recursive=True)
    assert cache_files

    cache_path = cache_files[0]
    with lzma.open(cache_path, "rb") as f:
        metric_cache = pickle.load(f)

    gt_poses = np.array(metric_cache.trajectory.poses, dtype=np.float32)

    # Off-road trajectory: move far off laterally
    offroad_poses = gt_poses.copy()
    offroad_poses[:, 1] += 20.0   # +20m lateral = clearly off road

    with PDMRewardWrapper(num_workers=1) as wrapper:
        rewards, reward_gt, sub_rewards = wrapper.score(
            offroad_poses[None, None],   # (B=1, K=1, N_f, 3)
            gt_poses[None],
            [cache_path],
        )

    da = sub_rewards["drivable_area"][0, 0].item()
    assert da == 0.0 or rewards[0, 0].item() < 0.3, \
        f"Off-road trajectory should have low/zero score; drivable_area={da:.2f}, reward={rewards[0,0]:.4f}"
