"""
PDM reward wrapper for DriveVLA-W0 GRPO (Task 1.5).

Wraps the NAVSIM PDM scorer as a batched reward function, paralleling
DiffusionDriveV2's get_pdm_score_para (model_rl.py:785-798).

Interface: PDMRewardWrapper.score(trajectories_phys, gt_trajectories_phys,
                                   metric_cache_paths)
  → rewards (B, K), reward_gt (B,), sub_rewards dict

Key differences from DD-v2:
  - local navsim has no pdm_score_para; batch logic is inlined in _pdm_worker
  - local navsim MultiMetricIndex has 2 entries (not 3); _multi_metrics is (2, N)
  - caller is responsible for denormalization before passing to this wrapper
"""

from __future__ import annotations

import lzma
import multiprocessing as mp
import pickle
import concurrent.futures as cf
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Pre-load drivevla env's C-extension packages before any navsim/nuplan imports.
# nuplan depends on shapely/geopandas/pandas/cv2/rasterio compiled for Python 3.9.
# By importing drivevla's Python 3.10 versions first, sys.modules caches them and
# nuplan reuses the correct versions. NUPLAN_SITE is appended (not prepended) to
# sys.path so it only provides nuplan (pure Python), never overrides drivevla's packages.
import sys as _sys
try:
    import shapely as _shapely      # noqa: F401
    import geopandas as _geopandas  # noqa: F401
    import pandas as _pandas        # noqa: F401
    import cv2 as _cv2              # noqa: F401
    import rasterio as _rasterio    # noqa: F401
    _NUPLAN_SITE = "/data1/miniconda3/envs/navsim/lib/python3.9/site-packages"
    if _NUPLAN_SITE not in _sys.path:
        _sys.path.append(_NUPLAN_SITE)
except ImportError:
    pass  # Packages absent -> navsim imports below fail -> _NAVSIM_AVAILABLE=False

# navsim / omegaconf / hydra are only available in the full NAVSIM environment.
# They are imported lazily inside functions/methods so that the module can be
# imported and mock-tested without those packages installed.
try:
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        MultiMetricIndex,
        WeightedMetricIndex,
    )
    _NAVSIM_AVAILABLE = True
except ImportError:
    _NAVSIM_AVAILABLE = False
    # Provide sentinel integer constants so _pairwise_scores/_pairwise_subscores
    # can be unit-tested with a mock scorer even without navsim installed.
    class _MultiMetricIndex:
        NO_COLLISION = 0
        DRIVABLE_AREA = 1

    class _WeightedMetricIndex:
        PROGRESS = 0
        TTC = 1
        COMFORTABLE = 2
        DRIVING_DIRECTION = 3

    MultiMetricIndex = _MultiMetricIndex
    WeightedMetricIndex = _WeightedMetricIndex

# ---------------------------------------------------------------------------
# Worker-global singletons (one PDMSimulator + PDMScorer per worker process)
# ---------------------------------------------------------------------------

SIMULATOR = None
SCORER = None

_DEFAULT_CONFIG = str(
    Path(__file__).parents[2]
    / "inference/navsim/navsim/navsim/planning/script/config/pdm_scoring"
    / "default_scoring_parameters.yaml"
)


def _init_pool(sim_cfg, scorer_cfg) -> None:
    global SIMULATOR, SCORER
    SIMULATOR = instantiate(sim_cfg)
    SCORER = instantiate(scorer_cfg)


# ---------------------------------------------------------------------------
# Score extraction helpers (ported from DD-v2 model_rl.py:38-117,
# adapted for local navsim where _multi_metrics is (2, N) not (3, N))
# ---------------------------------------------------------------------------

def _pairwise_scores(scorer) -> np.ndarray:
    """Compute pairwise PDM scores for proposals [1:] vs PDM reference [0].

    Mirrors DD-v2 model_rl.py:75-117 with local navsim's (2, N) multi-metrics.

    Args:
        scorer: PDMScorer after score_proposals() has been called.
                scorer._multi_metrics: (2, N)  NO_COLLISION × DRIVABLE_AREA
                scorer._weighted_metrics: (4, N) PROGRESS/TTC/COMFORTABLE/DIR
                scorer._progress_raw: (N,)

    Returns:
        np.ndarray: shape (N-1,) float32, scores for proposals [1:].
                    Index 0 (PDM reference trajectory) is excluded.
    """
    mm = scorer._multi_metrics                    # (2, N)
    wm = scorer._weighted_metrics.copy()          # (4, N) — must copy
    prog_raw = scorer._progress_raw               # (N,)
    weight_coef = scorer._config.weighted_metrics_array   # (4,) numpy array
    thresh = scorer._config.progress_distance_threshold

    multi_prod = mm.prod(axis=0)                  # (N,)

    # Pairwise progress normalisation vs PDM reference (index 0)
    raw_prog = prog_raw * multi_prod              # (N,)
    raw_prog_ref = raw_prog[0]                    # PDM reference progress
    max_pair = np.maximum(raw_prog_ref, raw_prog[1:])  # (N-1,)

    norm_prog = np.where(
        max_pair > thresh,
        raw_prog[1:] / (max_pair + 1e-6),
        np.where(multi_prod[1:] == 0.0, 0.0, 1.0),
    ).astype(np.float64)

    wm[WeightedMetricIndex.PROGRESS, 1:] = norm_prog

    weighted_scores = (wm[:, 1:] * weight_coef[:, None]).sum(axis=0)  # (N-1,)
    weighted_scores /= weight_coef.sum()

    final_scores = multi_prod[1:] * weighted_scores  # (N-1,)
    return final_scores.astype(np.float32)


def _pairwise_subscores(scorer) -> Dict[str, np.ndarray]:
    """Extract per-proposal sub-metric arrays (shape (N-1,) each).

    Mirrors DD-v2 model_rl.py:38-74, adapted for (2, N) multi-metrics.

    Returns:
        dict with keys: no_collision, drivable_area, progress, ttc,
                        comfort, dir_weighted, final.
        Each value is shape (N-1,) float64 (proposals [1:]).
    """
    mm = scorer._multi_metrics                    # (2, N)
    wm = scorer._weighted_metrics.copy()          # (4, N) — must copy
    prod = mm.prod(axis=0)                        # (N,)

    weight_coef = scorer._config.weighted_metrics_array
    thresh = scorer._config.progress_distance_threshold
    prog_raw = scorer._progress_raw               # (N,)

    # Pairwise progress normalisation (same as _pairwise_scores)
    raw_prog = prog_raw * prod
    raw_prog_ref = raw_prog[0]
    max_pair = np.maximum(raw_prog_ref, raw_prog[1:])
    norm_prog = np.where(
        max_pair > thresh,
        raw_prog[1:] / (max_pair + 1e-6),
        np.where(prod[1:] == 0.0, 0.0, 1.0),
    ).astype(np.float64)
    wm[WeightedMetricIndex.PROGRESS, 1:] = norm_prog

    wscore = (wm * weight_coef[:, None]).sum(axis=0) / weight_coef.sum()  # (N,)

    return {
        "no_collision":  mm[MultiMetricIndex.NO_COLLISION,         1:].copy(),
        "drivable_area": mm[MultiMetricIndex.DRIVABLE_AREA,        1:].copy(),
        "progress":      wm[WeightedMetricIndex.PROGRESS,          1:].copy(),
        "ttc":           wm[WeightedMetricIndex.TTC,               1:].copy(),
        "comfort":       wm[WeightedMetricIndex.COMFORTABLE,       1:].copy(),
        "dir_weighted":  wm[WeightedMetricIndex.DRIVING_DIRECTION, 1:].copy(),
        "final":         prod[1:] * wscore[1:],
    }


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess — no CUDA, no torch imports needed)
# ---------------------------------------------------------------------------

def _pdm_worker(args: Tuple) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Score K+1 trajectories (K candidates + GT) for one scene.

    The caller concatenates candidates and GT into all_traj_np:
      all_traj_np[-1] == GT trajectory,  all_traj_np[:-1] == candidates.

    Internally prepends the PDM reference trajectory so the trajectory_states
    layout matches pdm_score_para (reference/DiffusionDriveV2/…/pdm_score.py:205):
      trajectory_states[0]   = PDM reference  (for progress normalisation)
      trajectory_states[1:K+1] = candidates
      trajectory_states[K+1] = GT

    Returns:
        scores:    (K+1,) float32 — candidates + GT (PDM reference excluded)
        subscores: dict[str, (K+1,) float64]
    """
    cache_path, all_traj_np = args
    # all_traj_np: (K+1, N_f, 3) physical space; GT is the last row

    with lzma.open(cache_path, "rb") as f:
        metric_cache = pickle.load(f)

    initial_ego_state = metric_cache.ego_state

    # PDM reference states (used for progress normalisation)
    pdm_states = get_trajectory_as_array(
        metric_cache.trajectory,
        SIMULATOR.proposal_sampling,
        initial_ego_state.time_point,
    )

    # Convert all K+1 physical trajectories → simulated state arrays
    pred_states_list = []
    for i in range(len(all_traj_np)):
        traj = Trajectory(all_traj_np[i].astype(np.float32))
        pred_world = transform_trajectory(traj, initial_ego_state)
        pred_states_list.append(
            get_trajectory_as_array(
                pred_world, SIMULATOR.proposal_sampling, initial_ego_state.time_point
            )
        )
    pred_states_batch = np.stack(pred_states_list, axis=0)   # (K+1, T+1, dim)

    # Stack: [PDM ref | candidates | GT] → (K+2, T+1, dim)
    trajectory_states = np.concatenate(
        [pdm_states[None], pred_states_batch], axis=0
    )

    simulated_states = SIMULATOR.simulate_proposals(trajectory_states, initial_ego_state)
    SCORER.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
    )

    scores = _pairwise_scores(SCORER)        # (K+1,) candidates + GT
    subscores = _pairwise_subscores(SCORER)  # dict[str, (K+1,)]
    return scores, subscores


# ---------------------------------------------------------------------------
# Public wrapper class
# ---------------------------------------------------------------------------

class PDMRewardWrapper:
    """Batched PDM reward function using a ProcessPoolExecutor.

    Usage::

        wrapper = PDMRewardWrapper(num_workers=8)
        rewards, reward_gt, sub_rewards = wrapper.score(
            trajectories_phys,      # (B, K, N_f, 3) physical space
            gt_trajectories_phys,   # (B, N_f, 3)    physical space
            metric_cache_paths,     # List[str], len=B
        )
        wrapper.shutdown()
    """

    def __init__(
        self,
        num_workers: int = 8,
        scoring_config_path: str = _DEFAULT_CONFIG,
    ) -> None:
        if not _NAVSIM_AVAILABLE:
            raise ImportError(
                "navsim/omegaconf/hydra are not installed in the current environment. "
                "PDMRewardWrapper requires the full NAVSIM environment."
            )
        cfg = OmegaConf.load(scoring_config_path)
        self._sim_cfg = cfg.simulator
        self._scorer_cfg = cfg.scorer
        self._pool = cf.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_pool,
            initargs=(self._sim_cfg, self._scorer_cfg),
        )

    def score(
        self,
        trajectories_phys: np.ndarray,
        gt_trajectories_phys: np.ndarray,
        metric_cache_paths: List[str],
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Score K candidate trajectories per scene via PDM pool.

        Args:
            trajectories_phys:    (B, K, N_f, 3) float32, physical space (m/rad).
            gt_trajectories_phys: (B, N_f, 3)    float32, GT physical space.
            metric_cache_paths:   List[str] of length B, paths to .lzma files.
            device:               Target device for returned tensors.

        Returns:
            rewards:    (B, K) PDM scores for candidates ∈ [0, 1].
            reward_gt:  (B,)   PDM score for GT trajectory.
            sub_rewards: dict with keys no_collision, drivable_area, progress,
                         ttc, comfort, dir_weighted, final; each (B, K).
        """
        B, K = trajectories_phys.shape[:2]

        # Submit one job per scene; GT is appended as the last trajectory
        futures = []
        for b in range(B):
            all_traj = np.concatenate(
                [trajectories_phys[b], gt_trajectories_phys[b : b + 1]], axis=0
            )  # (K+1, N_f, 3), GT last
            futures.append(
                self._pool.submit(_pdm_worker, (metric_cache_paths[b], all_traj))
            )

        # Collect results
        all_scores: List[np.ndarray] = []
        all_gt: List[float] = []
        all_sub: List[Dict[str, np.ndarray]] = []
        for f in futures:
            scores, subscores = f.result()   # (K+1,), dict[str, (K+1,)]
            all_scores.append(scores[:-1])   # (K,) candidates
            all_gt.append(float(scores[-1])) # scalar GT score
            all_sub.append(subscores)

        rewards = torch.tensor(
            np.stack(all_scores), device=device, dtype=torch.float32
        )  # (B, K)
        reward_gt = torch.tensor(
            all_gt, device=device, dtype=torch.float32
        )  # (B,)
        sub_rewards: Dict[str, torch.Tensor] = {
            k: torch.tensor(
                np.stack([d[k][:-1] for d in all_sub]),  # (B, K) remove GT col
                device=device,
                dtype=torch.float32,
            )
            for k in all_sub[0].keys()
        }
        return rewards, reward_gt, sub_rewards

    def shutdown(self) -> None:
        """Gracefully shut down the worker pool."""
        self._pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()
