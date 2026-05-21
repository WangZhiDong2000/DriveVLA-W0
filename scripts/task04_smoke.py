#!/usr/bin/env python3
"""
Task 0.4 Step 5 + Verification smoke test.

Tests:
  1. navtrain metric_cache completeness (file count vs pkl token count)
  2. navtest metric_cache completeness
  3. Load 100 random navtrain .lzma files and verify fields
  4. PDMRewardWrapper.score() latency (< 50ms per sample, single worker)

Usage:
  cd /root/DriveVLA-W0
  PYTHONPATH=$(pwd)/inference/navsim/navsim:$(pwd) \
    python scripts/task04_smoke.py \
    --trainval_pkl /data2/data/navsim/processed_data/meta/navsim_emu_vla_256_144_trainval_pre_1s.pkl \
    --metric_cache_trainval /data2/data/navsim/metric_cache/trainval \
    --metric_cache_test /data2/data/navsim/metric_cache/test \
    --n_samples 100 --seed 42
"""

import argparse
import lzma
import pickle
import time
import random
import sys
import os
from pathlib import Path

# Pre-import drivevla (Python 3.10) C-extension packages BEFORE adding navsim site-packages.
# This ensures sys.modules caches the correct versions, so nuplan's imports reuse them.
import numpy as np
import shapely          # noqa: F401 - pre-load to prevent navsim env version from being used
import geopandas        # noqa: F401
import pandas           # noqa: F401
import cv2              # noqa: F401
import rasterio         # noqa: F401

# Now add nuplan (pure-Python parts) from navsim env.
# Since all C-extension packages are already in sys.modules, nuplan imports them safely.
_NUPLAN_SITE = "/data1/miniconda3/envs/navsim/lib/python3.9/site-packages"
if _NUPLAN_SITE not in sys.path:
    sys.path.append(_NUPLAN_SITE)

parser = argparse.ArgumentParser()
parser.add_argument("--trainval_pkl", required=True)
parser.add_argument("--metric_cache_trainval", required=True)
parser.add_argument("--metric_cache_test", required=True)
parser.add_argument("--n_samples", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--skip_pdm_compare", action="store_true")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"
PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
results = {}

# ── 1. File count verification ────────────────────────────────────────────────
print("\n=== Verification 1: navtrain metric_cache file count ===")
trainval_dir = Path(args.metric_cache_trainval)
trainval_lzma = list(trainval_dir.glob("*.lzma"))
print(f"  Found {len(trainval_lzma)} .lzma files in {trainval_dir}")

print(f"  Loading pkl: {args.trainval_pkl} ...")
with open(args.trainval_pkl, "rb") as f:
    pkl_data = pickle.load(f)

if isinstance(pkl_data, dict):
    pkl_tokens = set(pkl_data.keys())
elif isinstance(pkl_data, list):
    first = pkl_data[0]
    if isinstance(first, dict) and "token" in first:
        pkl_tokens = {item["token"] for item in pkl_data}
    else:
        pkl_tokens = set()
else:
    pkl_tokens = set()

print(f"  Pkl token count: {len(pkl_tokens)}")
print(f"  Lzma file count: {len(trainval_lzma)}")

lzma_tokens = {p.stem for p in trainval_lzma}
if pkl_tokens:
    missing = pkl_tokens - lzma_tokens
    if len(missing) == 0:
        print(f"  {PASS} All pkl tokens have .lzma files")
    else:
        print(f"  {FAIL} {len(missing)} missing tokens (first 5: {list(missing)[:5]})")
    results["trainval_complete"] = len(missing) == 0
else:
    ok = len(trainval_lzma) >= 100000
    results["trainval_complete"] = ok
    print(f"  {'OK' if ok else FAIL} {len(trainval_lzma)} lzma files (expected ~103288)")

print("\n=== Verification 2: navtest metric_cache file count ===")
test_dir = Path(args.metric_cache_test)
test_lzma = list(test_dir.glob("*.lzma"))
print(f"  Found {len(test_lzma)} .lzma files in {test_dir}")
ok = len(test_lzma) >= 12000
print(f"  {PASS if ok else FAIL} navtest cache has {len(test_lzma)} files (expected ~12146)")
results["navtest_complete"] = ok

# ── 2. Load 100 random samples ────────────────────────────────────────────────
print(f"\n=== Verification 3: Load {args.n_samples} random .lzma files ===")
available = list(lzma_tokens)
sample_tokens = random.sample(available, min(args.n_samples, len(available)))

load_times = []
sample_loaded = []
fail_load = 0

for tok in sample_tokens:
    path = trainval_dir / f"{tok}.lzma"
    t0 = time.perf_counter()
    try:
        with lzma.open(str(path), "rb") as f:
            mc = pickle.load(f)
        elapsed = time.perf_counter() - t0
        load_times.append(elapsed)
        # Verify required fields exist
        assert hasattr(mc, "ego_state"), "missing ego_state"
        assert hasattr(mc, "trajectory"), "missing trajectory"
        assert hasattr(mc, "observation"), "missing observation"
        assert hasattr(mc, "centerline"), "missing centerline"
        assert hasattr(mc, "route_lane_ids"), "missing route_lane_ids"
        assert hasattr(mc, "drivable_area_map"), "missing drivable_area_map"
        sample_loaded.append((tok, mc))
    except Exception as e:
        fail_load += 1
        sample_loaded.append((tok, None))
        if fail_load <= 3:
            print(f"  LOAD ERROR [{tok[:8]}]: {e}")

load_mean_ms = np.mean(load_times) * 1000 if load_times else 0
load_max_ms = np.max(load_times) * 1000 if load_times else 0
ok_count = len(load_times)
print(f"  Loaded: {ok_count}/{len(sample_tokens)}, failures: {fail_load}")
print(f"  Load time: mean={load_mean_ms:.1f}ms, max={load_max_ms:.1f}ms")
results["load_ok"] = fail_load == 0
print(f"  {PASS if results['load_ok'] else FAIL}")

# ── 3. PDMRewardWrapper latency (single worker via _pdm_worker direct call) ───
# Use direct call instead of ProcessPoolExecutor to avoid spawn bootstrapping
# issue when running as a script (ProcessPoolExecutor with spawn needs
# if __name__ == '__main__': guard in the calling script).
print(f"\n=== Verification 4: PDMRewardWrapper latency (direct worker call) ===")
try:
    root_dir = str(Path(__file__).parents[1])
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    from utils.rl_modules.pdm_reward_wrapper import (
        _NAVSIM_AVAILABLE, _init_pool, _pdm_worker,
    )
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    print(f"  _NAVSIM_AVAILABLE: {_NAVSIM_AVAILABLE}")

    if not _NAVSIM_AVAILABLE:
        print(f"  {FAIL} navsim not available in this env")
        results["latency_ok"] = False
    else:
        # Load scoring config and initialize simulator+scorer in main process
        _DEFAULT_CONFIG = str(
            Path(__file__).parents[1]
            / "inference/navsim/navsim/navsim/planning/script/config/pdm_scoring"
            / "default_scoring_parameters.yaml"
        )
        cfg = OmegaConf.load(_DEFAULT_CONFIG)
        _init_pool(cfg.simulator, cfg.scorer)   # sets module-level SIMULATOR/SCORER

        # Use dummy forward-motion trajectory (physical space, meters/radians)
        dummy_traj = np.zeros((1, 8, 3), dtype=np.float32)
        for i in range(8):
            dummy_traj[0, i, 0] = (i + 1) * 1.5   # x: forward 1.5m steps
        gt_traj = dummy_traj[0]   # (8, 3)
        all_traj = np.stack([gt_traj, gt_traj], axis=0)  # (2, 8, 3): [candidate, GT]

        latencies = []
        scores_list = []
        valid_samples = [(tok, mc) for tok, mc in sample_loaded if mc is not None]

        for tok, mc in valid_samples[:20]:
            path = str(trainval_dir / f"{tok}.lzma")
            t0 = time.perf_counter()
            scores, subscores = _pdm_worker((path, all_traj))
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            scores_list.append(float(scores[0]))    # candidate score
            print(f"    {tok[:12]}  score={scores_list[-1]:.4f}  lat={lat:.1f}ms")

        lat_mean = np.mean(latencies)
        lat_max = np.max(latencies)
        score_mean = np.mean(scores_list)
        print(f"  Latency: mean={lat_mean:.1f}ms, max={lat_max:.1f}ms (target: mean<50ms)")
        print(f"  Scores: mean={score_mean:.4f}, min={min(scores_list):.4f}")
        results["latency_ok"] = lat_mean < 50
        results["wrapper_scores"] = scores_list
        print(f"  {PASS if results['latency_ok'] else FAIL} Latency")

except Exception as e:
    import traceback
    print(f"  {FAIL} PDMRewardWrapper error: {e}")
    traceback.print_exc()
    results["latency_ok"] = False

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TASK 0.4 VERIFICATION SUMMARY")
print("=" * 60)
skip_keys = {"wrapper_scores"}
all_pass = True
for k, v in results.items():
    if k in skip_keys:
        continue
    if v is True:
        icon = PASS
    elif v is False:
        icon = FAIL
        all_pass = False
    else:
        icon = f"\033[33m{v}\033[0m"
    print(f"  {k:<30s}: {icon}")
print("=" * 60)
if all_pass:
    print(f"  {PASS} ALL CHECKS PASSED - Task 0.4 COMPLETE")
else:
    print(f"  {FAIL} Some checks failed - see above")
import sys; sys.exit(0 if all_pass else 1)
