"""
Offline K-Means anchor clustering for DriveVLA-W0 GRPO (Task 1.1).

Ports DiffusionDriveV2's anchor generation logic to the navsim mini dataset.
The produced .npy file is a drop-in replacement for DD-v2's kmeans_navsim_traj_20.npy
and is consumed by Task 2.2 (AnchorEmbedding) and Task 2.3 (AnchoredFlowPath).

Usage:
    python -m models.policy_head.anchor_kmeans \
        --pkl /path/to/navsim_emu_vla_256_144_mini_pre_1s.pkl \
        --norm-dir configs/normalizer_navsim_mini \
        --n-anchor 20 --n-waypoints 8 --seed 0 \
        --out cache/anchor_centers_N20.npy \
        --vis-out cache/anchor_centers_N20.png
"""

import argparse
import json
import math
import os
import pickle
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Action window constants matching pickle_generation_navsim_mini.py
_CENTER_IDX = 3   # index of current ego frame in the 12-step action window
_ACTION_DIM = 3   # (dx, dy, dtheta)


def _accumulate_se2(increments: np.ndarray) -> np.ndarray:
    """Accumulate per-step SE2 increments into cumulative positions.

    Each increment is expressed in the coordinate frame of the *previous* waypoint,
    matching the rel_all[j, j+1] convention in pickle_generation_navsim_mini.py.

    Args:
        increments: shape (T, 3), physical-space (dx_m, dy_m, dtheta_rad) per step.

    Returns:
        np.ndarray: shape (T, 3), cumulative (x, y, theta) in the starting ego frame.
    """
    T = len(increments)
    positions = np.empty((T, 3), dtype=np.float64)
    pos = np.zeros(3)
    for k, (dx, dy, dtheta) in enumerate(increments):
        c, s = math.cos(pos[2]), math.sin(pos[2])
        pos = np.array([
            pos[0] + dx * c - dy * s,
            pos[1] + dx * s + dy * c,
            pos[2] + dtheta,
        ])
        positions[k] = pos
    return positions


def load_mini_trajectories(
    pkl_path: str,
    norm_dir: str,
    n_waypoints: int = 8,
) -> np.ndarray:
    """Load GT trajectories from preprocessed pickle and return cumulative (x, y).

    The pickle stores per-step SE2 increments in [-1, 1] normalized space.
    This function denormalizes them to meters/radians and accumulates SE2 transforms
    starting from CENTER_IDX (current ego frame) to produce cumulative positions.

    Args:
        pkl_path: Path to preprocessed pickle file.
        norm_dir: Directory containing libero_q01.npy and libero_q99.npy.
        n_waypoints: Number of future waypoints to use (typically 8).

    Returns:
        np.ndarray: shape (N, n_waypoints, 2), cumulative (x, y) in meters,
                    in the current ego frame (same space as DD-v2 anchor clusters).
    """
    q01 = np.load(os.path.join(norm_dir, "libero_q01.npy")).astype(np.float64)
    q99 = np.load(os.path.join(norm_dir, "libero_q99.npy")).astype(np.float64)

    with open(pkl_path, "rb") as f:
        scenes = pickle.load(f)

    trajs = []
    for scene in scenes:
        a = np.array(scene["action"], dtype=np.float64)   # (12, 3) normalized [-1, 1]
        # Inverse of: normalized = 2*(physical - q01)/(q99 - q01 + 1e-8) - 1
        physical = (a + 1.0) / 2.0 * (q99 - q01) + q01   # (12, 3) meters/radians

        # Slice future n_waypoints steps starting from CENTER_IDX
        future = physical[_CENTER_IDX: _CENTER_IDX + n_waypoints]  # (n_waypoints, 3)
        if len(future) < n_waypoints:
            continue  # skip truncated edge frames

        cumulative = _accumulate_se2(future)   # (n_waypoints, 3) cumulative (x,y,theta)
        trajs.append(cumulative[:, :2])        # (n_waypoints, 2) keep (x, y) only

    result = np.stack(trajs, axis=0).astype(np.float32)   # (N, n_waypoints, 2)
    print(f"Loaded {len(result)} trajectories from {os.path.basename(pkl_path)}")
    print(f"  x range: [{result[...,0].min():.2f}, {result[...,0].max():.2f}] m")
    print(f"  y range: [{result[...,1].min():.2f}, {result[...,1].max():.2f}] m")
    return result


def bezier_xyyaw_numpy(xy: np.ndarray) -> np.ndarray:
    """Reconstruct heading via 8th-order Bezier curve derivative + atan2.

    Numpy port of DiffusionDriveV2 bezier_xyyaw
    (diffusiondrivev2_model_rl.py:1123-1171).
    A fixed origin (0, 0) is prepended to form the control polygon.
    Heading at waypoint k = atan2(dy, dx) of the curve's first derivative at t_k.

    Args:
        xy: shape (N, 8, 2), cumulative (x, y) waypoints in meters.

    Returns:
        np.ndarray: shape (N, 8, 3), (x, y, yaw) with yaw in radians. dtype float32.
    """
    N, n_f, _ = xy.shape
    assert n_f == 8, f"bezier_xyyaw_numpy expects 8 waypoints, got {n_f}"

    xy64 = xy.astype(np.float64)

    # Prepend origin (0,0) → control polygon (N, 9, 2)
    origin = np.zeros((N, 1, 2), dtype=np.float64)
    ctrl = np.concatenate([origin, xy64], axis=1)   # (N, 9, 2)
    n = ctrl.shape[1] - 1                            # degree = 8

    delta = ctrl[:, 1:, :] - ctrl[:, :-1, :]        # (N, 8, 2)  ΔP_i

    # Bernstein basis for degree-(n-1) polynomial used in first derivative
    binom = np.array([math.comb(n - 1, i) for i in range(n)], dtype=np.float64)   # (8,)
    t = np.arange(1, n + 1, dtype=np.float64) / n                                 # (8,)

    t_pow   = t[:, None] ** np.arange(0, n, dtype=np.float64)[None, :]            # (8,8)
    one_pow = (1 - t)[:, None] ** np.arange(n - 1, -1, -1, dtype=np.float64)[None, :]  # (8,8)
    basis   = binom[None, :] * t_pow * one_pow                                    # (8,8)

    # B'(t_k) = n * Σ_i basis[k,i] * ΔP_i  →  deriv (N, 8, 2)
    deriv = n * np.einsum("ki,sic->skc", basis, delta)

    yaw = np.arctan2(deriv[..., 1], deriv[..., 0])[..., None]   # (N, 8, 1)
    return np.concatenate([xy64, yaw], axis=-1).astype(np.float32)   # (N, 8, 3)


def fit(
    trajectories_xy: np.ndarray,
    n_anchor: int = 20,
    n_waypoints: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """K-Means anchor clustering on (x, y) trajectories.

    Args:
        trajectories_xy: shape (N, n_waypoints, 2), cumulative (x, y) in meters.
        n_anchor: Number of cluster centers.
        n_waypoints: Number of waypoints per trajectory (must be 8 for Bezier heading).
        seed: Random seed for KMeans reproducibility.

    Returns:
        Tuple of:
          - centers_xyyaw: np.ndarray, shape (n_anchor, n_waypoints, 3) float32,
                           anchor centers with heading reconstructed via Bezier.
          - labels: np.ndarray, shape (N,) int32, cluster assignment per trajectory.
    """
    N = trajectories_xy.shape[0]
    features = trajectories_xy.reshape(N, -1).astype(np.float64)   # (N, n_waypoints*2)

    km = KMeans(n_clusters=n_anchor, n_init=20, random_state=seed)
    km.fit(features)
    labels = km.labels_.astype(np.int32)

    centers_xy = km.cluster_centers_.reshape(n_anchor, n_waypoints, 2).astype(np.float32)
    centers_xyyaw = bezier_xyyaw_numpy(centers_xy)   # (n_anchor, n_waypoints, 3)
    return centers_xyyaw, labels


def compute_silhouette_and_sanity(
    features: np.ndarray,
    labels: np.ndarray,
    centers_flat: np.ndarray,
) -> dict:
    """Compute silhouette score and geometric sanity metrics.

    Args:
        features: (N, D) flattened trajectory features.
        labels: (N,) cluster assignment indices.
        centers_flat: (n_anchor, D) cluster centers in feature space.

    Returns:
        dict with keys: silhouette (float), min_cluster_size (int),
        min_inter_cluster_dist (float), cluster_sizes (list[int]).
    """
    n_anchor = centers_flat.shape[0]
    sil = float(silhouette_score(features, labels))
    cluster_sizes = [int((labels == k).sum()) for k in range(n_anchor)]
    dists = pdist(centers_flat)
    return {
        "silhouette": round(sil, 4),
        "min_cluster_size": int(min(cluster_sizes)),
        "min_inter_cluster_dist": round(float(dists.min()), 3),
        "cluster_sizes": cluster_sizes,
    }


def visualize_anchors(
    trajectories_xy: np.ndarray,
    labels: np.ndarray,
    centers_xyyaw: np.ndarray,
    out_path: str,
) -> None:
    """Save a two-panel visualization of K-Means anchor results.

    Left panel: all GT trajectories (thin grey) + cluster centers (thick colored).
    Right panel: per-cluster sample overlays (up to 50 per cluster).

    Args:
        trajectories_xy: (N, 8, 2) cumulative (x, y) trajectories in meters.
        labels: (N,) cluster assignments.
        centers_xyyaw: (n_anchor, 8, 3) cluster centers.
        out_path: Output PNG path.
    """
    n_anchor = centers_xyyaw.shape[0]
    cmap = matplotlib.colormaps.get_cmap("tab20")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left: overview ---
    ax = axes[0]
    for traj in trajectories_xy:
        ax.plot(traj[:, 0], traj[:, 1], color="lightgrey", lw=0.4, alpha=0.5)
    for k in range(n_anchor):
        c = centers_xyyaw[k]
        ax.plot(c[:, 0], c[:, 1], lw=2.5, color=cmap(k), label=f"#{k}")
    ax.set_aspect("equal")
    ax.set_title("All GT trajectories + cluster centers", fontsize=10)
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, left)")
    ax.legend(loc="upper left", fontsize=5, ncol=2)

    # --- Right: per-cluster samples ---
    ax2 = axes[1]
    for k in range(n_anchor):
        idxs = np.where(labels == k)[0]
        sample_idxs = idxs[:50]
        for i in sample_idxs:
            traj = trajectories_xy[i]
            ax2.plot(traj[:, 0], traj[:, 1], color=cmap(k), lw=0.5, alpha=0.4)
        c = centers_xyyaw[k]
        ax2.plot(c[:, 0], c[:, 1], lw=2.5, color=cmap(k))
    ax2.set_aspect("equal")
    ax2.set_title("Per-cluster samples (≤50 each) + centers", fontsize=10)
    ax2.set_xlabel("x (m, forward)")
    ax2.set_ylabel("y (m, left)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


def _pass_check(metrics: dict) -> bool:
    """Print PASS/FAIL report and return True if all criteria are met."""
    ok = True
    checks = [
        ("silhouette >= 0.15",      metrics["silhouette"] >= 0.15),
        ("min_cluster_size >= 1",   metrics["min_cluster_size"] >= 1),
        ("min_inter_cluster_dist > 5.0 m",
         metrics["min_inter_cluster_dist"] > 5.0),
    ]
    print("\n--- Task 1.1 PASS criteria ---")
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description="K-Means anchor clustering (Task 1.1)")
    parser.add_argument("--pkl",       required=True,  help="Path to preprocessed pickle")
    parser.add_argument("--norm-dir",  required=True,  help="Directory with libero_q01/q99.npy")
    parser.add_argument("--n-anchor",  type=int, default=20)
    parser.add_argument("--n-waypoints", type=int, default=8)
    parser.add_argument("--seed",      type=int, default=0)
    parser.add_argument("--out",       default="cache/anchor_centers_N20.npy")
    parser.add_argument("--vis-out",   default="cache/anchor_centers_N20.png")
    args = parser.parse_args()

    # 1. Load and denormalize trajectories
    trajs_xy = load_mini_trajectories(args.pkl, args.norm_dir, args.n_waypoints)
    N = trajs_xy.shape[0]

    # 2. K-Means clustering
    print(f"\nRunning KMeans(k={args.n_anchor}, n_init=20, seed={args.seed}) on {N} trajectories...")
    centers_xyyaw, labels = fit(trajs_xy, args.n_anchor, args.n_waypoints, args.seed)

    # 3. Silhouette & sanity
    features = trajs_xy.reshape(N, -1).astype(np.float64)
    centers_flat = centers_xyyaw[:, :, :2].reshape(args.n_anchor, -1).astype(np.float64)
    metrics = compute_silhouette_and_sanity(features, labels, centers_flat)
    print(f"\nSilhouette score:           {metrics['silhouette']:.4f}")
    print(f"Min cluster size:           {metrics['min_cluster_size']}")
    print(f"Min inter-cluster dist:     {metrics['min_inter_cluster_dist']:.2f} m")
    print(f"Cluster sizes (k=0..{args.n_anchor-1}):  {metrics['cluster_sizes']}")

    # 4. Physical range check
    cx = centers_xyyaw[:, :, 0]
    cy = centers_xyyaw[:, :, 1]
    cyaw = centers_xyyaw[:, :, 2]
    print(f"\nCenter x range:  [{cx.min():.2f}, {cx.max():.2f}] m  (expect < 60 m)")
    print(f"Center y range:  [{cy.min():.2f}, {cy.max():.2f}] m  (expect |·| < 15 m)")
    print(f"Center yaw range:[{cyaw.min():.3f}, {cyaw.max():.3f}] rad  (expect |·| < π)")

    _pass_check(metrics)

    # 5. Save cache
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, centers_xyyaw)
    print(f"\nSaved anchor centers → {args.out}  shape={centers_xyyaw.shape} dtype={centers_xyyaw.dtype}")

    meta = {
        "n_anchor": args.n_anchor,
        "n_waypoints": args.n_waypoints,
        "seed": args.seed,
        "n_samples": N,
        **metrics,
        "source_pkl": args.pkl,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = args.out + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata          → {meta_path}")

    # 6. Visualization
    visualize_anchors(trajs_xy, labels, centers_xyyaw, args.vis_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
