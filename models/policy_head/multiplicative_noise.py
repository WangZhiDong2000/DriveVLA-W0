"""
Multiplicative exploration noise for DriveVLA-W0 GRPO (Task 1.2).

Ports DiffusionDriveV2's 2-scalar multiplicative noise primitive to this repo.
Reference: diffusiondrivev2_model_rl.py:638-666 (DD-v2 scheduler step).

Design constraints (v2.2 §0.2 #10):
- Input MUST be in physical space (meters / radians), NOT the [-1, 1] normalized
  training space. Callers with normalized tensors must denormalize first, apply
  noise, then renormalize. This module does not handle normalization.
- Only (x, y) channels are perturbed; heading passes through unchanged.
- Two independent scalars (ε_long, ε_lat) are sampled per leading-batch element
  and broadcast across all N_f waypoints — matching DD-v2 paper §3.1 which
  explicitly rejects per-element additive Gaussian to avoid jagged paths.

Usage in this repo:
  Task 2.3 — anchor noise endpoint:  x1 = anchor ⊙ (1 + ε),  sigma=σ_anchor
  Task 2.5 — ODE step noise:         x  = x_mean ⊙ (1 + ε),  sigma=σ_step

CLI (vis only, requires cache/anchor_centers_N20.npy from Task 1.1):
    python -m models.policy_head.multiplicative_noise \\
        --anchor-centers cache/anchor_centers_N20.npy \\
        --sigma 0.04 --n-samples 20 \\
        --out cache/multiplicative_vs_additive.png
"""

import argparse
import os

import torch


def apply_multiplicative_noise(
    traj_physical: torch.Tensor,
    sigma: float,
    min_clip: float = 0.04,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply multiplicative noise (1 + ε_mul) to (x, y) channels; heading unchanged.

    Mirrors DD-v2 diffusiondrivev2_model_rl.py:638-666:
      std_dev_t_mul = clip(sigma, min=min_clip)
      ε_long, ε_lat ~ N(0, std_dev_t_mul²)   [one scalar each per leading-batch element]
      noisy = traj_physical * (1 + ε_mul)      [broadcast over waypoints, x/y only]

    Args:
        traj_physical: Shape (*, N_f, 3), physical space (m, m, rad). Any number
                       of leading dims is supported — e.g. (B, N_f, 3) for IL or
                       (B, K, N_f, 3) for GRPO rollout with K candidates.
        sigma:         Standard deviation of N(0, σ²) before clipping.
        min_clip:      Lower bound applied to sigma before sampling (DD-v2 default 0.04).
        generator:     Optional torch.Generator for reproducible sampling.

    Returns:
        noisy_traj: Same shape/dtype/device as traj_physical.
        eps_mul:    Shape (*, 1, 2) — the raw ε values, broadcastable back to
                    (*, N_f, 2). Useful for log-π computation in Task 2.6.
    """
    sigma_eff = max(float(sigma), float(min_clip))
    leading = traj_physical.shape[:-2]  # e.g. (B,) or (B, K)

    eps_xy = torch.randn(
        *leading, 1, 2,
        dtype=traj_physical.dtype,
        device=traj_physical.device,
        generator=generator,
    ) * sigma_eff  # (*, 1, 2)

    xy_noisy = traj_physical[..., :2] * (1.0 + eps_xy)  # broadcast (*, N_f, 2)
    noisy_traj = torch.cat([xy_noisy, traj_physical[..., 2:3]], dim=-1)
    return noisy_traj, eps_xy


# ---------------------------------------------------------------------------
# Visualization helpers — only used by __main__, not part of the public API
# ---------------------------------------------------------------------------

def _load_anchor_centers(path: str) -> torch.Tensor:
    """Load anchor centers from .npy file (Task 1.1 output).

    Args:
        path: Path to .npy file, shape (N_anchor, N_f, 3) float32, physical space.

    Returns:
        torch.Tensor: shape (N_anchor, N_f, 3).
    """
    import numpy as np
    arr = np.load(path).astype("float32")
    return torch.from_numpy(arr)


def _visualize_additive_vs_multiplicative(
    anchor_centers: torch.Tensor,
    sigma: float,
    n_samples: int,
    out_path: str,
) -> None:
    """Two-panel plot: multiplicative vs additive noise on anchor trajectories.

    Reproduces DD-v2 paper Fig. 3 comparison showing that multiplicative noise
    preserves smooth curvature while additive per-element noise produces jagged paths.

    Args:
        anchor_centers: (N_anchor, N_f, 3) physical-space anchor trajectories.
        sigma:          Noise std for both methods (same σ for fair comparison).
        n_samples:      Number of perturbed samples to draw per anchor.
        out_path:       Output PNG path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    N_anchor, N_f, _ = anchor_centers.shape
    n_show = min(5, N_anchor)  # show first 5 anchors to keep the plot readable

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["Multiplicative noise  (1 + ε_mul), smooth", "Additive noise  (ε_add), jagged"]

    for col, ax in enumerate(axes):
        ax.set_title(titles[col], fontsize=11)
        ax.set_xlabel("x (m, forward)")
        ax.set_ylabel("y (m, left)")
        ax.set_aspect("auto")

        for k in range(n_show):
            center = anchor_centers[k]  # (N_f, 3)
            # repeat center for batch sampling: (n_samples, N_f, 3)
            batch = center.unsqueeze(0).expand(n_samples, -1, -1)
            color = plt.cm.tab10(k / 10)

            if col == 0:
                # Multiplicative: use apply_multiplicative_noise
                noisy, _ = apply_multiplicative_noise(batch, sigma=sigma, min_clip=0.0)
                trajs = noisy.numpy()
            else:
                # Additive: independent Gaussian per element on (x, y)
                noise = torch.randn(n_samples, N_f, 2) * sigma
                trajs_xy = batch[..., :2] + noise
                trajs = torch.cat([trajs_xy, batch[..., 2:3]], dim=-1).numpy()

            for i in range(n_samples):
                ax.plot(trajs[i, :, 0], trajs[i, :, 1],
                        color=color, lw=0.8, alpha=0.5)
            # draw center in bold
            ax.plot(center[:, 0].numpy(), center[:, 1].numpy(),
                    color=color, lw=2.5, label=f"anchor #{k}")

        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"Multiplicative vs Additive noise  (σ={sigma}, n={n_samples} per anchor)",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize multiplicative vs additive noise (Task 1.2)"
    )
    parser.add_argument("--anchor-centers", required=True,
                        help="Path to anchor_centers_N20.npy (Task 1.1 output)")
    parser.add_argument("--sigma", type=float, default=0.04,
                        help="Noise std (default: 0.04 = DD-v2 σ_anchor)")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Perturbed samples per anchor to plot")
    parser.add_argument("--out", default="cache/multiplicative_vs_additive.png",
                        help="Output PNG path")
    args = parser.parse_args()

    centers = _load_anchor_centers(args.anchor_centers)
    print(f"Loaded anchor centers: shape={tuple(centers.shape)}, "
          f"x∈[{centers[:,  :, 0].min():.2f}, {centers[:, :, 0].max():.2f}] m, "
          f"y∈[{centers[:, :, 1].min():.2f}, {centers[:, :, 1].max():.2f}] m")

    _visualize_additive_vs_multiplicative(centers, args.sigma, args.n_samples, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
