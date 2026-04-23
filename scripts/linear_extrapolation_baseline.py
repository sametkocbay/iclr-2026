"""Linear-extrapolation baseline for one-step-ahead prediction.

Predicts output step 1 (= simulation step 6) by linear extrapolation:

    pred_t6 = velocity_in[:, 4] + (velocity_in[:, 4] - velocity_in[:, 3])
            = 2 * step5 - step4

Evaluates:
  • Relative-L2 (RL2) in the *scaled* space  (same as training loss)
  • Hint metric (mean ||pred - true||₂ per point) in the *unscaled* space
  • 3-D scatter plot of per-point velocity-error magnitude in physical units

Usage
-----
    python scripts/linear_extrapolation_baseline.py \
        --dataset-dir dataset_huggingface/warped-ifw \
        --num-samples 20 \
        --num-points 100000 \
        --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow importing project modules when run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    WarpedIFWDataset,
    build_loader,
    split_train_val_test,
    unscale_velocity_batch,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """RL2 = ||pred - target||_F / ||target||_F  (over all dims)."""
    return (
        (pred - target).norm() / target.norm().clamp(min=1e-8)
    ).item()


def hint_metric_per_point(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return per-point L2 error: shape (N,)."""
    # pred/target: (T_out, N, C)  → norm over C
    return (pred - target).norm(dim=-1).mean(dim=0)  # (N,)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    files = sorted(dataset_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {dataset_dir}")

    _, val_files, _ = split_train_val_test(
        files,
        val_fraction=0.1,
        test_fraction=0.1,
        seed=args.seed,
    )

    dataset = WarpedIFWDataset(
        val_files,
        num_points=args.num_points,
        random_crop=False,
        seed=args.seed,
        scaler_eps=1e-6,
    )
    n_samples = min(args.num_samples, len(dataset))
    print(f"Evaluating linear-extrapolation baseline on {n_samples} val samples "
          f"(num_points={args.num_points})")

    all_rl2: list[float] = []
    all_hint: list[float] = []

    # Keep last sample for the scatter plot.
    last_pos: torch.Tensor | None = None
    last_error_per_point: torch.Tensor | None = None
    last_airfoil_idcs: torch.Tensor | None = None

    for idx in range(n_samples):
        (
            t,
            pos,
            idcs_airfoil,
            velocity_in,     # (T_in, N, 3)  scaled
            velocity_out,    # (T_out, N, 3) scaled
            velocity_mean,   # (3,)
            velocity_std,    # (3,)
            _wall_distance,
            _surface_frame,
        ) = dataset[idx]

        # ---------- linear extrapolation in scaled space ----------
        # velocity_in shape: (T_in, N, 3) where T_in=5
        step4 = velocity_in[3]   # shape (N, 3), index 3 = 4th input (0-based)
        step5 = velocity_in[4]   # shape (N, 3), index 4 = 5th input (last)
        delta = step5 - step4    # frame-to-frame velocity change
        pred_step6_scaled = step5 + delta  # (N, 3), linear extrapolation

        # velocity_out shape: (T_out, N, 3); first element = step 6
        true_step6_scaled = velocity_out[0]  # (N, 3)

        # RL2 (scaled)
        rl2 = relative_l2(pred_step6_scaled, true_step6_scaled)
        all_rl2.append(rl2)

        # Unscale for hint metric (match trainer convention)
        # unscale_velocity_batch expects (B, T, N, C) or we do it manually
        pred_unscaled = pred_step6_scaled * velocity_std + velocity_mean   # (N, 3)
        true_unscaled = true_step6_scaled * velocity_std + velocity_mean   # (N, 3)

        hint = (pred_unscaled - true_unscaled).norm(dim=-1).mean().item()
        all_hint.append(hint)

        last_pos = pos.clone()
        last_error_per_point = (pred_unscaled - true_unscaled).norm(dim=-1)  # (N,)
        last_airfoil_idcs = idcs_airfoil.clone()

        print(f"  [{idx+1:>3}/{n_samples}]  RL2_scaled={rl2:.4f}  "
              f"hint_unscaled={hint:.4f}")

    mean_rl2 = float(np.mean(all_rl2))
    mean_hint = float(np.mean(all_hint))
    std_rl2 = float(np.std(all_rl2))
    std_hint = float(np.std(all_hint))

    print()
    print("=" * 55)
    print("  Linear-extrapolation baseline (step 4→5 → predict 6)")
    print("=" * 55)
    print(f"  RL2  (scaled)       : {mean_rl2:.4f}  ±  {std_rl2:.4f}")
    print(f"  Hint (unscaled m/s) : {mean_hint:.4f}  ±  {std_hint:.4f}")
    print("=" * 55)

    # ------------------------------------------------------------------
    # 3-D scatter plot: error magnitude on last sample
    # ------------------------------------------------------------------
    if last_pos is None or last_error_per_point is None:
        return

    pos_np = last_pos.numpy()           # (N, 3)
    err_np = last_error_per_point.numpy()  # (N,)

    # Downsample for rendering speed if very large
    max_plot_pts = 30_000
    if pos_np.shape[0] > max_plot_pts:
        rng = np.random.default_rng(0)
        sel = rng.choice(pos_np.shape[0], size=max_plot_pts, replace=False)
        pos_np = pos_np[sel]
        err_np = err_np[sel]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        pos_np[:, 0],
        pos_np[:, 1],
        pos_np[:, 2],
        c=err_np,
        cmap="plasma",
        s=2,
        linewidths=0,
        alpha=0.7,
    )

    # Mark airfoil points
    if last_airfoil_idcs is not None and last_airfoil_idcs.numel() > 0:
        afoil_np = last_pos[last_airfoil_idcs.long()].numpy()
        ax.scatter(
            afoil_np[:, 0],
            afoil_np[:, 1],
            afoil_np[:, 2],
            c="cyan",
            s=6,
            linewidths=0,
            alpha=0.9,
            label="airfoil",
            zorder=5,
        )
        ax.legend(loc="upper right", fontsize=9)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label("|pred − true|  [m/s]", fontsize=10)

    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Y", fontsize=9)
    ax.set_zlabel("Z", fontsize=9)
    ax.set_title(
        f"Linear-extrapolation error  (step 6)\n"
        f"mean RL2={mean_rl2:.4f}   mean hint={mean_hint:.4f} m/s",
        fontsize=11,
    )

    out_dir = PROJECT_ROOT / "outputs" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "linear_extrap_error_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nScatter plot saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset_huggingface/warped-ifw",
        help="Path to the directory containing .npz simulation files.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of val samples to evaluate.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=100_000,
        help="Points per sample (use 100000 for full resolution).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    main(args)
