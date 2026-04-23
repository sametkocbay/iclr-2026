"""Rolling Adams-Bashforth-2 extrapolation baseline on one .npz sample.

Strategy (auto-regressive, no oracle, 2nd-order Adams-Bashforth):
  u_{t+1} = u_t + 1.5*(u_t - u_{t-1}) - 0.5*(u_{t-1} - u_{t-2})

  pred_t6  = t5 + 1.5*(t5 - t4)  - 0.5*(t4 - t3)
  pred_t7  = pred_t6 + 1.5*(pred_t6 - t5)   - 0.5*(t5 - t4)
  pred_t8  = pred_t7 + 1.5*(pred_t7 - pred_t6) - 0.5*(pred_t6 - t5)
  ...

Computes RL2 (scaled) and hint metric (unscaled m/s) per step t6-t10, then:
  • Curve : RL2 and hint vs. horizon step
  • 3-D scatter per step: per-point velocity-error magnitude in m/s

Usage
-----
    python scripts/linear_extrap_single_sample.py \
        --file 1021_1-0.npz
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rl2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """||pred - target||_F / ||target||_F"""
    return ((pred - target).norm() / target.norm().clamp(min=1e-8)).item()


def hint(pred_unscaled: torch.Tensor, true_unscaled: torch.Tensor) -> float:
    """Mean per-point velocity-vector L2 error in physical units."""
    return (pred_unscaled - true_unscaled).norm(dim=-1).mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    npz_path = Path(args.file)
    if not npz_path.is_absolute():
        npz_path = PROJECT_ROOT / npz_path
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    with np.load(npz_path) as data:
        t_vals      = torch.from_numpy(data["t"]).float()          # (10,)
        pos         = torch.from_numpy(data["pos"]).float()         # (100000, 3)
        idcs_airfoil = torch.from_numpy(data["idcs_airfoil"]).long()
        velocity_in  = torch.from_numpy(data["velocity_in"]).float()  # (5, N, 3)
        velocity_out = torch.from_numpy(data["velocity_out"]).float() # (5, N, 3)

    # Per-channel standardisation from input frames (mirror dataset.py)
    eps = 1e-6
    v_mean = velocity_in.mean(dim=(0, 1), keepdim=True)   # (1, 1, 3)
    v_std  = velocity_in.std(dim=(0, 1), keepdim=True).clamp_min(eps)

    vin_s  = (velocity_in  - v_mean) / v_std   # (5, N, 3) scaled
    vout_s = (velocity_out - v_mean) / v_std   # (5, N, 3) scaled

    num_steps = velocity_out.shape[0]   # 5
    step_labels = [f"t{6 + i}" for i in range(num_steps)]

    # ---------- rolling Adams-Bashforth-2 extrapolation in scaled space ----------
    # Needs 3 frames: t-2, t-1, t
    preds_s: list[torch.Tensor] = []
    u_tm2 = vin_s[2]   # t3
    u_tm1 = vin_s[3]   # t4
    u_t   = vin_s[4]   # t5

    for _ in range(num_steps):
        pred = u_t + 1.5 * (u_t - u_tm1) - 0.5 * (u_tm1 - u_tm2)
        preds_s.append(pred)
        u_tm2 = u_tm1
        u_tm1 = u_t
        u_t   = pred

    # ---------- metrics per step ----------
    rl2_vals  : list[float] = []
    hint_vals : list[float] = []

    for i in range(num_steps):
        r = rl2(preds_s[i], vout_s[i])
        pred_unscaled = preds_s[i] * v_std.squeeze(0) + v_mean.squeeze(0)
        true_unscaled = velocity_out[i]
        h = hint(pred_unscaled, true_unscaled)
        rl2_vals.append(r)
        hint_vals.append(h)
        print(f"  {step_labels[i]}  RL2_scaled={r:.4f}   hint_unscaled={h:.4f} m/s")

    print()
    print("=" * 52)
    print(f"  {'Step':<8} {'RL2 (scaled)':>14}   {'Hint (m/s)':>12}")
    print("=" * 52)
    for i, lbl in enumerate(step_labels):
        print(f"  {lbl:<8} {rl2_vals[i]:>14.4f}   {hint_vals[i]:>12.4f}")
    print("=" * 52)

    out_dir = PROJECT_ROOT / "outputs" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Figure 1 — RL2 & hint curves
    # ------------------------------------------------------------------
    fig_curve, axes = plt.subplots(1, 2, figsize=(12, 4))

    steps_x = list(range(6, 11))

    axes[0].plot(steps_x, rl2_vals, marker="o", color="royalblue", linewidth=2)
    axes[0].set_xlabel("Predicted timestep", fontsize=11)
    axes[0].set_ylabel("RL2  (scaled space)", fontsize=11)
    axes[0].set_title("RL2 vs. horizon", fontsize=12)
    axes[0].set_xticks(steps_x)
    axes[0].grid(alpha=0.35)
    for x, y in zip(steps_x, rl2_vals):
        axes[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9)

    axes[1].plot(steps_x, hint_vals, marker="s", color="tomato", linewidth=2)
    axes[1].set_xlabel("Predicted timestep", fontsize=11)
    axes[1].set_ylabel("Mean ||pred − true||₂  [m/s]", fontsize=11)
    axes[1].set_title("Hint metric vs. horizon", fontsize=12)
    axes[1].set_xticks(steps_x)
    axes[1].grid(alpha=0.35)
    for x, y in zip(steps_x, hint_vals):
        axes[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9)

    fig_curve.suptitle(
        f"Adams-Bashforth-2 extrapolation baseline — {npz_path.name}", fontsize=13
    )
    fig_curve.tight_layout()
    curve_path = out_dir / "linear_extrap_curve.png"
    fig_curve.savefig(curve_path, dpi=150, bbox_inches="tight")
    print(f"Curve saved → {curve_path}")
    plt.close(fig_curve)

    # ------------------------------------------------------------------
    # Figure 2 — 3-D scatter per step
    # ------------------------------------------------------------------
    fig_scatter = plt.figure(figsize=(22, 10))
    fig_scatter.suptitle(
        f"Per-point velocity error  |pred − true|  [m/s]\n{npz_path.name}  —  Adams-Bashforth-2 extrapolation",
        fontsize=13, y=1.01
    )

    max_plot_pts = 25_000
    pos_np = pos.numpy()

    # Compute common vmax across all steps for a consistent colour scale
    all_errors = []
    for i in range(num_steps):
        pred_unscaled = (preds_s[i] * v_std.squeeze(0) + v_mean.squeeze(0)).numpy()
        true_unscaled = velocity_out[i].numpy()
        all_errors.append(np.linalg.norm(pred_unscaled - true_unscaled, axis=-1))
    vmax = float(np.percentile(np.concatenate(all_errors), 99))

    # Subsample indices (same for every step for fair comparison)
    if pos_np.shape[0] > max_plot_pts:
        rng = np.random.default_rng(0)
        sel = rng.choice(pos_np.shape[0], size=max_plot_pts, replace=False)
    else:
        sel = np.arange(pos_np.shape[0])

    afoil_np = pos[idcs_airfoil].numpy() if idcs_airfoil.numel() > 0 else None

    for i in range(num_steps):
        ax = fig_scatter.add_subplot(1, 5, i + 1, projection="3d")

        err_np = all_errors[i]
        sc = ax.scatter(
            pos_np[sel, 0], pos_np[sel, 1], pos_np[sel, 2],
            c=err_np[sel],
            cmap="plasma",
            vmin=0.0, vmax=vmax,
            s=1.5, linewidths=0, alpha=0.6,
        )

        if afoil_np is not None:
            ax.scatter(
                afoil_np[:, 0], afoil_np[:, 1], afoil_np[:, 2],
                c="cyan", s=3, linewidths=0, alpha=0.8, zorder=5,
            )

        ax.set_title(
            f"{step_labels[i]}\nRL2={rl2_vals[i]:.3f}  hint={hint_vals[i]:.3f}",
            fontsize=9
        )
        ax.set_xlabel("X", fontsize=7, labelpad=2)
        ax.set_ylabel("Y", fontsize=7, labelpad=2)
        ax.set_zlabel("Z", fontsize=7, labelpad=2)
        ax.tick_params(labelsize=6)

    # Shared colorbar
    cbar_ax = fig_scatter.add_axes([0.92, 0.15, 0.015, 0.65])
    sm = plt.cm.ScalarMappable(cmap="plasma",
                                norm=plt.Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    fig_scatter.colorbar(sm, cax=cbar_ax, label="|pred − true|  [m/s]")

    scatter_path = out_dir / "linear_extrap_scatter_all_steps.png"
    fig_scatter.savefig(scatter_path, dpi=150, bbox_inches="tight")
    print(f"Scatter saved → {scatter_path}")
    plt.close(fig_scatter)


# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=str,
        default="1021_1-0.npz",
        help="Path to .npz file (absolute or relative to project root).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
