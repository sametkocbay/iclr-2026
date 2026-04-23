"""Persistence baseline: predict every future step as u_t5.

This is the simplest possible baseline — no extrapolation, no model.
It answers: "how much does the flow actually change from t5 to t6/7/8/9/10?"

The RL2 numbers here are the **floor of difficulty** for the task.
Any model that cannot beat these numbers is learning nothing.

Usage
-----
    python scripts/persistence_baseline.py --file 1021_1-0.npz
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


def rl2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return ((pred - target).norm() / target.norm().clamp(min=1e-8)).item()


def hint(pred_phys: torch.Tensor, true_phys: torch.Tensor) -> float:
    return (pred_phys - true_phys).norm(dim=-1).mean().item()


def main(args: argparse.Namespace) -> None:
    npz_path = Path(args.file)
    if not npz_path.is_absolute():
        npz_path = PROJECT_ROOT / npz_path
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    with np.load(npz_path) as data:
        t_vals       = torch.from_numpy(data["t"]).float()
        pos          = torch.from_numpy(data["pos"]).float()
        idcs_airfoil = torch.from_numpy(data["idcs_airfoil"]).long()
        velocity_in  = torch.from_numpy(data["velocity_in"]).float()   # (5, N, 3)
        velocity_out = torch.from_numpy(data["velocity_out"]).float()  # (5, N, 3)

    eps = 1e-6
    v_mean = velocity_in.reshape(-1, 3).mean(0)                              # (3,)
    v_std  = velocity_in.reshape(-1, 3).std(0, unbiased=False).clamp_min(eps)

    vin_s  = (velocity_in  - v_mean) / v_std
    vout_s = (velocity_out - v_mean) / v_std

    # persistence prediction = last input frame for every horizon
    t5_s    = vin_s[-1]          # (N, 3) scaled
    t5_phys = velocity_in[-1]    # (N, 3) physical

    num_steps   = velocity_out.shape[0]   # 5
    step_labels = [f"t{6 + i}" for i in range(num_steps)]
    steps_x     = list(range(6, 11))

    rl2_vals, hint_vals = [], []
    for i in range(num_steps):
        r = rl2(t5_s, vout_s[i])
        h = hint(t5_phys, velocity_out[i])
        rl2_vals.append(r)
        hint_vals.append(h)

    # ── print table ──────────────────────────────────────────────────────────
    print(f"\nSample : {npz_path.name}")
    print(f"Prediction: copy t5 → every future step (no extrapolation)\n")
    sep = "=" * 52
    print(sep)
    print(f"  {'Step':<8} {'RL2 (scaled)':>14}   {'Hint (m/s)':>12}")
    print(sep)
    for i, lbl in enumerate(step_labels):
        print(f"  {lbl:<8} {rl2_vals[i]:>14.4f}   {hint_vals[i]:>12.4f}")
    print(sep)
    print(f"  {'mean':<8} {np.mean(rl2_vals):>14.4f}   {np.mean(hint_vals):>12.4f}")
    print(sep)
    print()
    print("These numbers define the task difficulty floor.")
    print("A model must score below these to be doing useful work.\n")

    # ── Figure 1: curves ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, vals, ylabel, title, color in zip(
        axes,
        [rl2_vals, hint_vals],
        ["RL2  (scaled space)", "Mean ||u_t5 − u_true||₂  [m/s]"],
        ["RL2 of persistence (t5) vs ground truth", "Hint metric of persistence vs ground truth"],
        ["royalblue", "tomato"],
    ):
        ax.plot(steps_x, vals, marker="o", color=color, lw=2.5, ms=8)
        ax.fill_between(steps_x, vals, alpha=0.12, color=color)
        ax.set_xlabel("Ground-truth timestep", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(steps_x)
        ax.grid(alpha=0.35)
        for x, y in zip(steps_x, vals):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9)

    fig.suptitle(
        f"Task difficulty — persistence baseline (copy t5)\n{npz_path.name}",
        fontsize=13,
    )
    fig.tight_layout()
    out_dir = PROJECT_ROOT / "outputs" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / "persistence_baseline_curve.png"
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    print(f"Curve saved → {curve_path}")
    plt.close(fig)

    # ── Figure 2: 3-D scatter per step (|t5 - true| in m/s) ─────────────────
    pos_np = pos.numpy()
    all_err = [
        np.linalg.norm((t5_phys - velocity_out[i]).numpy(), axis=-1)
        for i in range(num_steps)
    ]
    vmax = float(np.percentile(np.concatenate(all_err), 99))

    max_pts = 25_000
    rng = np.random.default_rng(0)
    sel = rng.choice(pos_np.shape[0], size=min(max_pts, pos_np.shape[0]), replace=False)
    afoil_np = pos[idcs_airfoil].numpy() if idcs_airfoil.numel() > 0 else None

    fig2 = plt.figure(figsize=(22, 8))
    fig2.suptitle(
        f"|u_t5 − u_true|  [m/s]  per step  —  {npz_path.name}",
        fontsize=13, y=1.01,
    )
    for i in range(num_steps):
        ax = fig2.add_subplot(1, num_steps, i + 1, projection="3d")
        ax.scatter(
            pos_np[sel, 0], pos_np[sel, 1], pos_np[sel, 2],
            c=all_err[i][sel], cmap="inferno",
            vmin=0.0, vmax=vmax, s=1.5, linewidths=0, alpha=0.65,
        )
        if afoil_np is not None:
            ax.scatter(
                afoil_np[:, 0], afoil_np[:, 1], afoil_np[:, 2],
                c="cyan", s=3, linewidths=0, alpha=0.9, zorder=5,
            )
        ax.set_title(
            f"{step_labels[i]}\nRL2={rl2_vals[i]:.3f}  hint={hint_vals[i]:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7); ax.tick_params(labelsize=6)

    cbar_ax = fig2.add_axes([0.92, 0.15, 0.015, 0.65])
    sm = plt.cm.ScalarMappable(
        cmap="inferno", norm=plt.Normalize(vmin=0.0, vmax=vmax)
    )
    sm.set_array([])
    fig2.colorbar(sm, cax=cbar_ax, label="|u_t5 − u_true|  [m/s]")

    scatter_path = out_dir / "persistence_baseline_scatter.png"
    fig2.savefig(scatter_path, dpi=150, bbox_inches="tight")
    print(f"Scatter saved → {scatter_path}")
    plt.close(fig2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", type=str, default="1021_1-0.npz",
                   help="Path to .npz (relative to project root or absolute).")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
