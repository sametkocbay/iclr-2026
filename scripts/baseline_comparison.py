"""Comprehensive baseline comparison for CFD velocity prediction.

Strategies compared (all auto-regressive, no oracle):
  1. persistence   : u_{t+1} = u_t
  2. linear        : u_{t+1} = u_t + (u_t - u_{t-1})
  3. damped        : u_{t+1} = u_t + 0.5*(u_t - u_{t-1})
  4. ema           : u_{t+1} = 0.7*u_t + 0.3*u_{t-1}
  5. AB2           : u_{t+1} = u_t + 1.5*(u_t-u_{t-1}) - 0.5*(u_{t-1}-u_{t-2})
  6. diffuse+lin   : linear + alpha*Laplacian(u_t)
  7. convective    : u_t - dt*(u·∇u)   (Euler advection step)
  8. conv+linear   : linear + convective correction

Usage
-----
    python scripts/baseline_comparison.py --file 1021_1-0.npz
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# kNN graph
# ─────────────────────────────────────────────────────────────────────────────

def build_knn(pos: torch.Tensor, k: int = 16) -> torch.Tensor:
    """Return (N, k) long tensor of k nearest neighbor indices (self excluded)."""
    pos_np = pos.numpy()
    tree = cKDTree(pos_np)
    _, idx = tree.query(pos_np, k=k + 1)       # (N, k+1)
    return torch.from_numpy(idx[:, 1:]).long()  # (N, k)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial operators  (work in *scaled* space unless noted)
# ─────────────────────────────────────────────────────────────────────────────

def laplacian_smooth(u_s: torch.Tensor, knn_idx: torch.Tensor, alpha: float) -> torch.Tensor:
    """alpha * (mean_neighbors - u_s)  — discrete Laplacian in scaled space."""
    return alpha * (u_s[knn_idx].mean(1) - u_s)   # (N, 3)


def advective_correction(
    u_s: torch.Tensor,
    pos: torch.Tensor,
    knn_idx: torch.Tensor,
    dt: float,
    v_mean: torch.Tensor,
    v_std: torch.Tensor,
) -> torch.Tensor:
    """Returns -dt*(u·∇u) in *scaled* space.

    Computed in physical units then re-scaled so that spatial gradients are
    physically meaningful regardless of normalisation.

    Discretisation (SPH-like finite difference):
        (u·∇u)_i ≈ (1/k) Σ_j  (u_i · r_ij) * (u_j - u_i) / |r_ij|²
    where r_ij = pos_j - pos_i.
    """
    u = u_s * v_std + v_mean            # physical velocity  (N, 3)
    neigh_pos = pos[knn_idx]            # (N, k, 3)
    neigh_u   = u[knn_idx]             # (N, k, 3)
    r   = neigh_pos - pos.unsqueeze(1)             # (N, k, 3)
    d2  = r.square().sum(-1, keepdim=True).clamp(min=1e-12)  # (N, k, 1)
    # (u_i · r_ij): inner product giving "how much u points toward neighbor"
    u_dot_r = (u.unsqueeze(1) * r).sum(-1, keepdim=True)    # (N, k, 1)
    du      = neigh_u - u.unsqueeze(1)                       # (N, k, 3)
    conv_phys = (u_dot_r * du / d2).mean(1)  # (N, 3)  [m/s²]
    # Clamp to ±10× the per-channel velocity std to prevent blow-up in
    # low-density kNN regions (e.g. near domain boundaries).
    max_corr = 10.0 * v_std                  # (3,) broadcast-safe
    correction_phys = (-dt * conv_phys).clamp(-max_corr, max_corr)
    return correction_phys / v_std           # back to scaled units  (N, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline functions
# Each accepts (history, *, pos, knn_idx, dt, v_mean, v_std)
# history[-1] = u_t, history[-2] = u_{t-1}, history[-3] = u_{t-2}
# Returns next frame in *scaled* space, shape (N, 3)
# ─────────────────────────────────────────────────────────────────────────────

def _persistence(h, **_):
    return h[-1].clone()

def _linear(h, **_):
    return h[-1] + (h[-1] - h[-2])

def _damped(h, **_):
    return h[-1] + 0.5 * (h[-1] - h[-2])

def _ema(h, **_):
    return 0.7 * h[-1] + 0.3 * h[-2]

def _ab2(h, **_):
    return h[-1] + 1.5 * (h[-1] - h[-2]) - 0.5 * (h[-2] - h[-3])

def _diffuse(h, *, knn_idx, **_):
    linear = h[-1] + (h[-1] - h[-2])
    return linear + laplacian_smooth(h[-1], knn_idx, alpha=0.15)

def _convective(h, *, pos, knn_idx, dt, v_mean, v_std, **_):
    return h[-1] + advective_correction(h[-1], pos, knn_idx, dt, v_mean, v_std)

def _conv_linear(h, *, pos, knn_idx, dt, v_mean, v_std, **_):
    linear = h[-1] + (h[-1] - h[-2])
    return linear + advective_correction(h[-1], pos, knn_idx, dt, v_mean, v_std)


BASELINES: OrderedDict = OrderedDict([
    ("persistence",  _persistence),
    ("linear",       _linear),
    ("damped×0.5",   _damped),
    ("ema 0.7/0.3",  _ema),
    ("AB2",          _ab2),
    ("diffuse+lin",  _diffuse),
    ("convective",   _convective),
    ("conv+linear",  _conv_linear),
])


# ─────────────────────────────────────────────────────────────────────────────
# Rollout + metrics
# ─────────────────────────────────────────────────────────────────────────────

def rollout(fn, vin_s: torch.Tensor, n_steps: int, **kwargs) -> list[torch.Tensor]:
    history = list(vin_s[-3:].unbind(0))   # [(N,3), (N,3), (N,3)]
    preds: list[torch.Tensor] = []
    for _ in range(n_steps):
        pred = fn(history, **kwargs)
        preds.append(pred)
        history = history[1:] + [pred]
    return preds


def rl2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return ((pred - target).norm() / target.norm().clamp(min=1e-8)).item()


def hint(pred_s: torch.Tensor, true_phys: torch.Tensor,
         v_mean: torch.Tensor, v_std: torch.Tensor) -> float:
    pred_phys = pred_s * v_std + v_mean
    return (pred_phys - true_phys).norm(dim=-1).mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

COLORS = plt.cm.get_cmap("tab10", len(BASELINES))
MARKERS = list("osDv^P*X")


def plot_curves(steps_x, results, out_dir, sample_name):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for i, (name, m) in enumerate(results.items()):
        c, mk = COLORS(i), MARKERS[i % len(MARKERS)]
        axes[0].plot(steps_x, m["rl2"],  f"-{mk}", label=name, color=c, lw=2, ms=7)
        axes[1].plot(steps_x, m["hint"], f"-{mk}", label=name, color=c, lw=2, ms=7)
    for ax, ylabel, title in zip(
        axes,
        ["RL2  (scaled space)", "Mean ||pred−true||₂  [m/s]"],
        ["RL2 vs. horizon", "Hint metric vs. horizon"],
    ):
        ax.set_xlabel("Predicted timestep", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(steps_x)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.35)
    fig.suptitle(f"Baseline comparison — {sample_name}", fontsize=13)
    fig.tight_layout()
    p = out_dir / "baseline_comparison_curves.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Curves saved → {p}")
    plt.close(fig)


def plot_bar(results, out_dir):
    names = list(results.keys())
    mean_rl2  = [float(np.mean(m["rl2"]))  for m in results.values()]
    mean_hint = [float(np.mean(m["hint"])) for m in results.values()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = [COLORS(i) for i in range(len(names))]
    x = np.arange(len(names))

    axes[0].bar(x, mean_rl2,  color=colors, edgecolor="k", linewidth=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Mean RL2 (t6-t10)", fontsize=11)
    axes[0].set_title("Mean RL2 — lower is better", fontsize=12)
    axes[0].grid(axis="y", alpha=0.35)
    for xi, v in zip(x, mean_rl2):
        axes[0].text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, mean_hint, color=colors, edgecolor="k", linewidth=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Mean hint [m/s] (t6-t10)", fontsize=11)
    axes[1].set_title("Mean hint — lower is better", fontsize=12)
    axes[1].grid(axis="y", alpha=0.35)
    for xi, v in zip(x, mean_hint):
        axes[1].text(xi, v + 0.05, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    p = out_dir / "baseline_comparison_bars.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Bar chart saved → {p}")
    plt.close(fig)


def plot_scatter_best(pos, best_name, preds_s, vout_s, velocity_out,
                      v_mean, v_std, idcs_airfoil, step_labels, out_dir):
    pos_np = pos.numpy()
    num_steps = len(preds_s)
    all_err = []
    for i in range(num_steps):
        p = (preds_s[i] * v_std + v_mean).numpy()
        all_err.append(np.linalg.norm(p - velocity_out[i].numpy(), axis=-1))
    vmax = float(np.percentile(np.concatenate(all_err), 99))

    max_pts = 25_000
    rng = np.random.default_rng(0)
    sel = rng.choice(pos_np.shape[0], size=min(max_pts, pos_np.shape[0]), replace=False)
    afoil_np = pos[idcs_airfoil].numpy() if idcs_airfoil.numel() > 0 else None

    fig = plt.figure(figsize=(22, 8))
    fig.suptitle(
        f"Per-point |pred − true| [m/s]  —  best baseline: {best_name}",
        fontsize=13, y=1.01,
    )
    for i in range(num_steps):
        ax = fig.add_subplot(1, num_steps, i + 1, projection="3d")
        ax.scatter(pos_np[sel, 0], pos_np[sel, 1], pos_np[sel, 2],
                   c=all_err[i][sel], cmap="plasma", vmin=0.0, vmax=vmax,
                   s=1.5, linewidths=0, alpha=0.6)
        if afoil_np is not None:
            ax.scatter(afoil_np[:, 0], afoil_np[:, 1], afoil_np[:, 2],
                       c="cyan", s=3, linewidths=0, alpha=0.9, zorder=5)
        r = rl2(preds_s[i], vout_s[i])
        h = hint(preds_s[i], velocity_out[i], v_mean, v_std)
        ax.set_title(f"{step_labels[i]}\nRL2={r:.3f}  hint={h:.3f}", fontsize=9)
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7); ax.tick_params(labelsize=6)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.65])
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="|pred − true| [m/s]")
    safe = best_name.replace(" ", "_").replace("/", "_").replace("×", "x")
    p = out_dir / f"baseline_best_{safe}_scatter.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Scatter saved → {p}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

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

    dt_phys = float(t_vals[1] - t_vals[0])
    eps = 1e-6

    # Per-channel standardisation from input frames
    v_mean = velocity_in.reshape(-1, 3).mean(0)                                  # (3,)
    v_std  = velocity_in.reshape(-1, 3).std(0, unbiased=False).clamp_min(eps)    # (3,)

    vin_s  = (velocity_in  - v_mean) / v_std   # (5, N, 3)
    vout_s = (velocity_out - v_mean) / v_std   # (5, N, 3)

    num_steps   = velocity_out.shape[0]
    step_labels = [f"t{6 + i}" for i in range(num_steps)]
    steps_x     = list(range(6, 11))

    print(f"Sample  : {npz_path.name}")
    print(f"N pts   : {pos.shape[0]}  |  dt = {dt_phys:.4e} s")
    print(f"v_mean  : {v_mean.tolist()}  m/s")
    print(f"v_std   : {v_std.tolist()}  m/s")
    print(f"\nBuilding kNN graph (k=16) …")
    knn_idx = build_knn(pos, k=16)
    print("Done.\n")

    kwargs = dict(pos=pos, knn_idx=knn_idx, dt=dt_phys, v_mean=v_mean, v_std=v_std)

    # ── run all baselines ────────────────────────────────────────────────────
    results: dict[str, dict] = {}
    for name, fn in BASELINES.items():
        preds = rollout(fn, vin_s, num_steps, **kwargs)
        results[name] = {
            "rl2":   [rl2(preds[i], vout_s[i]) for i in range(num_steps)],
            "hint":  [hint(preds[i], velocity_out[i], v_mean, v_std) for i in range(num_steps)],
            "preds": preds,
        }

    # ── print tables ─────────────────────────────────────────────────────────
    cw = 10
    header = f"{'Baseline':<18}" + "".join(f"{l:>{cw}}" for l in step_labels) + f"{'mean':>{cw}}"
    sep = "─" * len(header)

    for metric_key, label in [("rl2", "RL2 (scaled) — lower is better"),
                               ("hint", "Hint (m/s)   — lower is better")]:
        print(sep)
        print(f"  {label}")
        print(sep)
        print(header)
        print(sep)
        for name, m in results.items():
            vals = m[metric_key]
            best_mark = " ◀" if name == min(results, key=lambda n: float(np.mean(results[n][metric_key]))) else ""
            row = (f"{name:<18}" +
                   "".join(f"{v:>{cw}.4f}" for v in vals) +
                   f"{np.mean(vals):>{cw}.4f}{best_mark}")
            print(row)
        print()

    best_name = min(results, key=lambda n: float(np.mean(results[n]["rl2"])))
    print(f"→ Best baseline by mean RL2: {best_name}  "
          f"({np.mean(results[best_name]['rl2']):.4f})")

    # ── plots ────────────────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / "outputs" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_curves(steps_x, results, out_dir, npz_path.name)
    plot_bar(results, out_dir)
    plot_scatter_best(
        pos, best_name, results[best_name]["preds"],
        vout_s, velocity_out, v_mean, v_std,
        idcs_airfoil, step_labels, out_dir,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file", type=str, default="1021_1-0.npz",
                   help="Path to .npz (relative to project root or absolute).")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
