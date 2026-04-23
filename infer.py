"""
Submission inference check.

Initialises MLP exactly as required by the competition:
    model = MLP()
    velocity_out = model(t, pos, idcs_airfoil, velocity_in)

Then evaluates RL2 against the ground-truth on 1021_1-0.npz.
"""

import numpy as np
import torch

from models.mlp import MLP


def rl2(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return ((pred - gt).norm() / (gt.norm() + 1e-8)).item()


# ── Load sample ──────────────────────────────────────────────────────────────
d = np.load("1021_1-0.npz", allow_pickle=True)

t            = torch.from_numpy(d["t"].astype(np.float32)).unsqueeze(0)          # (1, 10)
pos          = torch.from_numpy(d["pos"].astype(np.float32)).unsqueeze(0)        # (1, 100k, 3)
velocity_in  = torch.from_numpy(d["velocity_in"].astype(np.float32)).unsqueeze(0) # (1, 5, 100k, 3)
velocity_out_gt = torch.from_numpy(d["velocity_out"].astype(np.float32))         # (5, 100k, 3)
idcs_airfoil = [torch.from_numpy(d["idcs_airfoil"].astype(np.int64))]

# ── Instantiate exactly as the competition requires ───────────────────────────
model = MLP()
model.eval()
print(f"Loaded weights from: {model.loaded_weights_path}\n")

# ── Run inference with the bare 4-argument signature ─────────────────────────
with torch.no_grad():
    velocity_out = model(t, pos, idcs_airfoil, velocity_in)  # (1, 5, 100k, 3)

velocity_out = velocity_out.squeeze(0)  # (5, 100k, 3)

# ── Metrics ───────────────────────────────────────────────────────────────────
print("=== RL2 per timestep ===")
for i in range(5):
    print(f"  t+{i + 1}: {rl2(velocity_out[i], velocity_out_gt[i]):.4f}")

overall = rl2(velocity_out, velocity_out_gt)
persistence = rl2(
    velocity_in.squeeze(0)[-1].unsqueeze(0).expand(5, -1, -1),
    velocity_out_gt,
)
print(f"\nOverall RL2:          {overall:.4f}")
print(f"Persistence RL2:      {persistence:.4f}")
print(f"Improvement:          {(persistence - overall) / persistence * 100:.1f}%")
