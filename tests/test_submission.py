from __future__ import annotations

from pathlib import Path

import torch

from models import MLP


def test_submission_model_loads_default_checkpoint_and_runs_inference_once() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    expected_checkpoint = (
        repo_root
        / "baseline_mno_seed118_base_2_32k_budget_crop_5580430"
        / "checkpoints"
        / "best.pt"
    )
    assert expected_checkpoint.exists(), f"Missing expected checkpoint: {expected_checkpoint}"

    resolved_checkpoint = MLP._resolve_weights_path(None)
    assert resolved_checkpoint == expected_checkpoint

    model = MLP()
    model.eval()
    model.use_torch_cluster_knn = False

    batch_size = 1
    num_pos = 32

    t = torch.linspace(
        0.0,
        0.9,
        steps=model.num_t_in + model.num_t_out,
        dtype=torch.float32,
    ).unsqueeze(0)
    pos = torch.randn(batch_size, num_pos, 3, dtype=torch.float32)
    velocity_in = torch.randn(
        batch_size,
        model.num_t_in,
        num_pos,
        model.output_channels,
        dtype=torch.float32,
    )
    idcs_airfoil = [torch.arange(0, min(8, num_pos), dtype=torch.long)]

    with torch.inference_mode():
        velocity_out = model(
            t=t,
            pos=pos,
            idcs_airfoil=idcs_airfoil,
            velocity_in=velocity_in,
        )

    assert velocity_out.shape == (
        batch_size,
        model.num_t_out,
        num_pos,
        model.output_channels,
    )
    assert torch.isfinite(velocity_out).all()
