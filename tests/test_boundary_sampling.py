from __future__ import annotations

import torch

from src.data.dataset import build_boundary_biased_indices


def test_boundary_biased_indices_keep_surface_anchors_and_offwall_shell() -> None:
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [0.0, 0.05],
            [1.0, 0.05],
            [2.0, 0.05],
            [3.0, 0.05],
            [0.0, 0.5],
            [1.0, 0.5],
            [2.0, 0.5],
            [3.0, 0.5],
        ],
        dtype=torch.float32,
    )
    idcs_airfoil = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    selected = build_boundary_biased_indices(
        pos,
        idcs_airfoil,
        target_num_pos=8,
        boundary_point_fraction=0.75,
        seed=0,
    )

    selected_set = set(selected.tolist())
    airfoil_set = set(idcs_airfoil.tolist())
    shell_set = {4, 5, 6, 7}

    assert len(selected) == 8
    assert selected.unique().numel() == 8
    assert selected_set & airfoil_set
    assert shell_set.issubset(selected_set)


def test_boundary_biased_indices_interleave_shell_across_surface_regions() -> None:
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 0.01],
            [0.0, 0.02],
            [0.0, 0.03],
            [10.0, 0.20],
            [5.0, 5.0],
            [6.0, 6.0],
        ],
        dtype=torch.float32,
    )
    idcs_airfoil = torch.tensor([0, 1], dtype=torch.long)

    selected = build_boundary_biased_indices(
        pos,
        idcs_airfoil,
        target_num_pos=6,
        boundary_point_fraction=0.75,
        seed=0,
    )

    selected_set = set(selected.tolist())
    assert 2 in selected_set
    assert 5 in selected_set
