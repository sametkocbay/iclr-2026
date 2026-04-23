from __future__ import annotations

import torch
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

from models.mlp.training import build_training_components


def test_build_training_components_fixed_scheduler_keeps_lr_constant() -> None:
    model = torch.nn.Linear(3, 2)
    training = build_training_components(
        model,
        steps_per_epoch=4,
        max_lr=1e-3,
        epochs=3,
        lr_scheduler_name="fixed",
    )

    assert isinstance(training.scheduler, LambdaLR)
    initial_lr = training.optimizer.param_groups[0]["lr"]
    training.optimizer.zero_grad(set_to_none=True)
    model(torch.zeros(1, 3)).sum().backward()
    training.optimizer.step()
    training.scheduler.step()
    assert training.optimizer.param_groups[0]["lr"] == initial_lr


def test_build_training_components_one_cycle_scheduler_is_available() -> None:
    model = torch.nn.Linear(3, 2)
    training = build_training_components(
        model,
        steps_per_epoch=4,
        max_lr=1e-3,
        epochs=3,
        lr_scheduler_name="one-cycle",
    )

    assert isinstance(training.scheduler, OneCycleLR)


def test_build_training_components_rejects_unknown_scheduler_name() -> None:
    model = torch.nn.Linear(3, 2)

    try:
        build_training_components(
            model,
            steps_per_epoch=4,
            lr_scheduler_name="cosine",
        )
    except ValueError as exc:
        assert "Unknown lr_scheduler_name" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown lr_scheduler_name.")