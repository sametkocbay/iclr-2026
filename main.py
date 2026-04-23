from __future__ import annotations

import argparse

from models import (
    MLP as Model,
    SpatiotemporalMNO,
    get_model_class,
)
from src.data import (
    WarpedIFWDataset,
    build_loader as _build_loader,
    compute_velocity_standardization as _compute_velocity_standardization,
    resolve_overfit_file,
    scale_velocity as _scale_velocity,
    split_train_val_test,
    unscale_velocity_batch as _unscale_velocity_batch,
)
from src.training import (
    NUM_POS,
    NUM_T_IN,
    NUM_T_OUT,
    evaluate,
    hint_metric,
    parse_args as _parse_args,
    run_full_test_inference,
    set_seed as _set_seed,
)
from src.training import trainer as _trainer


def set_seed(seed: int) -> None:
    _set_seed(seed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _parse_args(argv)


def train(args: argparse.Namespace) -> None:
    # Keep notebook monkey-patching compatibility: if callers rebind main.Model,
    # apply the same class to the trainer module before execution.
    _trainer.Model = Model
    _trainer.train(args)


__all__ = [
    "Model",
    "SpatiotemporalMNO",
    "NUM_POS",
    "NUM_T_IN",
    "NUM_T_OUT",
    "WarpedIFWDataset",
    "_build_loader",
    "_compute_velocity_standardization",
    "_scale_velocity",
    "_unscale_velocity_batch",
    "evaluate",
    "get_model_class",
    "hint_metric",
    "parse_args",
    "resolve_overfit_file",
    "run_full_test_inference",
    "set_seed",
    "split_train_val_test",
    "train",
]


if __name__ == "__main__":
    train(parse_args())
