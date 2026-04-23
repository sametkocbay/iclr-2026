from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src.training.trainer import parse_args as parse_train_args
from src.utils.config import apply_parser_defaults_from_config, read_config_defaults_from_cli


def test_apply_parser_defaults_from_config_non_strict_filters_unknown_keys() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--foo", type=int, default=1)
    parser.add_argument("--bar", type=str, default="x")

    apply_parser_defaults_from_config(
        parser,
        {"foo": 7, "baz": 99},
        strict_unknown_keys=False,
    )

    args = parser.parse_args([])
    assert args.foo == 7
    assert args.bar == "x"


def test_apply_parser_defaults_from_config_strict_raises_on_unknown_keys() -> None:
    parser = argparse.ArgumentParser(prog="demo")
    parser.add_argument("--foo", type=int, default=1)

    with pytest.raises(SystemExit):
        apply_parser_defaults_from_config(
            parser,
            {"foo": 3, "unknown_key": 42},
            strict_unknown_keys=True,
        )


def test_read_config_defaults_from_cli_without_config_returns_empty() -> None:
    defaults = read_config_defaults_from_cli(["--foo", "bar"])
    assert defaults == {}


def test_trainer_parse_args_rejects_unknown_yaml_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "train_config_invalid.yaml"
    config_path.write_text("continuity_unknown_option: 1\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        parse_train_args(["--config", str(config_path)])
