from __future__ import annotations

from .mlp import MLP
from .spatiotemporal_mno import SpatiotemporalMNO

CANONICAL_MODEL_REGISTRY = {
    "mlp": MLP,
    "spatiotemporal_mno": SpatiotemporalMNO,
}

MODEL_ALIASES = {
    "stmno": "spatiotemporal_mno",
    "st_mno": "spatiotemporal_mno",
    "spatiotemporal-mno": "spatiotemporal_mno",
}

MODEL_REGISTRY = {
    **CANONICAL_MODEL_REGISTRY,
    **{alias: CANONICAL_MODEL_REGISTRY[target] for alias, target in MODEL_ALIASES.items()},
}


def normalise_model_name(model_name: str) -> str:
    key = model_name.strip().lower()
    return MODEL_ALIASES.get(key, key)


def get_model_class(model_name: str):
    normalised = normalise_model_name(model_name)
    try:
        return CANONICAL_MODEL_REGISTRY[normalised]
    except KeyError as exc:
        available = ", ".join(sorted(CANONICAL_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model_name='{model_name}'. Available models: {available}."
        ) from exc


__all__ = [
    "CANONICAL_MODEL_REGISTRY",
    "MLP",
    "MODEL_ALIASES",
    "MODEL_REGISTRY",
    "SpatiotemporalMNO",
    "get_model_class",
    "normalise_model_name",
]
