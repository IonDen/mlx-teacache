"""FLUX.1 dev configuration. mflux-free."""
from __future__ import annotations

from typing import Any

from mlx_teacache.coefficients import (
    _UPSTREAM_FLUX_COEFFS as _LEGACY_COEFFS,  # Worker: COPY the tuple from src/mlx_teacache/coefficients.py::_UPSTREAM_FLUX_COEFFS
)

COEFFICIENTS: tuple[float, float, float, float, float] = _LEGACY_COEFFS

DEFAULT_THRESH: float = 0.20

RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 25, "guidance": 3.5},
}

LICENSE: str = "FLUX.1-dev Non-Commercial License"

META: dict[str, Any] = {
    "variant_id": "flux1-dev",
    "display_name": "FLUX.1 dev",
    "hf_model_id": "black-forest-labs/FLUX.1-dev",
    "non_distilled": True,
    "memory_cap_hint_gb": None,
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
}
