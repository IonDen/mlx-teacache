"""FLUX.2 Klein base 4B configuration. mflux-free."""
from __future__ import annotations

from typing import Any

from mlx_teacache.coefficients import (
    _FLUX2_KLEIN_BASE_4B_COEFFS as _LEGACY_COEFFS,
)

COEFFICIENTS: tuple[float, float, float, float, float] = _LEGACY_COEFFS

DEFAULT_THRESH: float = 0.17

RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 50, "guidance": 4.0},
    "low_step": {"num_inference_steps": 25, "guidance": 1.0},
}

LICENSE: str = "Apache-2.0"

META: dict[str, Any] = {
    "variant_id": "flux2-klein-base-4b",
    "display_name": "FLUX.2 Klein base 4B",
    "hf_model_id": "black-forest-labs/FLUX.2-klein-base-4B",
    "non_distilled": True,
    "memory_cap_hint_gb": None,
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B",
}
