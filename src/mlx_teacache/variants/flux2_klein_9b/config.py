"""FLUX.2 Klein 9B (distilled) configuration. mflux-free."""
from __future__ import annotations

from typing import Any

# Cross-import from coefficients.py for v0.6.0 transition.
from mlx_teacache.coefficients import _FLUX2_KLEIN_9B_COEFFS as _LEGACY_COEFFS

COEFFICIENTS: tuple[float, float, float, float, float] = _LEGACY_COEFFS

DEFAULT_THRESH: float | None = None  # distilled gate doesn't engage; use package fallback 0.20

RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 8, "guidance": 1.0},
}

LICENSE: str = "FLUX Non-Commercial"

META: dict[str, Any] = {
    "variant_id": "flux2-klein-9b",
    "display_name": "FLUX.2 Klein 9B",
    "hf_model_id": "black-forest-labs/FLUX.2-klein-9B",
    "non_distilled": False,
    "memory_cap_hint_gb": None,
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B",
}
