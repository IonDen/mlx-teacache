"""FLUX.2 Klein base 9B configuration. mflux-free.

Coefficients are REUSED from flux2_klein_base_4b — same architecture
family, same calibration recipe. v0.5.0 validated this reuse with
SSIM 0.986 at 50 steps + g=4.0; see scripts/validate_klein_base_9b.py
and _artifacts/validation_klein_base_9b.json for the evidence."""
from __future__ import annotations

from typing import Any

# Cross-import to preserve object identity with base-4b's COEFFICIENTS.
# The test `test_klein_base_9b_reuses_base_4b_coefficients` asserts
# `BASE_9B is BASE_4B`.
from mlx_teacache.variants.flux2_klein_base_4b.config import (
    COEFFICIENTS as _BASE_4B_COEFFS,
)

COEFFICIENTS: tuple[float, float, float, float, float] = _BASE_4B_COEFFS

DEFAULT_THRESH: float = 0.17  # reused from base-4b v0.4.0 sweep

RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 50, "guidance": 4.0},
    "low_step": {"num_inference_steps": 25, "guidance": 1.0},
}

LICENSE: str = "FLUX Non-Commercial"

META: dict[str, Any] = {
    "variant_id": "flux2-klein-base-9b",
    "display_name": "FLUX.2 Klein base 9B",
    "hf_model_id": "black-forest-labs/FLUX.2-klein-base-9B",
    "non_distilled": True,
    "memory_cap_hint_gb": 24,
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B",
}
