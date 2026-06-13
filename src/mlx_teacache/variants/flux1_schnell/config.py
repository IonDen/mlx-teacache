"""FLUX.1 schnell configuration. mflux-free."""

from typing import Any

from mlx_teacache.variants.flux1_dev.config import COEFFICIENTS as _DEV_COEFFS

# Same FLUX.1 transformer architecture — reuse the same coefficient tuple object.
# test_coefficients_shared_with_dev asserts `SCHNELL is DEV` (identity check).
COEFFICIENTS: tuple[float, float, float, float, float] = _DEV_COEFFS

DEFAULT_THRESH: float = 0.20

RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 4, "guidance": 1.0},
}

LICENSE: str = "Apache-2.0"

META: dict[str, Any] = {
    "variant_id": "flux1-schnell",
    "display_name": "FLUX.1 schnell",
    "hf_model_id": "black-forest-labs/FLUX.1-schnell",
    "non_distilled": False,
    "memory_cap_hint_gb": None,
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
}
