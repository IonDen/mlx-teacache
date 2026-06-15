"""Qwen-Image base configuration. mflux-free.

PROVISIONAL (v0.9.0 in-flight): COEFFICIENTS and DEFAULT_THRESH below are
placeholders that DISABLE caching (DEFAULT_THRESH = 0.0 → the gate fast-path
always computes, never caches → identical to vanilla). They are replaced with the
calibrated Signal-A/B fit + the SSIM-knee threshold by scripts/calibrate_qwen.py
(plan Task 7). Do NOT ship v0.9.0 with these placeholders — tests/test_calibration
_artifacts.py pins the real JSON and will fail until calibration lands.
"""

from typing import Any

# PLACEHOLDER — replaced by the calibrated origin-constrained fit (Task 7).
COEFFICIENTS: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

# PLACEHOLDER — 0.0 disables caching (safe vanilla behavior). Task 7 sets the
# SSIM-knee value from scripts/sweep (or the calibrate script).
DEFAULT_THRESH: float = 0.0

RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 20, "guidance": 4.0},
}

LICENSE: str = "Apache-2.0"

META: dict[str, Any] = {
    "variant_id": "qwen-image",
    "display_name": "Qwen-Image",
    "hf_model_id": "Qwen/Qwen-Image",
    "non_distilled": True,
    "memory_cap_hint_gb": 22,
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/Qwen/Qwen-Image",
}
