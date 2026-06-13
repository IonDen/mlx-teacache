"""FLUX.2 Klein 4B (distilled) configuration. mflux-free."""

from typing import Any

# Derived in-repo by scripts/calibrate_flux2.py --variant klein-4b on 2026-05-15:
#   10 prompts × 8 steps × seed=42 on M1 Max 32GB, bf16, guidance=1.0,
#   512×512, vanilla forward (no caching). 70 consecutive-step pairs of
#   (rel_l1(mod_in_t, mod_in_{t-1}), rel_l1(body_out_t, body_out_{t-1})).
#   numpy.polyfit degree=4 → R² = 0.6530.
# See scripts/_calibration_flux2_klein_4b.json for the full report.
# Stored verbatim; do not hand-edit. New calibrations bump revision and minor version.
COEFFICIENTS: tuple[float, float, float, float, float] = (
    236.9190176127698,
    -201.47401360106662,
    66.91354236854073,
    -11.14796738073235,
    1.2674506310647067,
)

DEFAULT_THRESH: float | None = None  # distilled gate doesn't engage; use package fallback 0.20

RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 8, "guidance": 1.0},
}

LICENSE: str = "Apache-2.0"

META: dict[str, Any] = {
    "variant_id": "flux2-klein-4b",
    "display_name": "FLUX.2 Klein 4B",
    "hf_model_id": "black-forest-labs/FLUX.2-klein-4B",
    "non_distilled": False,
    "memory_cap_hint_gb": None,
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
}
