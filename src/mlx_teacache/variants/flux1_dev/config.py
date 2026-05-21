"""FLUX.1 dev configuration. mflux-free."""

from __future__ import annotations

from typing import Any

# Vendored from https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py
# License: Apache-2.0. See NOTICE for attribution.
#
# Upstream source uses these values in numpy poly1d order (high-to-low):
#   coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01,
#                   -3.82021401e+00,  2.64230861e-01]
#   rescale_func = np.poly1d(coefficients)
#
# Our `poly_eval` uses the same high-to-low convention (see gate.py). The
# earlier version of this constant had coefficients c3..c0 transcribed
# incorrectly (predicted distances ~10x too large), which prevented the
# default rel_l1_thresh=0.25 from ever skipping a step on FLUX.1-dev. Fixed
# 2026-05-15; verified via the test_image_quality_flux1.py SSIM gate that
# the cache now engages at the documented threshold.
COEFFICIENTS: tuple[float, float, float, float, float] = (
    498.651651244,
    -283.781631,
    55.8554382,
    -3.82021401,
    0.264230861,
)

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
