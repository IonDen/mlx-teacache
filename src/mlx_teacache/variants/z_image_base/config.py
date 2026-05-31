"""Z-Image base configuration. mflux-free.

Coefficients are the SIGNAL B fit (first-main-layer residual rel-L1 ->
worst-branch main_out rel-L1), origin-constrained, read verbatim from
scripts/_calibration_z_image.json["signals"]["B"]["coefficients_c4_to_c0"]
(2026-05-31 calibration: 10 prompts / 7 fit + 3 held-out, 50 steps, q8,
512x512, guidance=4.0 CFG, seed=42).

Signal selection rationale (see the 2026-05-31 calibration findings): Signal B
fit R^2 = 0.400 / held-out 0.179. Signal A (noise-refiner output,
caption-independent) was rejected at R^2 = 0.069 — its rel-L1 range
[0.01, 0.12] is too compressed to track the body change. R^2 = 0.400 is in
line with the shipped FLUX.2 variants (klein-base-4b ships at 0.106,
klein-9b at 0.471); per-step fit R^2 is not the arbiter of caching efficacy —
the rescale-poly + accumulator threshold is, and the threshold sweep is the
real go/no-go.
"""

from __future__ import annotations

from typing import Any

# Origin-constrained polyfit (trailing 0.0 = poly(0) = 0). Stored verbatim;
# do not hand-edit. New calibrations bump the integration's provenance revision.
COEFFICIENTS: tuple[float, float, float, float, float] = (
    -898.9907628349583,
    367.7086118008557,
    -45.41511572598643,
    3.95114319842774,
    0.0,
)

# PROVISIONAL — to be set at the SSIM knee by scripts/sweep_threshold_z_image.py
# (post-Phase-3 sweep). 0.15 is the mid-range estimate: the Signal B polynomial
# predicts ~0.09-0.22 per step over the operating range x in [0.03, 0.10], so a
# 0.15 accumulator threshold engages without over-skipping. Confirm + refine.
DEFAULT_THRESH: float = 0.15

RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 50, "guidance": 4.0},
}

LICENSE: str = "Apache-2.0"

META: dict[str, Any] = {
    "variant_id": "z-image-base",
    "display_name": "Z-Image base",
    "hf_model_id": "Tongyi-MAI/Z-Image",
    "non_distilled": True,
    "memory_cap_hint_gb": 22,
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/Tongyi-MAI/Z-Image",
}
