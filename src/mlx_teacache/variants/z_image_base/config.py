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

# The rel_l1 range the fit above was made on (scripts/_calibration_z_image.json, signal B: x_min, x_max).
# The gate clamps the measured delta into it before evaluating the polynomial:
# this fit is origin-constrained (p(0) = 0), so extrapolating below x_min would
# price a tiny delta as almost no change although no calibrated step changed
# that little. Stored verbatim; must move with the coefficients.
CALIBRATED_RANGE: tuple[float, float] = (0.027887196237753466, 0.25914120883567665)

# Set at the SSIM knee from scripts/sweep_threshold_z_image.py
# (tests/_artifacts/sweep_z_image/results_z_image.json, 2026-06-01 sweep):
# SSIM holds >= 0.99 through 0.12 (15/48 steps skipped, SSIM 0.9913) then cliffs
# to ~0.974 at 0.15 and plateaus. 0.12 is the quality-first default — just before
# the cliff, near-indistinguishable from vanilla. Skip counts are deterministic;
# the headline speedup is the 3-rep bench (_artifacts/v0.10.0_bench_z_image.json;
# the earlier v0.7.0 report is _artifacts/_bench_z_image_v0_7_0.json).
DEFAULT_THRESH: float = 0.12

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
