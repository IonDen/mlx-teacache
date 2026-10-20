"""Qwen-Image-2512 configuration. mflux-free.

Qwen-Image-2512 is a newer checkpoint of the Qwen-Image architecture; on mflux 0.19 and later the `qwen-image`
alias loads it. It shares the transformer, so the gate taps the same signal (Signal A, the modulated block-0 image
input) and the variant reuses the Qwen-Image proxy forward. Only the calibration differs.

COEFFICIENTS are 2512's own fit, read verbatim from
scripts/_calibration_qwen_2512.json["signals"]["A"]["coefficients_c4_to_c0"]
(2026-10-20 chunked calibration on Qwen/Qwen-Image-2512 revision 25468b98e3276ca6700de15c6628e51b7de54a26:
10 prompts / 7 fit + 3 held-out, 50 steps, q4, 768x768, guidance=4.0 CFG, seed=42, mflux 0.23.0;
fit R^2 0.8183, held-out 0.7852).

DEFAULT_THRESH = 0.25 is set from scripts/sweep_threshold_qwen.py --model Qwen/Qwen-Image-2512 --plain-q4 at the
calibration recipe (red-apple prompt, seed 42, 768x768, 50 steps, guidance 4.0, stock q4, text encoders freed after
prompt encoding; mflux 0.23.0, MLX 0.32.2, M1 Max 32 GB, 2026-10-20; 48 active steps). Skipped steps / SSIM against
the vanilla image / longest skip run:

    0.15 -> 22 / 0.9914 / 3
    0.20 -> 27 / 0.9777 / 4
    0.25 -> 29 / 0.9694 / 4
    0.30 -> 32 / 0.9470 / 5
    0.40 -> 35 / 0.9283 / 6

The default is the highest swept threshold whose SSIM stays at or above the 0.95 parity floor, whose longest skip
run stays under the runaway cap of 8, and whose image shows no artefact beyond slightly softer fine speckle on the apple skin.
0.30, the original checkpoint's default, falls below the floor on this checkpoint (0.9470), so 2512 ships a lower
default than the original.
"""

from typing import Any

# Origin-constrained fit (trailing 0.0 = poly(0) = 0). Signal A, read verbatim
# from scripts/_calibration_qwen_2512.json; do not hand-edit. A new calibration bumps
# the integration's provenance revision.
COEFFICIENTS: tuple[float, float, float, float, float] = (
    -13.200421839759947,
    4.851407025105924,
    2.4891133197469846,
    0.2453630032662089,
    0.0,
)

# The rel_l1 range the fit above was made on (scripts/_calibration_qwen_2512.json, signal A: x_min, x_max).
# The gate clamps the measured delta into it before evaluating the polynomial. Stored verbatim; must move with the
# coefficients.
CALIBRATED_RANGE: tuple[float, float] = (0.08094493952514002, 0.4705715918534816)

# Sweep table in the module docstring: 29 of 48 active steps skipped, SSIM 0.9694, longest skip run 4.
DEFAULT_THRESH: float = 0.25

RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 50, "guidance": 4.0},
}

LICENSE: str = "Apache-2.0"

META: dict[str, Any] = {
    "variant_id": "qwen-image-2512",
    "display_name": "Qwen-Image-2512",
    "hf_model_id": "Qwen/Qwen-Image-2512",
    "non_distilled": True,
    "memory_cap_hint_gb": 22,
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/Qwen/Qwen-Image-2512",
}
