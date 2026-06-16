"""Qwen-Image base configuration. mflux-free.

COEFFICIENTS are the SIGNAL A fit (modulated block-0 image input rel-L1 ->
worst-branch body_out rel-L1), origin-constrained, read verbatim from
scripts/_calibration_qwen.json["signals"]["A"]["coefficients_c4_to_c0"]
(2026-06-16 calibration: 10 prompts / 7 fit + 3 held-out, 20 steps, q4, 512x512,
guidance=4.0 CFG, seed=42; fit R^2 0.9464, held-out 0.9439).

Signal A (the integration's gate signal) was chosen over Signal B (first-block
residual, R^2 0.9516) despite B's marginally higher fit: A is caption-independent
(so the shared CFG gate decision is exact) and cheaper on a skip step (no extra
block-0 run). Qwen-Image is FLUX-shaped, so the FLUX-canonical modulated-input
signal calibrates well (R^2 ~0.95, vs Z-Image's 0.40 and FLUX.2's 0.11-0.47).

DEFAULT_THRESH is still 0.0 (caching disabled, gate fast-path = vanilla) pending
the SSIM-knee threshold sweep. The 512x512 recipe is a memory fallback from the
nominal 768x768 (768^2 peaked 28.3 GB > the 24.96 GB device working-set ceiling
on a 32 GB M1 Max).
"""

from typing import Any

# Origin-constrained fit (trailing 0.0 = poly(0) = 0). Signal A, read verbatim
# from scripts/_calibration_qwen.json; do not hand-edit. A new calibration bumps
# the integration's provenance revision.
COEFFICIENTS: tuple[float, float, float, float, float] = (
    45.1442944188745,
    -52.91131412057809,
    17.078368811436963,
    0.10393043235768819,
    0.0,
)

# Still 0.0 (caching disabled) pending the SSIM-knee sweep — set to the knee
# threshold once the sweep characterizes skip-count vs SSIM for Signal A.
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
