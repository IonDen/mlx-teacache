"""Qwen-Image base configuration. mflux-free.

COEFFICIENTS are the SIGNAL A fit (modulated block-0 image input rel-L1 ->
worst-branch body_out rel-L1), origin-constrained, read verbatim from
scripts/_calibration_qwen.json["signals"]["A"]["coefficients_c4_to_c0"]
(2026-06-17 chunked calibration: 10 prompts / 7 fit + 3 held-out, 50 steps, q4,
768x768, guidance=4.0 CFG, seed=42; fit R^2 0.8490, held-out 0.8451).

Signal A (the integration's gate signal) was kept over Signal B (first-block
residual, R^2 0.8809) despite B's marginally higher fit: A is caption-independent
(so the shared CFG gate decision is exact) and cheaper on a skip step (no extra
block-0 run). Qwen-Image is FLUX-shaped, so the FLUX-canonical modulated-input
signal calibrates well (R^2 ~0.85, vs Z-Image's 0.40 and FLUX.2's 0.11-0.47). The
R^2 is lower than the prior 512x512/20-step fit (0.9464) because the heavier
768x768/50-step recipe samples a finer, noisier per-step rel-L1 distribution.

DEFAULT_THRESH = 0.30 is set from scripts/sweep_threshold_qwen.py at the
768x768/50-step recipe (red-apple, seed 42, Signal A), run on the recommended
mixed-precision build (q8 edge blocks + bf16 embeddings — see docs/variants;
mlx-teacache itself stays quant-agnostic). The stock-q4 coefficients transfer
cleanly to it (a sensible 0->29 skip ramp at high SSIM). SSIM degrades gracefully
with NO cliff: 0.9951 at 0.20, 0.9873 at 0.30, 0.9809 at 0.40, 0.9783 at 0.50. At
0.30 the gate skips 24 of 48 active steps (~50%) at SSIM 0.987 — visually identical
to vanilla. The sweep's single-rep wall-clock is thermal noise; the headline speedup
comes from the multi-rep bench.
"""

from typing import Any

# Origin-constrained fit (trailing 0.0 = poly(0) = 0). Signal A, read verbatim
# from scripts/_calibration_qwen.json; do not hand-edit. A new calibration bumps
# the integration's provenance revision.
COEFFICIENTS: tuple[float, float, float, float, float] = (
    -12.954226906135869,
    8.883805167578382,
    -0.9363839862290331,
    1.4538816050570036,
    0.0,
)

# From scripts/sweep_threshold_qwen.py (768x768/50, Signal A, mixed-precision build):
# SSIM degrades gracefully with no cliff (0.987 at 0.30); 24/48 active steps skipped
# (~50%) at 0.30, visually identical to vanilla. Quality-first default.
DEFAULT_THRESH: float = 0.30

RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 50, "guidance": 4.0},
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
