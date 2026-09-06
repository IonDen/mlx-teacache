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
768x768/50-step recipe (red-apple, seed 42, Signal A). Re-swept 2026-09-06 on
stock q4 under the v0.10.0 consecutive-delta gate: 0.15 -> 24 of 48 active steps
skipped, SSIM 0.980, longest streak 2; 0.20 -> 26 / 0.980 / 2; 0.25 -> 30 / 0.976 / 3;
0.30 -> 33 / 0.967 / 4. SSIM degrades gracefully with NO cliff and every point
clears the 0.95 parity floor, so 0.30 stays: the fastest point that still holds
visual equivalence (the 0.9.x quality point is back at 0.20). The first sweep
(2026-06, mixed-precision build, 0.9.x gate: 0.9951 at 0.20, 0.9873 at 0.30,
0.9809 at 0.40, 0.9783 at 0.50; 24 of 48 skipped at 0.30) is superseded. The
sweep's single-rep wall-clock is thermal noise; the headline speedup comes from
the multi-rep bench.
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

# From scripts/sweep_threshold_qwen.py (768x768/50, Signal A, stock q4, v0.10.0 gate,
# 2026-09-06): SSIM degrades gracefully with no cliff (0.980 at 0.20, 0.976 at 0.25,
# 0.967 at 0.30); 33/48 active steps skipped at 0.30, longest streak 4. Reaffirmed.
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
