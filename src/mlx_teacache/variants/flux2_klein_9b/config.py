"""FLUX.2 Klein 9B (distilled) configuration. mflux-free."""
from __future__ import annotations

from typing import Any

# Derived in-repo by scripts/calibrate_flux2.py --variant klein-9b
# --fit-mode origin on 2026-05-16: 10 prompts × 8 steps × seed=42 on
# M1 Max 32GB, bf16, guidance=1.0, 512×512, vanilla forward (no caching).
# 70 consecutive-step pairs of (rel_l1(mod_in_t, mod_in_{t-1}),
# rel_l1(body_out_t, body_out_{t-1})). Origin-constrained least-squares
# fit (forces poly(0)=0 so the polynomial is physically sensible at small
# input rel_l1). See scripts/_calibration_flux2_klein_9b.json for the
# full report.
#
# Note: at the package default rel_l1_thresh=0.20 these coefficients do
# not trigger any skips on Klein 9B's 8-step distilled schedule — the
# empirical y range starts at 0.25 (every adjacent-step body_out change
# exceeds the threshold). Apply does not raise or warn at this state;
# the wrapper is still useful through `mx.compile` avoidance on chips
# that compile `_predict`. See docs/superpowers/notes/ for the
# 2026-05-16 postmortem and the v0.4 FLUX.2 research task.
#
# Stored verbatim; do not hand-edit. New calibrations bump revision.
COEFFICIENTS: tuple[float, float, float, float, float] = (
    -523.8412980807129,
    530.2492512602308,
    -177.64385734082498,
    20.893264957040557,
    0.0,
)

DEFAULT_THRESH: float | None = None  # distilled gate doesn't engage; use package fallback 0.20

RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 8, "guidance": 1.0},
}

LICENSE: str = "FLUX Non-Commercial"

META: dict[str, Any] = {
    "variant_id": "flux2-klein-9b",
    "display_name": "FLUX.2 Klein 9B",
    "hf_model_id": "black-forest-labs/FLUX.2-klein-9B",
    "non_distilled": False,
    "memory_cap_hint_gb": None,
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B",
}
