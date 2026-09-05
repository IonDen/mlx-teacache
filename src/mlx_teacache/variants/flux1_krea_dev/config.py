"""FLUX.1 Krea [dev] configuration. mflux-free.

Krea [dev] is a FLUX.1-dev-architecture finetune published by Black Forest Labs
with Krea (``black-forest-labs/FLUX.1-Krea-dev``); mflux exposes it through the
same FLUX.1 loader as ``dev``. It shares the transformer, so the gate taps the
same modulated block-0 input and the variant reuses the FLUX.1 proxy strategy.

COEFFICIENTS are Krea's own fit. FLUX.1-dev's vendored tuple was scored on
Krea's calibration pairs by ``scripts/calibrate_flux1.py`` and does not
transfer: a finetune shifts the per-step change distribution the polynomial
was fit on (see the note above the tuple), so a Krea-specific fit ships.
"""

from typing import Any

# Krea's own fit, read verbatim from scripts/_calibration_flux1_krea_dev.json
# (free fit over 10 prompts x 27 consecutive-step pairs at 28 steps / g=4.5 /
# 512x512 / q4 / seed 42; R^2 0.6817). FLUX.1-dev's vendored tuple was
# scored on the same pairs and does NOT transfer: R^2 -496.3 — its
# 499*x^4 term explodes at Krea's per-step changes (rel-L1 up to ~0.66, roughly
# three times dev's). Do not hand-edit; a new calibration bumps the provenance.
COEFFICIENTS: tuple[float, float, float, float, float] = (
    24.18037744588824,
    -45.41912128506917,
    26.20213376107934,
    -3.579234954660995,
    0.40140693642366004,
)

# From scripts/calibrate_flux1.py --model krea-dev --sweep (red-apple prompt, 28 steps,
# g=4.5, 512x512, q4, seed 42, 2026-09-05): SSIM vs vanilla 0.990 at 0.30 with 10 of
# 26 active steps skipped and no two skips in a row; 0.890 at 0.35, 0.888 at 0.40,
# 0.866 at 0.60, 0.863 at 0.80. The knee is sharp, so 0.30 ships. The polynomial's
# intercept (~0.40 predicted change per step, minimum ~0.26 over the calibration
# trajectories) means the package fallback 0.20 would skip nothing on this model.
DEFAULT_THRESH: float = 0.30

# Black Forest Labs' published recipe for Krea [dev]: 28 steps, guidance 4.5.
RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 28, "guidance": 4.5},
}

LICENSE: str = "FLUX.1-dev Non-Commercial License"

META: dict[str, Any] = {
    "variant_id": "flux1-krea-dev",
    "display_name": "FLUX.1 Krea [dev]",
    "hf_model_id": "black-forest-labs/FLUX.1-Krea-dev",
    "non_distilled": True,
    "memory_cap_hint_gb": None,
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev",
}
