"""FLUX.1 Krea [dev] configuration. mflux-free.

Krea [dev] is a FLUX.1-dev-architecture finetune published by Black Forest Labs
with Krea (``black-forest-labs/FLUX.1-Krea-dev``); mflux exposes it through the
same FLUX.1 loader as ``dev``. It shares the transformer, so the gate taps the
same modulated block-0 input and the variant reuses the FLUX.1 proxy strategy.

COEFFICIENTS start as FLUX.1-dev's vendored ali-vilab tuple. A finetune shifts
the distribution that polynomial was fit on, so the tuple is scored on Krea's
own calibration pairs by ``scripts/calibrate_flux1.py`` before release; the
provenance in ``integration.py`` records the measured R² and whether the shared
tuple was kept or a Krea-specific fit shipped.
"""

from typing import Any

# FLUX.1-dev's vendored tuple (see variants/flux1_dev/config.py for the upstream
# provenance and the poly(0) note). Replaced only if Krea's own calibration says so.
COEFFICIENTS: tuple[float, float, float, float, float] = (
    498.651651244,
    -283.781631,
    55.8554382,
    -3.82021401,
    0.264230861,
)

# Provisional: FLUX.1-dev's default until the Krea threshold sweep sets it.
DEFAULT_THRESH: float = 0.20

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
