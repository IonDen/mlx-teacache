"""FLUX.2 Klein base 4B configuration. mflux-free."""

from typing import Any

# Origin-constrained polyfit, derived in-repo on 2026-05-17 from
# flux2-klein-base-4B at 25-step schedule (non-distilled). The trailing
# 0.0 reflects the origin constraint (poly(0) = 0). Coefficient values
# read from scripts/_calibration_flux2_klein_base_4b.json's
# coefficients_c4_to_c0 field. R^2 is low (0.106) — much lower than
# FLUX.1-family or Klein 9B. The polynomial output range [0.144, 0.233]
# straddles the package default rel_l1_thresh=0.20, so the gate is
# structurally capable of engaging (unlike distilled Klein 4B/9B where
# the polynomial never dips below 0.20). The bench in v0.4.0's release
# gate confirms engagement empirically.
#
# Stored verbatim; do not hand-edit. New calibrations bump revision.
COEFFICIENTS: tuple[float, float, float, float, float] = (
    -1841.022165607874,
    848.4417137572868,
    -131.3554469956159,
    8.179509586828413,
    0.0,
)

DEFAULT_THRESH: float = 0.17

RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 50, "guidance": 4.0},
    "low_step": {"num_inference_steps": 25, "guidance": 1.0},
}

LICENSE: str = "Apache-2.0"

META: dict[str, Any] = {
    "variant_id": "flux2-klein-base-4b",
    "display_name": "FLUX.2 Klein base 4B",
    "hf_model_id": "black-forest-labs/FLUX.2-klein-base-4B",
    "non_distilled": True,
    "memory_cap_hint_gb": None,
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B",
}
