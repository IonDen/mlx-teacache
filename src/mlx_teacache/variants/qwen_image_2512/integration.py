"""Qwen-Image-2512 integration. Uses the Qwen-Image proxy and forward through `apply_qwen`; only the calibration
differs. mflux imported lazily (the registry loads this only after detect.matches() wins)."""

from typing import Any

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache.handle import TeaCacheHandle
from mlx_teacache.variants.qwen_image.integration import QwenCalibration, apply_qwen

from .config import CALIBRATED_RANGE, COEFFICIENTS, DEFAULT_THRESH, META

_PROVENANCE = Provenance(
    source="builtin",
    revision="in-repo-2026-10-20-origin-signalA-768-50-2512",
    calibration_dataset=(
        "10 prompts (7 fit / 3 held-out) x 50 steps x seed=42, M1 Max 32GB, q4, "
        "768x768, guidance=4.0 (CFG), origin-constrained polyfit, chunked per-prompt; "
        "Qwen/Qwen-Image-2512 revision 25468b98e3276ca6700de15c6628e51b7de54a26, mflux 0.23.0"
    ),
    fit_metric=(
        "constrained-LSQ R^2 on consecutive-step Signal-A (modulated block-0 input, "
        "worst-branch body_out) rel-L1 pairs (poly(0)=0)"
    ),
    fit_metric_value=0.8183,
    reference_url="https://github.com/IonDen/mlx-teacache/blob/main/scripts/calibrate_qwen.py",
    default_thresh=DEFAULT_THRESH,
)

CALIBRATION_2512 = QwenCalibration(
    variant_id=META["variant_id"],
    coefficients=COEFFICIENTS,
    calibrated_range=CALIBRATED_RANGE,
    default_thresh=DEFAULT_THRESH,
    provenance=_PROVENANCE,
    hf_model_id=META["hf_model_id"],
)


def apply(
    flux: Any,
    *,
    rel_l1_thresh: float | None = None,
    coefficients: tuple[float, float, float, float, float] | None = None,
    skip_first_n_steps: int = 1,
    skip_last_n_steps: int = 1,
) -> TeaCacheHandle:
    """Qwen-Image-2512 apply: the shared Qwen apply with this checkpoint's calibration.

    A one-line delegator on purpose: apply_qwen's uncalibrated-checkpoint warning uses a stacklevel that assumes
    exactly one wrapper frame between it and apply_teacache."""
    return apply_qwen(
        flux,
        CALIBRATION_2512,
        rel_l1_thresh=rel_l1_thresh,
        coefficients=coefficients,
        skip_first_n_steps=skip_first_n_steps,
        skip_last_n_steps=skip_last_n_steps,
    )
