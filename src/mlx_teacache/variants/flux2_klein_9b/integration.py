"""FLUX.2 Klein 9B (distilled) integration. Uses the shared FLUX.2 forward
+ factory (the same code as flux2_klein_base_4b). Distilled klein-9b at
g=1.0 uses only the no-CFG forward path; CFG forward is not exercised.
"""

from typing import Any

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.resolve import BuiltinFit, resolve_rule
from mlx_teacache.handle import TeaCacheHandle
from mlx_teacache.integrations.mflux.forward_flux2 import make_teacache_predict_factory
from mlx_teacache.integrations.mflux.patch import install_predict_patch

from .config import COEFFICIENTS, DEFAULT_THRESH

_PROVENANCE = Provenance(
    source="builtin",
    revision="in-repo-2026-05-16-origin",
    calibration_dataset=(
        "10 prompts × 8 steps × seed=42, M1 Max 32GB, bf16, 512x512, guidance=1.0, origin-constrained polyfit"
    ),
    fit_metric="constrained-LSQ R^2 on 70 consecutive-step (mod_in, body_out) rel-L1 pairs (poly(0)=0)",
    fit_metric_value=0.4710289350635284,
    reference_url="https://github.com/IonDen/mlx-teacache/blob/main/scripts/calibrate_flux2.py",
)

_BUILTIN = BuiltinFit(
    coefficients=COEFFICIENTS,
    provenance=_PROVENANCE,
    default_thresh=DEFAULT_THRESH,
    calibrated_range=None,
)


def apply(
    flux: Any,
    *,
    rel_l1_thresh: float | None = None,
    coefficients: tuple[float, float, float, float, float] | None = None,
    skip_first_n_steps: int = 1,
    skip_last_n_steps: int = 1,
) -> TeaCacheHandle:
    """FLUX.2 Klein 9B (distilled) apply.

    Distilled gate doesn't engage at the package fallback rel_l1_thresh
    (0.20); the wrapper is still useful through mx.compile-path avoidance
    on Max/Ultra chips. DEFAULT_THRESH is None — caller's value (if any)
    or the package fallback wins.
    """
    # Caller > the variant default, whatever the coefficients > the package fallback 0.20. DEFAULT_THRESH is None,
    # so without a caller threshold the fallback applies.
    rule = resolve_rule(
        rel_l1_thresh=rel_l1_thresh,
        coefficients=coefficients,
        builtin=_BUILTIN,
        policy="variant-default",
    )
    return install_predict_patch(
        flux,
        make_factory=make_teacache_predict_factory,
        rule=rule,
        skip_first_n_steps=skip_first_n_steps,
        skip_last_n_steps=skip_last_n_steps,
    )
