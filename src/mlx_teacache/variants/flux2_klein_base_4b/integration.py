"""FLUX.2 Klein base 4B integration: its provenance and apply(), which installs the shared
FLUX.2 `_predict` replacement (integrations/mflux/forward_flux2.py).

mflux is never imported at module level. The registry loads this
lazily, after detect.matches() wins.
"""

from typing import Any

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.resolve import BuiltinFit, resolve_rule
from mlx_teacache.handle import TeaCacheHandle
from mlx_teacache.integrations.mflux.forward_flux2 import PredictFactory as PredictFactory
from mlx_teacache.integrations.mflux.forward_flux2 import PredictFn as PredictFn
from mlx_teacache.integrations.mflux.forward_flux2 import (
    _flux2_apply_tail_and_combine as _flux2_apply_tail_and_combine,
)
from mlx_teacache.integrations.mflux.forward_flux2 import _flux2_extract_mod_input as _flux2_extract_mod_input
from mlx_teacache.integrations.mflux.forward_flux2 import _flux2_run_body as _flux2_run_body
from mlx_teacache.integrations.mflux.forward_flux2 import _step_decision_from_gate as _step_decision_from_gate
from mlx_teacache.integrations.mflux.forward_flux2 import (
    flux2_cfg_forward_with_gate as flux2_cfg_forward_with_gate,
)
from mlx_teacache.integrations.mflux.forward_flux2 import flux2_forward_with_gate as flux2_forward_with_gate
from mlx_teacache.integrations.mflux.forward_flux2 import (
    make_teacache_predict_factory as make_teacache_predict_factory,
)
from mlx_teacache.integrations.mflux.patch import install_predict_patch

from .config import CALIBRATED_RANGE, COEFFICIENTS, DEFAULT_THRESH

_PROVENANCE = Provenance(
    source="builtin",
    revision="in-repo-2026-05-17-origin",
    calibration_dataset=(
        "10 prompts × 25 steps × seed=42, M1 Max 32GB, bf16, 512x512, "
        "guidance=1.0, origin-constrained polyfit"
    ),
    fit_metric="constrained-LSQ R^2 on consecutive-step (mod_in, body_out) rel-L1 pairs (poly(0)=0)",
    fit_metric_value=0.10643408169124158,
    reference_url="https://github.com/IonDen/mlx-teacache/blob/main/scripts/calibrate_flux2.py",
    default_thresh=DEFAULT_THRESH,
)


# ---------- apply() — translated from v0.5.x api.py::apply_teacache FLUX.2 branch ----------

_BUILTIN = BuiltinFit(
    coefficients=COEFFICIENTS,
    provenance=_PROVENANCE,
    default_thresh=DEFAULT_THRESH,
    calibrated_range=CALIBRATED_RANGE,
)


def apply(
    flux: Any,
    *,
    rel_l1_thresh: float | None = None,
    coefficients: tuple[float, float, float, float, float] | None = None,
    skip_first_n_steps: int = 1,
    skip_last_n_steps: int = 1,
) -> TeaCacheHandle:
    """Patch a FLUX.2 Klein base 4B model (the other Klein variants call this too).

    FLUX.2 wraps flux._predict (not flux.transformer). VariantPatch rollback
    deletes the _predict instance attribute; finalizer unsubscribes callback.
    No stats finalize in either."""
    # Caller > the variant default with the built-in coefficients > the package fallback 0.20. A custom polynomial
    # gets the fallback because the default was tuned for the bundled one.
    rule = resolve_rule(
        rel_l1_thresh=rel_l1_thresh,
        coefficients=coefficients,
        builtin=_BUILTIN,
        policy="fallback-for-caller-coefficients",
    )
    return install_predict_patch(
        flux,
        make_factory=make_teacache_predict_factory,
        rule=rule,
        skip_first_n_steps=skip_first_n_steps,
        skip_last_n_steps=skip_last_n_steps,
    )
