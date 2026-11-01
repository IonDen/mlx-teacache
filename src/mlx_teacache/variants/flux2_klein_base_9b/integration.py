"""FLUX.2 Klein base 9B integration. Uses the shared FLUX.2 forward + factory
(the same code as flux2_klein_base_4b). The 9B variant uses the SAME forward
path (both no-CFG at low_step recipe and CFG-per-branch at the canonical
50-step + g=4.0 recipe) — only the metadata (Provenance) and memory_cap
hint differ.
"""

from typing import Any

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.resolve import BuiltinFit, resolve_rule
from mlx_teacache.handle import TeaCacheHandle
from mlx_teacache.integrations.mflux.forward_flux2 import make_teacache_predict_factory
from mlx_teacache.integrations.mflux.patch import install_predict_patch

from .config import CALIBRATED_RANGE, COEFFICIENTS, DEFAULT_THRESH

_PROVENANCE = Provenance(
    source="builtin",
    revision="in-repo-2026-05-18-reuse-base-4b",
    calibration_dataset=(
        "REUSED from flux2-klein-base-4b — same architecture family + same recipe "
        "(10 prompts × 25 steps × seed=42, M1 Max 32GB, bf16, 512x512, guidance=1.0, "
        "origin-constrained polyfit); validated empirically at 50 steps + guidance=4.0"
    ),
    fit_metric="constrained-LSQ R^2 on consecutive-step (mod_in, body_out) rel-L1 pairs (poly(0)=0)",
    fit_metric_value=0.10643408169124158,
    reference_url="https://github.com/IonDen/mlx-teacache/blob/main/_artifacts/validation_klein_base_9b.json",
    default_thresh=DEFAULT_THRESH,
)

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
    """FLUX.2 Klein base 9B apply. Mirrors flux2_klein_base_4b.apply() but
    with this variant's COEFFICIENTS, DEFAULT_THRESH, and _PROVENANCE."""
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
