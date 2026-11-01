"""FLUX.1 Krea [dev] integration. Uses the shared FLUX.1 proxy and forward;
the code is shared. Krea is a FLUX.1-dev-architecture finetune, so the transformer,
the gate signal and the patch strategy are the same. Only the metadata
(provenance, recipe, default threshold) differs.
"""

from typing import Any

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.resolve import BuiltinFit, resolve_rule
from mlx_teacache.handle import TeaCacheHandle
from mlx_teacache.integrations.mflux.forward_flux1 import ProxyFlux1Transformer
from mlx_teacache.integrations.mflux.patch import install_proxy_patch

from .config import COEFFICIENTS, DEFAULT_THRESH

_PROVENANCE = Provenance(
    source="builtin",
    revision="in-repo-2026-09-05",
    calibration_dataset=(
        "10 prompts x 28 steps x seed=42, M1 Max 32GB, bf16, q4, 512x512, guidance=4.5 "
        "(scripts/calibrate_flux1.py --model krea-dev); FLUX.1-dev's vendored tuple scored R^2 -496 "
        "on the same pairs, so Krea ships its own fit"
    ),
    fit_metric="numpy.polyfit R^2 on 270 consecutive-step (mod_in, body_out) rel-L1 pairs",
    fit_metric_value=0.6817,
    reference_url="https://github.com/IonDen/mlx-teacache/blob/main/scripts/calibrate_flux1.py",
    default_thresh=DEFAULT_THRESH,
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
    """FLUX.1 Krea [dev] apply. Same mechanics as FLUX.1 dev: proxy the
    transformer, register the lifecycle callback, wrap generate_image."""
    # FLUX.1 keeps its default threshold whatever the coefficients.
    rule = resolve_rule(
        rel_l1_thresh=rel_l1_thresh, coefficients=coefficients, builtin=_BUILTIN, policy="variant-default"
    )
    return install_proxy_patch(
        flux,
        make_proxy=ProxyFlux1Transformer,
        rule=rule,
        skip_first_n_steps=skip_first_n_steps,
        skip_last_n_steps=skip_last_n_steps,
    )
