"""FLUX.1 schnell integration. Uses the shared FLUX.1 proxy and forward; the code
is shared (same transformer architecture). Only the public-facing
metadata (provenance) and the apply() defaults differ.
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
    revision="upstream-flux-v1-shared",
    calibration_dataset="upstream ali-vilab TeaCache (FLUX architecture is shared between dev and schnell)",
    reference_url="https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py",
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
    """FLUX.1 schnell apply. Public-API-equivalent of the FLUX.1 schnell
    branch of v0.5.x apply_teacache."""
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
