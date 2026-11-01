"""FLUX.1 dev integration: its provenance and apply(), which installs the shared FLUX.1
proxy (integrations/mflux/forward_flux1.py) and the lifecycle hooks.

mflux is never imported at module level. The package registry loads
this lazily, after detect.matches() wins.
"""

from typing import Any

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.resolve import BuiltinFit, resolve_rule
from mlx_teacache.handle import TeaCacheHandle
from mlx_teacache.integrations._session import InternalHandle
from mlx_teacache.integrations.mflux.forward_flux1 import (
    ProxyFlux1Transformer as ProxyFlux1Transformer,
)
from mlx_teacache.integrations.mflux.forward_flux1 import (
    _flux1_extract_mod_input as _flux1_extract_mod_input,
)
from mlx_teacache.integrations.mflux.forward_flux1 import (
    _flux1_prelude as _flux1_prelude,
)
from mlx_teacache.integrations.mflux.forward_flux1 import (
    _flux1_run_body as _flux1_run_body,
)
from mlx_teacache.integrations.mflux.forward_flux1 import (
    _flux1_tail as _flux1_tail,
)
from mlx_teacache.integrations.mflux.forward_flux1 import (
    _step_decision_from_gate as _step_decision_from_gate,
)
from mlx_teacache.integrations.mflux.forward_flux1 import (
    flux1_forward_with_gate as flux1_forward_with_gate,
)
from mlx_teacache.integrations.mflux.patch import install_proxy_patch

from .config import COEFFICIENTS, DEFAULT_THRESH

_PROVENANCE = Provenance(
    source="builtin",
    revision="upstream-flux-v1",
    calibration_dataset="upstream ali-vilab TeaCache (no in-repo calibration)",
    reference_url="https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py",
    default_thresh=DEFAULT_THRESH,
)


# The shared session type under the name the tests still import from here.
_InternalHandle = InternalHandle


# ----- apply() — translated from v0.5.x api.py::apply_teacache FLUX.1 branch -----


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
    """FLUX.1 dev apply. Public-API-equivalent of the FLUX.1 branch of
    v0.5.x apply_teacache."""
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
