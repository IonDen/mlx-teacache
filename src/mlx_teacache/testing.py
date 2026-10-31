"""Helpers for tests that need a TeaCacheHandle without a model.

`fake_handle()` builds a handle that patches nothing. The caller sets its stats, provenance, rel_l1_thresh,
variant_id, coefficients, decision_rule_revision and mflux_version; the skip window is fixed at 1 and 1.
`reset_untested_mflux_warning()` re-arms the once-per-process warning about an unverified mflux release.

This module imports no mflux and no pytest, so it works on an install without the [mflux] extra and outside a test
runner. `import mlx_teacache` does not import it; import `mlx_teacache.testing` where you need it. Its names are part
of the public API covered by the stability promise at
https://github.com/IonDen/mlx-teacache/blob/main/docs/stability.md.
"""

from collections.abc import Sequence

from mlx_teacache import _mflux_versions
from mlx_teacache._kernel import decision_rule as _decision_rule
from mlx_teacache._kernel.coefficients import Provenance, validate_custom
from mlx_teacache._kernel.stats import TeaCacheStats
from mlx_teacache.handle import TeaCacheHandle, VariantPatch

__all__ = ["fake_handle", "reset_untested_mflux_warning"]


def fake_handle(
    *,
    stats: TeaCacheStats | None = None,
    provenance: Provenance | None = None,
    rel_l1_thresh: float = 0.12,
    variant_id: str = "fake",
    coefficients: Sequence[float] | None = None,
    decision_rule_revision: int | None = None,
    mflux_version: str | None = None,
) -> TeaCacheHandle:
    """A TeaCacheHandle that patches nothing, for tests that need one without loading a model.

    The caller controls stats, provenance, rel_l1_thresh, variant_id, coefficients, decision_rule_revision and
    mflux_version; the skip window is fixed at 1 and 1. stats defaults to a new, empty TeaCacheStats and provenance to Provenance.for_user_supplied().
    coefficients defaults to five zeros; a value you pass is checked like apply_teacache's (five finite floats,
    otherwise TeaCacheValueError) and stored as a tuple. decision_rule_revision defaults to the current revision.
    The fixed window matches the apply_teacache defaults. restore() works as on a real handle: it freezes the
    stats, and a second call does nothing. Building the handle emits no warning."""
    handle = TeaCacheHandle(
        patch=VariantPatch(),
        stats=TeaCacheStats() if stats is None else stats,
        provenance=Provenance.for_user_supplied() if provenance is None else provenance,
        rel_l1_thresh=rel_l1_thresh,
    )
    handle._variant_id = variant_id
    if coefficients is not None:
        handle._coefficients = validate_custom(coefficients)
    else:
        handle._coefficients = (0.0, 0.0, 0.0, 0.0, 0.0)
    handle._decision_rule_revision = (
        _decision_rule.DECISION_RULE_REVISION if decision_rule_revision is None else decision_rule_revision
    )
    handle._mflux_version = mflux_version
    return handle


def reset_untested_mflux_warning() -> None:
    """Re-arm the warning apply_teacache emits once per process on an mflux release newer than the newest one this
    version was verified on, so the next apply_teacache call can emit it again."""
    _mflux_versions._warned = False
