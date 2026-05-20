"""TeaCacheHandle + VariantPatch contract.

Critical property (audit F2/F3): handle does NOT call stats finalize.
Stats commit/discard stays in the mflux lifecycle wrapper.
"""
from __future__ import annotations


def test_handle_runs_rollbacks_in_reverse_install_order():
    from mlx_teacache._kernel.stats import TeaCacheStats
    from mlx_teacache.handle import TeaCacheHandle, VariantPatch

    log: list[str] = []
    patch = VariantPatch(
        rollbacks=[lambda: log.append("r1"), lambda: log.append("r2")],
        finalizers=[],
    )
    stats = TeaCacheStats()
    h = TeaCacheHandle(patch=patch, stats=stats,
                      provenance=_dummy_provenance(), rel_l1_thresh=0.2)
    h.restore()
    assert log == ["r2", "r1"]


def test_handle_restore_is_idempotent():
    from mlx_teacache._kernel.stats import TeaCacheStats
    from mlx_teacache.handle import TeaCacheHandle, VariantPatch

    counter = {"n": 0}
    patch = VariantPatch(rollbacks=[lambda: counter.update(n=counter["n"] + 1)], finalizers=[])
    h = TeaCacheHandle(patch=patch, stats=TeaCacheStats(),
                      provenance=_dummy_provenance(), rel_l1_thresh=0.2)
    h.restore()
    h.restore()
    assert counter["n"] == 1


def test_handle_does_not_finalize_stats():
    """Audit F2: stats commit stays in mflux lifecycle. Handle restore must
    not call finalize_last_generation; that's the lifecycle's job in
    integrations/mflux/lifecycle.py:wrap_generate_image."""
    from mlx_teacache._kernel.stats import StepDecision, TeaCacheStats
    from mlx_teacache.handle import TeaCacheHandle, VariantPatch

    stats = TeaCacheStats()
    stats.record(StepDecision(step_idx=0, timestep=1.0, rel_l1=None,
                              accumulated_distance=0.0, decision="computed"))
    # Staging has 1 entry; public counters still 0.
    h = TeaCacheHandle(patch=VariantPatch(), stats=stats,
                      provenance=_dummy_provenance(), rel_l1_thresh=0.2)
    h.restore()
    # Public counters unchanged — restore did NOT commit.
    assert stats.computed_count == 0
    assert stats.generations == 0


def test_handle_has_no_variant_branches():
    """Audit F3: TeaCacheHandle is variant-agnostic. Static-grep check."""
    import inspect

    from mlx_teacache import handle as handle_module

    source = inspect.getsource(handle_module)
    code = "\n".join(ln for ln in source.splitlines()
                     if not ln.lstrip().startswith("#")).lower()
    for bad in ("flux1", "flux2", "klein"):
        assert bad not in code, f"handle.py must not mention {bad!r}"


def _dummy_provenance():
    from mlx_teacache._kernel.coefficients import Provenance
    return Provenance(source="builtin")
