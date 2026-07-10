"""Failure-path contracts for best-effort handle restoration."""

from types import SimpleNamespace

import pytest

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.stats import TeaCacheStats
from mlx_teacache.handle import TeaCacheHandle, VariantPatch


def _make_handle(rollbacks, on_restored=None):
    patch = VariantPatch(
        rollbacks=list(rollbacks),
        finalizers=[],
        on_restored=list(on_restored or []),
    )
    return TeaCacheHandle(
        patch=patch,
        stats=TeaCacheStats(),
        provenance=Provenance(source="builtin"),
        rel_l1_thresh=0.2,
    )


def test_restore_runs_all_rollbacks_even_when_one_raises():
    ran: list[str] = []
    patch_rollbacks = [
        lambda: ran.append("first"),
        lambda: (_ for _ in ()).throw(RuntimeError("rollback boom")),
        lambda: ran.append("third"),
    ]
    handle = _make_handle(patch_rollbacks)
    with pytest.raises(RuntimeError, match="rollback boom"):
        handle.restore()
    assert ran == ["third", "first"]
    assert handle._torn_down is False


def test_failed_restore_keeps_sentinel_set():
    flux = SimpleNamespace()
    handle = _make_handle(
        rollbacks=[lambda: (_ for _ in ()).throw(RuntimeError("boom"))],
        on_restored=[lambda: delattr(flux, "_teacache_handle")],
    )
    flux._teacache_handle = handle
    with pytest.raises(RuntimeError, match="boom"):
        handle.restore()
    assert getattr(flux, "_teacache_handle", None) is handle


def test_clean_restore_runs_on_restored_and_marks_torn_down():
    flux = SimpleNamespace()
    handle = _make_handle(
        rollbacks=[],
        on_restored=[lambda: delattr(flux, "_teacache_handle")],
    )
    flux._teacache_handle = handle
    handle.restore()
    assert getattr(flux, "_teacache_handle", None) is None
    assert handle._torn_down is True
