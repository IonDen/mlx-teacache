"""Failure-path contracts for best-effort handle restoration."""

from types import SimpleNamespace

import pytest

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.stats import TeaCacheStats
from mlx_teacache.handle import TeaCacheHandle, VariantPatch


def _make_handle(rollbacks, finalizers=None, on_restored=None):
    patch = VariantPatch(
        rollbacks=list(rollbacks),
        finalizers=list(finalizers or []),
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


def test_restore_runs_all_finalizers_after_action_failures():
    ran: list[str] = []

    def _raise(label, message):
        ran.append(label)
        raise RuntimeError(message)

    handle = _make_handle(
        rollbacks=[
            lambda: ran.append("rollback-ok"),
            lambda: _raise("rollback-fail", "rollback boom"),
        ],
        finalizers=[
            lambda: _raise("finalizer-fail", "finalizer boom"),
            lambda: ran.append("finalizer-ok"),
        ],
        on_restored=[lambda: ran.append("on-restored")],
    )

    with pytest.raises(RuntimeError, match="rollback boom"):
        handle.restore()

    assert ran == ["rollback-fail", "rollback-ok", "finalizer-fail", "finalizer-ok"]
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
