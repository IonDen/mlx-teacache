"""mlx_teacache.testing: a handle without a model, and a way to re-arm the untested-mflux warning.

Pure-core. Expected defaults are literals from the module's documented signature. The import checks run in fresh
interpreters so they cannot depend on what earlier tests already imported.
"""

import subprocess
import sys
import types
import warnings

import pytest

import mlx_teacache._mflux_versions as versions
from mlx_teacache import (
    Provenance,
    StatsFrozenError,
    StepDecision,
    TeaCacheHandle,
    TeaCacheStats,
    TeaCacheUntestedMfluxWarning,
    TeaCacheValueError,
)
from mlx_teacache._kernel.gate import DECISION_RULE_REVISION

# The next minor release after the newest verified one (derived as in test_untested_mflux_warning).
_VERIFIED = versions.release_tuple(versions.NEWEST_VERIFIED_MFLUX)
assert _VERIFIED is not None
_MAJOR, _MINOR, *_ = _VERIFIED
NEWER = f"{_MAJOR}.{_MINOR + 1}.0"


def _testing() -> types.ModuleType:
    """Imported per test, so a missing helper fails that test and not the whole file at collection."""
    import mlx_teacache.testing

    return mlx_teacache.testing


def test_fake_handle_defaults() -> None:
    """Bug: a default drifts (a placeholder changes, or a fact is left unset and raises)."""
    h = _testing().fake_handle()
    assert isinstance(h, TeaCacheHandle)
    assert h.stats.generations == 0
    assert h.provenance == Provenance(source="user")
    assert h.rel_l1_thresh == 0.12
    assert h.variant_id == "fake"
    assert h.coefficients == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert (h.skip_first_n_steps, h.skip_last_n_steps) == (1, 1)
    assert h.decision_rule_revision == 1
    assert h.decision_rule_revision == DECISION_RULE_REVISION
    assert h.mflux_version is None


def test_fake_handle_defaults_do_not_share_stats() -> None:
    """Bug: a module-level default TeaCacheStats, so one test's recorded steps leak into the next fake."""
    testing = _testing()
    assert testing.fake_handle().stats is not testing.fake_handle().stats


_PASSED = {
    "stats": TeaCacheStats(),
    "provenance": Provenance(source="builtin", revision="r1"),
    "rel_l1_thresh": 0.33,
    "variant_id": "flux1-dev",
    "coefficients": (1.0, -2.0, 3.0, -4.0, 5.0),
    "decision_rule_revision": 7,
    "mflux_version": "0.24.0",
}


@pytest.mark.parametrize("name", sorted(_PASSED))
def test_fake_handle_passes_each_value_through(name: str) -> None:
    """Bug: a keyword is ignored and the property keeps its default."""
    value = _PASSED[name]
    h = _testing().fake_handle(**{name: value})
    assert getattr(h, name) == value
    if name in ("stats", "provenance"):
        assert getattr(h, name) is value


def test_fake_handle_coefficients_are_coerced_to_a_float_tuple() -> None:
    """Bug: the caller's list is stored as is, so the handle reports a list (or ints) where a real one reports a
    5-float tuple."""
    h = _testing().fake_handle(coefficients=[1, 2, 3, 4, 5])
    assert h.coefficients == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert type(h.coefficients) is tuple
    assert all(type(c) is float for c in h.coefficients)


def test_fake_handle_is_keyword_only() -> None:
    """Bug: a positional parameter, so a caller's argument order silently binds to the wrong fact."""
    with pytest.raises(TypeError):
        _testing().fake_handle(TeaCacheStats())  # type: ignore[misc]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_fake_handle_rejects_non_finite_coefficients(bad: float) -> None:
    """Bug: a NaN or infinite tuple accepted, which apply_teacache would refuse."""
    with pytest.raises(TeaCacheValueError):
        _testing().fake_handle(coefficients=(0.0, 0.0, bad, 0.0, 0.0))


def test_fake_handle_rejects_a_wrong_length() -> None:
    """Bug: coefficients stored without validation."""
    with pytest.raises(TeaCacheValueError):
        _testing().fake_handle(coefficients=(0.0, 0.0, 0.0, 0.0))


def test_fake_handle_restore_is_idempotent_and_freezes_stats() -> None:
    """Bug: the second restore raises, or the stats stay writable after restore."""
    h = _testing().fake_handle()
    h.restore()
    h.restore()
    with pytest.raises(StatsFrozenError):
        h.stats.record(
            StepDecision(step_idx=0, timestep=1.0, rel_l1=None, accumulated_distance=0.0, decision="forced")
        )


def test_fake_handle_works_as_a_context_manager() -> None:
    """Bug: the fake's patch cannot be torn down, so `with fake_handle() as h:` raises on exit."""
    with _testing().fake_handle() as h:
        assert h.variant_id == "fake"
    with pytest.raises(StatsFrozenError):
        h.stats.record(
            StepDecision(step_idx=0, timestep=1.0, rel_l1=None, accumulated_distance=0.0, decision="forced")
        )


def test_fake_handle_never_warns() -> None:
    """Bug: the fake goes through the deprecated setters, so a warnings-as-errors suite fails on building it."""
    testing = _testing()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h = testing.fake_handle(
            stats=TeaCacheStats(),
            provenance=Provenance(source="builtin"),
            rel_l1_thresh=0.2,
            variant_id="x",
            coefficients=(1.0, 2.0, 3.0, 4.0, 5.0),
            decision_rule_revision=1,
            mflux_version="0.24.0",
        )
        h.restore()


def test_reset_rearms_the_untested_mflux_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug: reset is a no-op or clears the wrong flag, so a test that needs the once-per-process warning sees
    nothing."""
    monkeypatch.setattr(versions, "installed_mflux_version", lambda: NEWER)
    monkeypatch.setattr(versions, "_warned", True)  # as every test starts (conftest)
    reset = _testing().reset_untested_mflux_warning

    def _calls() -> int:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            versions.warn_if_untested_mflux()
        return len([w for w in caught if issubclass(w.category, TeaCacheUntestedMfluxWarning)])

    assert _calls() == 0
    assert reset() is None
    assert _calls() == 1
    assert _calls() == 0
    reset()
    assert _calls() == 1


def test_public_names() -> None:
    """Bug: a helper drops out of __all__ (star-imports and documentation tools stop seeing it), or a private
    name is exported."""
    assert sorted(_testing().__all__) == ["fake_handle", "reset_untested_mflux_warning"]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)


def test_testing_imports_without_mflux_or_pytest() -> None:
    """Bug: an mflux or pytest import sneaks in, breaking installs without the [mflux] extra and code that uses
    the helpers outside pytest."""
    result = _run(
        "import sys\n"
        "sys.modules['mflux'] = None\n"
        "sys.modules['pytest'] = None\n"
        "sys.modules['_pytest'] = None\n"
        "import mlx_teacache.testing as t\n"
        "h = t.fake_handle()\n"
        "h.restore()\n"
        "t.reset_untested_mflux_warning()\n"
        "print(sorted(t.__all__))\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "['fake_handle', 'reset_untested_mflux_warning']"


def test_package_root_does_not_import_testing() -> None:
    """Bug: the package root imports the helpers, so every user pays for them and the on-demand rule breaks."""
    result = _run(
        "import sys\n"
        "import mlx_teacache\n"
        "from mlx_teacache import apply_teacache\n"
        "assert 'mlx_teacache.testing' not in sys.modules, 'imported by the root'\n"
        "print('OK')\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"
