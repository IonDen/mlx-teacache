"""apply_teacache warns once per process, and still patches, when the installed mflux is newer than the newest
release this version was verified on."""

import importlib.metadata
import importlib.util
import inspect
import re
import sys
import warnings
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace

import pytest

import mlx_teacache._mflux_versions as versions
from mlx_teacache import IncompatibleModelError, TeaCacheUntestedMfluxWarning, apply_teacache
from tests._fakes import FaithfulCallbackRegistry

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


# The next minor release after the newest verified one: newer than verified, still inside the [mflux] range.
_VERIFIED = versions.release_tuple(versions.NEWEST_VERIFIED_MFLUX)
assert _VERIFIED is not None
_MAJOR, _MINOR, *_ = _VERIFIED
NEWER = f"{_MAJOR}.{_MINOR + 1}.0"


def _fake_flux1_dev() -> SimpleNamespace:
    """Duck-typed FLUX.1 dev that apply() patches without mflux or weights (same shape as test_api_warnings)."""
    return SimpleNamespace(
        model_config=SimpleNamespace(aliases=["dev"]),
        transformer=SimpleNamespace(),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
    )


@pytest.fixture
def installed_mflux(monkeypatch: pytest.MonkeyPatch) -> Callable[[str | None], None]:
    """Pretend an mflux version is installed (None: not installed), with the once-per-process flag reset (the
    autouse conftest fixture set it to True; autouse fixtures run first, so this reset wins)."""
    monkeypatch.setattr(versions, "_warned", False)

    def _set(value: str | None) -> None:
        monkeypatch.setattr(versions, "installed_mflux_version", lambda: value)

    return _set


def _apply(times: int = 1) -> tuple[list[warnings.WarningMessage], list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        variant_ids = []
        for _ in range(times):
            handle = apply_teacache(_fake_flux1_dev())
            variant_ids.append(handle.variant_id)
            handle.restore()
    return [w for w in caught if issubclass(w.category, TeaCacheUntestedMfluxWarning)], variant_ids


def test_newer_mflux_warns_once_per_process_and_still_patches(installed_mflux) -> None:
    """Bug: the once-per-process flag is never set (a warning on every apply), or a newer mflux is refused."""
    installed_mflux(NEWER)
    caught, variant_ids = _apply(times=2)
    assert len(caught) == 1
    assert variant_ids == ["flux1-dev", "flux1-dev"]
    message = str(caught[0].message)
    assert f"mflux {NEWER}" in message
    assert versions.NEWEST_VERIFIED_MFLUX in message
    assert "https://github.com/IonDen/mlx-teacache/issues" in message


@pytest.mark.parametrize("installed", [versions.NEWEST_VERIFIED_MFLUX, "0.20.0", "0.17.5"])
def test_verified_or_older_mflux_does_not_warn(installed_mflux, installed: str) -> None:
    """Bug: the comparison is >= instead of > (the verified release itself warns), or older releases warn."""
    installed_mflux(installed)
    caught, _ = _apply()
    assert caught == []


def test_without_mflux_installed_apply_neither_warns_nor_fails(installed_mflux) -> None:
    """Bug: on an install without mflux the version check raises or warns before a duck-typed model is patched."""
    installed_mflux(None)
    caught, variant_ids = _apply()
    assert caught == [] and variant_ids == ["flux1-dev"]


def test_installed_mflux_version_is_none_when_the_package_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug: PackageNotFoundError from the metadata lookup escapes apply_teacache on a pure-core install."""

    def _missing(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(versions, "version", _missing)
    assert versions.installed_mflux_version() is None


def test_warning_points_at_the_callers_line(installed_mflux) -> None:
    """Bug: a wrong stacklevel attributes the warning to mlx_teacache internals (api.py or _mflux_versions.py), so a
    caller's module- or line-scoped warnings filter cannot target it."""
    installed_mflux(NEWER)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call_line = inspect.currentframe().f_lineno + 1
        handle = apply_teacache(_fake_flux1_dev())
    handle.restore()
    (warning,) = [w for w in caught if issubclass(w.category, TeaCacheUntestedMfluxWarning)]
    assert Path(warning.filename).resolve() == Path(__file__).resolve()
    assert warning.lineno == call_line


def test_every_other_test_starts_with_the_warning_already_shown() -> None:
    """Bug: the autouse conftest fixture is dropped, so on a newer mflux (the newest-in-range and mflux-main CI jobs)
    the once-per-process warning fires inside whichever unrelated test applies first and breaks its warning
    assertions depending on test order."""
    assert versions._warned is True


def test_pytest_ignores_exactly_this_warning(installed_mflux) -> None:
    """Bug: the warning is reworded (or the filter edited) so pytest's ignore entry stops matching, and every apply
    test fails under filterwarnings=error in the CI jobs that install a newer mflux."""
    installed_mflux(NEWER)
    caught, _ = _apply()
    pyproject = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
    filters = pyproject["tool"]["pytest"]["ini_options"]["filterwarnings"]
    entries = [entry for entry in filters if entry.startswith("ignore:mflux")]
    assert len(entries) == 1
    _action, message, category = entries[0].split(":")
    assert re.match(message, str(caught[0].message), re.IGNORECASE)
    assert category == "UserWarning" and issubclass(TeaCacheUntestedMfluxWarning, UserWarning)


def test_installed_mflux_version_reads_the_real_environment() -> None:
    """Bug: the lookup mishandles the real PackageNotFoundError on an install without mflux (it escapes, or it
    reports a version for an absent package), or it reports a version other than the installed one. Unpatched. It
    only discriminates in the pure-core lane, where mflux is not installed and the absent-package path runs; with
    mflux installed, the expected value is read the same way as the code under test does."""
    expected = None if importlib.util.find_spec("mflux") is None else importlib.metadata.version("mflux")
    assert versions.installed_mflux_version() == expected


def test_unmatched_model_raises_without_spending_the_untested_warning(installed_mflux) -> None:
    """Bug: warn_if_untested_mflux() moves above the registry loop, so a model apply_teacache refuses still emits
    the untested-mflux warning and uses up the once-per-process flag before any model is actually patched."""
    installed_mflux(NEWER)
    turbo = SimpleNamespace(
        model_config=SimpleNamespace(aliases=["z-image-turbo"]),
        transformer=SimpleNamespace(),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(IncompatibleModelError):
            apply_teacache(turbo)
    assert [w for w in caught if issubclass(w.category, TeaCacheUntestedMfluxWarning)] == []
    # The refusal did not spend the flag: the next apply on a matching model still warns, once.
    matched, variant_ids = _apply()
    assert variant_ids == ["flux1-dev"]
    assert len(matched) == 1


def test_zero_cond_t_qwen_model_raises_without_spending_the_untested_warning(installed_mflux) -> None:
    """Bug: the zero_cond_t refusal sits in the Qwen integration's apply() instead of detection, so apply_teacache
    emits the untested-mflux warning (and uses up the once-per-process flag) for a Qwen-Image-Edit-2511 transformer
    it then refuses."""
    installed_mflux(NEWER)
    edit_2511 = SimpleNamespace(
        model_config=SimpleNamespace(aliases=["qwen-image", "qwen"], model_name=None),
        transformer=SimpleNamespace(zero_cond_t=True),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(IncompatibleModelError):
            apply_teacache(edit_2511)
    assert [w for w in caught if issubclass(w.category, TeaCacheUntestedMfluxWarning)] == []
    matched, variant_ids = _apply()
    assert variant_ids == ["flux1-dev"]
    assert len(matched) == 1


def test_a_raising_warnings_filter_does_not_silence_later_calls(installed_mflux) -> None:
    """Bug: the once-per-process flag is set before warnings.warn runs, so a caller's "error" filter that makes the
    first call raise also silences every later call, where the warning would otherwise reach a caller who relaxed
    the filter."""
    installed_mflux(NEWER)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2):
            with pytest.raises(TeaCacheUntestedMfluxWarning):
                versions.warn_if_untested_mflux()
    caught, _ = _apply()
    assert len(caught) == 1
    caught_again, _ = _apply()
    assert caught_again == []
