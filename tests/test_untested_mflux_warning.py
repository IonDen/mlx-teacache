"""apply_teacache warns once per process, and still patches, when the installed mflux is newer than the newest
release this version was verified on."""

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
from mlx_teacache import TeaCacheUntestedMfluxWarning, apply_teacache
from tests._fakes import FaithfulCallbackRegistry

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


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
    installed_mflux("0.22.0")
    caught, variant_ids = _apply(times=2)
    assert len(caught) == 1
    assert variant_ids == ["flux1-dev", "flux1-dev"]
    message = str(caught[0].message)
    assert "mflux 0.22.0" in message
    assert versions.NEWEST_VERIFIED_MFLUX in message
    assert "https://github.com/IonDen/mlx-teacache/issues" in message


@pytest.mark.parametrize("installed", ["0.21.0", "0.20.0", "0.17.5"])
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
    installed_mflux("0.22.0")
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
    installed_mflux("0.22.0")
    caught, _ = _apply()
    pyproject = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
    filters = pyproject["tool"]["pytest"]["ini_options"]["filterwarnings"]
    entries = [entry for entry in filters if entry.startswith("ignore:mflux")]
    assert len(entries) == 1
    _action, message, category = entries[0].split(":")
    assert re.match(message, str(caught[0].message), re.IGNORECASE)
    assert category == "UserWarning" and issubclass(TeaCacheUntestedMfluxWarning, UserWarning)
