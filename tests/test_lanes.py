"""Lane classification is keyed on the path relative to tests/, not the basename."""

import pytest

from tests._lanes import MFLUX_FILES, is_mflux_file


def test_root_level_name_is_mflux() -> None:
    """Bug caught: an allowlisted root-level file drops out of the mflux lane."""
    assert is_mflux_file("test_detect.py") is True


def test_same_basename_under_variants_is_not_mflux() -> None:
    """Bug caught: a basename match marks tests/variants/*/test_detect.py as mflux."""
    assert is_mflux_file("variants/flux1_dev/test_detect.py") is False


def test_explicit_relative_path_entry_is_mflux() -> None:
    """Bug caught: a variants/ file that imports mflux is not kept out of the pure-core lane."""
    assert is_mflux_file("variants/flux1_dev/test_integration_smoke.py") is True


def test_unlisted_file_is_not_mflux() -> None:
    """Bug caught: the helper answers True for files that are not on the allowlist."""
    assert is_mflux_file("test_gate.py") is False


def test_allowlist_has_no_duplicates() -> None:
    """Bug caught: a duplicated allowlist entry (a silent typo in a set literal)."""
    with pytest.raises(ValueError, match="duplicate"):
        from tests._lanes import build_allowlist

        build_allowlist(["a.py", "b.py", "a.py"])
    assert len(MFLUX_FILES) == 18
