"""The real drift test on an mflux release newer than every recorded row: it passes while no fingerprinted function
moved and fails as soon as one did. Uses the installed mflux's real fingerprints; only the version is pretended."""

import re
from importlib.metadata import version

import pytest

import tests.test_mflux_forward_drift as drift
from mlx_teacache._mflux_versions import NEWEST_VERIFIED_MFLUX, release_tuple
from tests._mflux_surface import installed_vcs_commit

pytestmark = pytest.mark.skipif(
    installed_vcs_commit("mflux") is not None,
    reason="mflux is a git install; these tests are about a released wheel",
)


def _next_minor_after_every_row() -> str:
    """The next minor release after the newest KNOWN row, so the pretended version stays newer as rows are added."""
    newest = max(release_tuple(row) or (0, 0, 0) for row in drift.KNOWN)
    return f"{newest[0]}.{newest[1] + 1}.0"


@pytest.fixture
def pretend_newer_release(monkeypatch: pytest.MonkeyPatch) -> str:
    newer = _next_minor_after_every_row()
    monkeypatch.setattr(drift, "version", lambda name: newer)
    return newer


def test_the_real_drift_test_passes_a_newer_release_when_nothing_moved(pretend_newer_release: str) -> None:
    """Bug: the shipped drift test still fails every unrecorded release (the pre-0.13 rule), or holds a newer release
    to a row other than the newest, so a harmless newer mflux release turns every CI lane red."""
    if version("mflux") != NEWEST_VERIFIED_MFLUX:
        pytest.skip(
            f"the installed mflux must be {NEWEST_VERIFIED_MFLUX}, the newest recorded row (the lock lane)"
        )
    drift.test_installed_mflux_forwards_match_their_verified_fingerprints()


def test_the_real_drift_test_fails_a_newer_release_when_one_function_moved(
    pretend_newer_release: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: the shipped drift test waves an unrecorded newer release through on its version number without comparing
    the installed fingerprints."""
    moved = drift._installed_fingerprints()
    moved["z_image.ZImage.generate_image"] = "0000000000000000"
    monkeypatch.setattr(drift, "_installed_fingerprints", lambda: moved)
    expected = rf"z_image\.ZImage\.generate_image changed in mflux {re.escape(pretend_newer_release)}"
    with pytest.raises(AssertionError, match=expected):
        drift.test_installed_mflux_forwards_match_their_verified_fingerprints()
