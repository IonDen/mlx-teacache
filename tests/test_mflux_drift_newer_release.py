"""The real drift test on an mflux release newer than every recorded row: it passes while no fingerprinted function
moved and fails as soon as one did. Uses the installed mflux's real fingerprints; only the version is pretended."""

from importlib.metadata import version

import pytest

import tests.test_mflux_forward_drift as drift
from mlx_teacache._mflux_versions import NEWEST_VERIFIED_MFLUX
from tests._mflux_surface import installed_vcs_commit

pytestmark = pytest.mark.skipif(
    installed_vcs_commit("mflux") is not None,
    reason="mflux is a git install; these tests are about a released wheel",
)


@pytest.fixture
def pretend_mflux_0_22(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drift, "version", lambda name: "0.22.0")


def test_the_real_drift_test_passes_a_newer_release_when_nothing_moved(pretend_mflux_0_22) -> None:
    """Bug: the shipped drift test still fails every unrecorded release (the pre-0.13 rule), or holds a newer release
    to a row other than the newest, so a harmless mflux 0.22 turns every CI lane red."""
    if version("mflux") != NEWEST_VERIFIED_MFLUX:
        pytest.skip(
            f"the installed mflux must be {NEWEST_VERIFIED_MFLUX}, the newest recorded row (the lock lane)"
        )
    drift.test_installed_mflux_forwards_match_their_verified_fingerprints()


def test_the_real_drift_test_fails_a_newer_release_when_one_function_moved(
    pretend_mflux_0_22, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: the shipped drift test waves an unrecorded newer release through on its version number without comparing
    the installed fingerprints."""
    moved = drift._installed_fingerprints()
    moved["z_image.ZImage.generate_image"] = "0000000000000000"
    monkeypatch.setattr(drift, "_installed_fingerprints", lambda: moved)
    with pytest.raises(AssertionError, match=r"z_image\.ZImage\.generate_image changed in mflux 0\.22\.0"):
        drift.test_installed_mflux_forwards_match_their_verified_fingerprints()
