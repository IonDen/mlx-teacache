"""Judging an mflux git install: a checkout of mflux main still reports the last release's version, so the version
rule cannot tell it from the release. It is held to the reviewed-main row instead. All of this runs without mflux."""

import json
from pathlib import Path

import pytest

from tests._mflux_surface import drift_failures_for_install, installed_vcs_commit
from tests.test_mflux_forward_drift import KNOWN, REVIEWED_MAIN

_KNOWN = {"0.21.0": {"f": "old", "g": "same"}}
_MAIN = {"commit": "04fc1011f19c3e8be43145c0baf2972478ebdc0f", "fingerprints": {"f": "new", "g": "same"}}


def _dist_info(root: Path, direct_url: object | None) -> None:
    info = root / "fakemflux-0.21.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fakemflux\nVersion: 0.21.0\n")
    if direct_url is not None:
        (info / "direct_url.json").write_text(json.dumps(direct_url))


def test_a_git_install_reports_its_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug: a git install of mflux is not recognised, so main is held to the release row and stays red."""
    _dist_info(tmp_path, {"url": "https://x/y.git", "vcs_info": {"vcs": "git", "commit_id": "abc123"}})
    monkeypatch.syspath_prepend(str(tmp_path))
    assert installed_vcs_commit("fakemflux") == "abc123"


def test_a_wheel_install_has_no_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug: a wheel install is mistaken for a git install and judged against the main row."""
    _dist_info(tmp_path, None)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert installed_vcs_commit("fakemflux") is None


def test_a_direct_url_without_vcs_info_has_no_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug: a local-path or archive install (direct_url.json without vcs_info) crashes or counts as git."""
    _dist_info(tmp_path, {"url": "file:///x", "dir_info": {}})
    monkeypatch.syspath_prepend(str(tmp_path))
    assert installed_vcs_commit("fakemflux") is None


def test_a_non_git_vcs_has_no_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug: a hg/svn install with a commit_id is judged against the git main row."""
    _dist_info(tmp_path, {"url": "https://x/y", "vcs_info": {"vcs": "hg", "commit_id": "abc123"}})
    monkeypatch.syspath_prepend(str(tmp_path))
    assert installed_vcs_commit("fakemflux") is None


def test_a_direct_url_that_is_not_json_has_no_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug: a corrupt direct_url.json crashes the drift test instead of counting as no git commit."""
    info = tmp_path / "fakemflux-0.21.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: fakemflux\nVersion: 0.21.0\n")
    (info / "direct_url.json").write_text("{not json")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert installed_vcs_commit("fakemflux") is None


def test_a_vcs_info_that_is_not_a_mapping_has_no_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: a malformed vcs_info (a string) raises AttributeError instead of reporting no commit."""
    _dist_info(tmp_path, {"vcs_info": "x"})
    monkeypatch.syspath_prepend(str(tmp_path))
    assert installed_vcs_commit("fakemflux") is None


def test_a_missing_distribution_has_no_commit() -> None:
    """Bug: an uninstalled package raises instead of reporting no commit."""
    assert installed_vcs_commit("definitely-not-installed-teacache-0146") is None


def test_a_git_install_is_judged_by_the_main_row_not_its_version() -> None:
    """Bug: main reporting 0.21.0 is held to the 0.21.0 row, so the intended Z-Image change is red forever."""
    assert drift_failures_for_install("0.21.0", "abc", {"f": "new", "g": "same"}, _KNOWN, _MAIN) == []


def test_a_wheel_is_judged_by_the_version_rule() -> None:
    """Bug: a wheel is judged against the main row, so the released 0.21.0 goes red."""
    assert drift_failures_for_install("0.21.0", None, {"f": "old", "g": "same"}, _KNOWN, _MAIN) == []


def test_main_failure_names_the_installed_and_the_reviewed_commit() -> None:
    """Bug: a function that moved on main since review is not reported, or is reported without saying which commit
    is installed and which was reviewed."""
    installed = "1234567890abcdef"
    failures = drift_failures_for_install("0.21.0", installed, {"f": "new", "g": "moved"}, _KNOWN, _MAIN)
    assert len(failures) == 1
    assert "g changed on mflux main" in failures[0]
    assert "installed 1234567, reviewed 04fc101" in failures[0]


def test_a_wheel_newer_than_every_row_passes_when_it_carries_the_reviewed_main_change() -> None:
    """Bug: mflux 0.22.0 is tagged with the already-reviewed main change, and the blocking newest-in-range job
    goes red because only the newest release row is consulted."""
    assert drift_failures_for_install("0.22.0", None, {"f": "new", "g": "same"}, _KNOWN, _MAIN) == []


def test_a_wheel_newer_than_every_row_passes_when_it_matches_the_newest_row() -> None:
    """Bug: the main-row fallback replaces the newest-row comparison instead of adding to it."""
    assert drift_failures_for_install("0.22.0", None, {"f": "old", "g": "same"}, _KNOWN, _MAIN) == []


def test_a_wheel_newer_than_every_row_fails_naming_both_references() -> None:
    """Bug: a newer wheel matching neither reference is waved through, or reported against only one of them."""
    failures = drift_failures_for_install("0.22.0", None, {"f": "third", "g": "same"}, _KNOWN, _MAIN)
    text = "\n".join(failures)
    assert failures
    assert "verified 0.21.0" in text
    assert "reviewed main 04fc101" in text


def test_a_recorded_release_that_differs_from_its_row_fails_even_if_it_matches_main() -> None:
    """Bug: the main-row escape hatch is applied to a recorded release, so a changed 0.21.0 wheel passes."""
    failures = drift_failures_for_install("0.21.0", None, {"f": "new", "g": "same"}, _KNOWN, _MAIN)
    assert len(failures) == 1 and "f changed in mflux 0.21.0" in failures[0]


def test_an_unrecorded_release_inside_the_span_fails_even_if_it_matches_main() -> None:
    """Bug: the main-row escape hatch is applied to an unverified in-span release."""
    known = {"0.21.0": {"f": "old", "g": "same"}, "0.19.0": {"f": "old", "g": "same"}}
    failures = drift_failures_for_install("0.20.0", None, {"f": "new", "g": "same"}, known, _MAIN)
    assert len(failures) == 1 and "0.20.0 has no verified row" in failures[0]


def test_reviewed_main_has_the_same_labels_as_every_release_row() -> None:
    """Bug: a target added to one table and not the other, so main is silently held to fewer functions."""
    labels = set(REVIEWED_MAIN["fingerprints"])
    assert all(set(row) == labels for row in KNOWN.values())
