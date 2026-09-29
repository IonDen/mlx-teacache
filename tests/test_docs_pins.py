"""Doc-accuracy regression guard: every install pin in the public docs must
always match the CHANGELOG's current top release, so they can never silently
drift apart across a version bump. Pure-core: reads files only."""

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
_PIN = r"mlx-teacache\[mflux\]==(\d+\.\d+\.\d+)"


def _changelog_latest() -> str:
    changelog = (_REPO_ROOT / "CHANGELOG.md").read_text()
    latest_match = re.search(r"^## \[(\d+\.\d+\.\d+)\]", changelog, flags=re.M)
    assert latest_match is not None, "CHANGELOG.md has no versioned '## [x.y.z]' entry"
    return latest_match.group(1)


@pytest.mark.parametrize("doc_path", ["README.md", "docs/manual-verification.md"])
def test_install_pin_matches_the_changelog_top_release(doc_path: str) -> None:
    """Bug: a release bumps the CHANGELOG but a doc keeps telling users to install the previous version."""
    pins = set(re.findall(_PIN, (_REPO_ROOT / doc_path).read_text()))
    latest = _changelog_latest()
    assert pins == {latest}, f"{doc_path} pins {pins}, CHANGELOG latest is {latest}"
