"""Doc-accuracy regression guard: the manual-verification install pin must
always match the CHANGELOG's current top release, so the two can never
silently drift apart across a version bump. Pure-core: reads files only."""

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent


def test_manual_verification_pins_the_current_release():
    doc = (_REPO_ROOT / "docs" / "manual-verification.md").read_text()
    changelog = (_REPO_ROOT / "CHANGELOG.md").read_text()
    doc_pins = set(re.findall(r"mlx-teacache\[mflux\]==(\d+\.\d+\.\d+)", doc))
    latest_match = re.search(r"^## \[(\d+\.\d+\.\d+)\]", changelog, flags=re.M)
    assert latest_match is not None, "CHANGELOG.md has no versioned '## [x.y.z]' entry"
    latest = latest_match.group(1)
    assert doc_pins == {latest}, f"manual-verification pins {doc_pins}, CHANGELOG latest is {latest}"
