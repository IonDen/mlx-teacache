"""Regression guards for public-doc hygiene (no mflux needed -> pure-core lane)."""

import re
from pathlib import Path

from packaging.requirements import Requirement

_REPO = Path(__file__).resolve().parent.parent


def _pip_install_targets(text: str) -> list[str]:
    """Extract the requirement token from every `pip install "<tok>"` in a doc."""
    targets: list[str] = []
    for match in re.finditer(r'pip install\s+"([^"]+)"', text):
        token = match.group(1)
        if token.startswith("-") or "://" in token:
            continue
        targets.append(token)
    return targets


def test_readme_install_commands_parse():
    text = (_REPO / "README.md").read_text()
    targets = _pip_install_targets(text)
    assert targets, "expected at least one pip install command in README"
    for token in targets:
        Requirement(token)
