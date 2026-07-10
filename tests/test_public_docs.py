"""Regression guards for public-doc hygiene (no mflux needed -> pure-core lane)."""

import re
from pathlib import Path

from packaging.requirements import Requirement

_REPO = Path(__file__).resolve().parent.parent
_PUBLIC_DOCS = [
    "README.md",
    "CHANGELOG.md",
    "COMPARISON.md",
    "ROADMAP.md",
    "docs/variants/flux2-klein-base-4b.md",
    "docs/variants/flux2-klein-base-9b.md",
    "docs/variants/z-image-base.md",
    "docs/variants/qwen-image.md",
]


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


def test_no_public_doc_cites_gitignored_artifacts():
    offenders = []
    for relative_path in _PUBLIC_DOCS:
        path = _REPO / relative_path
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if "tests/_artifacts/" in line:
                offenders.append(f"{relative_path}:{line_number}")
    assert not offenders, f"public docs cite gitignored tests/_artifacts/: {offenders}"
