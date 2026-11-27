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
    "docs/stability.md",
    "docs/variants/flux2-klein-base-4b.md",
    "docs/variants/flux2-klein-base-9b.md",
    "docs/variants/z-image-base.md",
    "docs/variants/qwen-image.md",
    "docs/variants/qwen-image-2512.md",
    "docs/comparison/flux1-dev.md",
    "docs/comparison/flux1-krea-dev.md",
    "docs/comparison/klein-base-4b.md",
    "docs/comparison/z-image-base.md",
    "docs/comparison/klein-base-9b.md",
    "docs/comparison/klein-distilled-vs-base.md",
    "docs/comparison/qwen-image.md",
    "docs/benchmarks.md",
    "docs/limitations.md",
    "docs/how-it-works.md",
]
# The pages that took the README's record-bound and figure-free detail.
_NEW_PAGES = ("docs/benchmarks.md", "docs/limitations.md", "docs/how-it-works.md")
# A Markdown link target, as scripts/sdist_smoke.sh reads it.
_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)")
_ABSOLUTE = ("https://", "http://", "mailto:", "#")
# A README link into this repository's files on GitHub: the path, then the fragment if any.
_REPO_URL = re.compile(
    r"https://github\.com/IonDen/mlx-teacache/(?:blob|tree)/main/([^\s)#>\"']+)(?:#([^\s)>\"']+))?"
)
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")


def _heading_slugs(text: str) -> set[str]:
    """The anchors GitHub gives the ATX headings of a Markdown text: lowercase, everything but letters, digits,
    spaces, hyphens and underscores dropped, spaces to hyphens, and "-1", "-2" on repeats. Lines inside fenced code
    blocks are not headings."""
    slugs: set[str] = set()
    seen: dict[str, int] = {}
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        match = None if in_fence else _HEADING.match(line)
        if match is None:
            continue
        slug = re.sub(r"[^\w\- ]", "", match.group(1).lower()).replace(" ", "-")
        repeat = seen.get(slug, 0)
        seen[slug] = repeat + 1
        slugs.add(f"{slug}-{repeat}" if repeat else slug)
    return slugs


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


def test_all_public_docs_exist():
    """Bug: a listed page is renamed or missing, so the hygiene checks silently skip it."""
    missing = [path for path in _PUBLIC_DOCS if not (_REPO / path).exists()]
    assert not missing, f"_PUBLIC_DOCS lists non-existent files: {missing}"


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


def test_mflux_files_allowlist_all_exist():
    from tests._lanes import MFLUX_FILES

    tests_dir = _REPO / "tests"
    missing = sorted(rel_path for rel_path in MFLUX_FILES if not (tests_dir / rel_path).exists())
    assert not missing, f"MFLUX_FILES lists non-existent files: {missing}"


def test_public_docstring_summaries_name_all_supported_model_families():
    import mlx_teacache
    from mlx_teacache import apply_teacache

    summaries = [
        (mlx_teacache.__doc__ or "").splitlines()[0].lower(),
        (apply_teacache.__doc__ or "").splitlines()[0].lower(),
    ]
    for summary in summaries:
        for family in ("flux", "qwen-image", "z-image"):
            assert family in summary, f"{family} missing from public summary: {summary!r}"


def test_public_docs_cite_only_committed_report_paths():
    """Bug: a bench or calibration report moves (scripts/ -> _artifacts/) and a public page keeps
    citing the old path, a dead reference for every reader. CHANGELOG is exempt: its entries record
    the paths as they were at that release."""
    offenders = []
    for relative_path in _PUBLIC_DOCS:
        if relative_path == "CHANGELOG.md":
            continue
        text = (_REPO / relative_path).read_text()
        for cited in re.findall(r"`((?:scripts|_artifacts)/[^`*\s]+\.json)`", text):
            if not (_REPO / cited).exists():
                offenders.append(f"{relative_path}: {cited}")
    assert not offenders, f"public docs cite report paths that do not exist: {offenders}"


def test_readme_links_are_absolute_for_pypi():
    """Bug: a relative link such as `docs/stability.md` in README.md, which PyPI renders as a dead link on the
    project page. In-page `#anchors` are allowed."""
    relative = [
        f"README.md:{line_number}: {target}"
        for line_number, line in enumerate((_REPO / "README.md").read_text().splitlines(), 1)
        for target in _LINK.findall(line)
        if not target.startswith(_ABSOLUTE)
    ]
    assert not relative, f"README links that are not absolute: {relative}"


def test_new_pages_relative_links_resolve():
    """Bug: a link moved out of the README keeps its README-relative target (`docs/variants/...` from inside
    `docs/`) or names a page that does not exist, so it is dead on GitHub. Fragments are not checked."""
    dead = []
    for page in _NEW_PAGES:
        path = _REPO / page
        assert path.is_file(), f"{page} does not exist"
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            for target in _LINK.findall(line):
                if target.startswith(_ABSOLUTE):
                    continue
                file_part = target.split("#", 1)[0]
                if file_part and not (path.parent / file_part).exists():
                    dead.append(f"{page}:{line_number}: {target}")
    assert not dead, f"relative links whose target does not exist: {dead}"


def test_heading_slugs_follow_github():
    """Bug: the slug rule drifts from GitHub's (keeps the "." of "FLUX.1", or squeezes the "--" left by " + "), a
    "# comment" inside a fenced code block counts as a heading, or a repeated heading misses GitHub's "-1" suffix.
    Each lets a dead anchor pass, or fails a live one, in the two README link tests below."""
    text = (
        "# mlx-teacache\n\n## Threshold sweep for FLUX.1 dev\n\n## License + acknowledgements\n\n"
        "```sh\n# or with uv:\n```\n\n## Notes\n\n### Notes\n"
    )
    assert _heading_slugs(text) == {
        "mlx-teacache",
        "threshold-sweep-for-flux1-dev",
        "license--acknowledgements",
        "notes",
        "notes-1",
    }


def test_readme_repo_links_resolve():
    """Bug: a README link into this repository names a path that does not exist (`docs/benchmark.md`), or deep-links
    a heading that was renamed (`## Performance by chip` on docs/benchmarks.md). The README is the PyPI page, so a dead
    link there stays dead until the next release."""
    links = _REPO_URL.findall((_REPO / "README.md").read_text())
    assert links, "no README link into the repository found; the URL pattern no longer matches the README"
    dead = []
    for path, fragment in links:
        target = _REPO / path
        if not target.exists():
            dead.append(f"{path}: no such path")
        elif fragment and path.endswith(".md") and fragment not in _heading_slugs(target.read_text()):
            dead.append(f"{path}#{fragment}: no such heading")
    assert not dead, f"README links into the repository that are dead: {dead}"


def test_readme_in_page_anchors_resolve():
    """Bug: a README heading is renamed (`## License obligations`) while an in-page link still points at its old
    anchor (`#license-obligations`), which then goes nowhere on GitHub and PyPI."""
    readme = (_REPO / "README.md").read_text()
    anchors = [target for target in _LINK.findall(readme) if target.startswith("#")]
    assert anchors, "no in-page README anchor found"
    slugs = _heading_slugs(readme)
    missing = [anchor for anchor in anchors if anchor[1:] not in slugs]
    assert not missing, f"README in-page anchors with no heading: {missing}"
