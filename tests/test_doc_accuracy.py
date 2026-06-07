"""Doc-accuracy regression guards (backlog 0030). Pure-core: reads files only."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The FLUX.1-dev headline was reconciled to 1.46x (0017's committed bench
# _artifacts/v0.6.3_bench_flux1_dev.json); no user-facing doc may still claim the
# stale 1.48x. Scanned targets are the user-trusted surfaces only -- CHANGELOG.md
# is intentionally excluded (its 1.48x mentions are archival history + a
# "removed 1.48x" note). This file holds the needles, so it is NOT scanned.
# If a different metric ever legitimately reads 1.48x, re-scope this guard.
_STALE_1_48X_TARGETS = (
    "docs/variants/flux1-dev.md",
    "README.md",
    "tests/test_image_quality_flux1.py",
)


def test_no_stale_1_48x_speedup_claim():
    offenders = []
    for rel in _STALE_1_48X_TARGETS:
        text = (_REPO_ROOT / rel).read_text()
        if "1.48x" in text or "1.48×" in text:  # ASCII 'x' and Unicode U+00D7
            offenders.append(rel)
    assert not offenders, f"stale 1.48x speedup claim still present in: {offenders}"


def test_calibration_doc_does_not_route_edits_to_dead_coefficients_shim():
    # coefficients.py is a dead re-export shim (v0.6.0); the doc must not tell
    # users to edit it. Live tuples are in variants/<id>/config.py, provenance in
    # each integration.py. The shim is mentioned once (the explanation), which
    # does not use these action-phrasings.
    text = (_REPO_ROOT / "docs" / "calibration.md").read_text()
    for forbidden in ("into `coefficients.py`", "in coefficients.py"):
        assert forbidden not in text, f"calibration.md still routes edits to the shim: {forbidden!r}"
