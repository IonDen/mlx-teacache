"""Comparison page generator: numbers come from the report, prose stays hand-written (pure-core lane)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _comparison_recipes as cr  # noqa: E402
import _generate_comparison as gen  # noqa: E402
import bench_comparison as bc  # noqa: E402

GIB = 1024**3


def _result(cond: str, **over: object) -> dict:
    a = cond == "a"
    base = {
        "condition": cond,
        "width": 768,
        "height": 1024,
        "load_seconds": 21.0 if a else 22.0,
        "encode_seconds": 2.1 if a else 2.2,
        "generation_seconds": 239.14 if a else 198.9,
        "compute_seconds": [30.0] + [9.0] * 24 if a else [30.0] + [9.5, 1.2] * 12,
        "preview_seconds": [0.24] * 25 if a else [0.26] * 25,
        "mlx_peak_load_bytes": 9 * GIB,
        "mlx_peak_encode_bytes": int(9.5 * GIB),
        "mlx_peak_generation_bytes": int((10.4 if a else 7.6) * GIB),
        "memory": {
            "peak_resident_bytes": int((11.3 if a else 8.8) * GIB),
            "peak_footprint_bytes": int((14.8 if a else 12.1) * GIB),
            "min_host_free_pct": 41.0 if a else 47.0,
            "phases": {},
        },
        "frames": 25,
        "released_encoders": [],
        "stamp": {"prompt_sha256": "p"},
        "git_sha": "abc1234",
        "mlx_teacache_version": "0.11.2.dev1",
    }
    if not a:
        kinds = ["computed"] + ["computed", "skipped"] * 12
        base.update(
            rel_l1_thresh=0.2,
            decision_kinds=kinds,
            skipped=12,
            computed=13,
            max_consecutive_skips=1,
            skip_pattern="".join("S" if k == "skipped" else "C" for k in kinds),
        )
    base.update(over)
    return base


def _report() -> dict:
    entry = bc.assemble_entry(
        cr.recipe_for("flux1-dev"),
        _result("a"),
        _result("b"),
        ssim=0.9712,
        provenance={
            "version_mflux": "0.20.0",
            "version_mlx": "0.32.2",
            "version_mlx_taef": "0.8.3",
            "mlx_teacache_version": "0.11.2",
        },
    )
    return {
        "schema_version": 2,
        "prompt": "P",
        "seed": 42,
        "qwen_prompt_suffix": ", S.",
        "hardware": {"chip": "Apple M1 Max", "ram_gb": 32, "os": "macOS 27.0", "python": "3.12.9"},
        "variants": {"flux1-dev": entry},
    }


def test_summary_rows_put_each_condition_in_its_own_column() -> None:
    """Bug: A and B columns swapped, or the wall and preview-subtracted speedups swapped."""
    s = gen.render_blocks(_report())["flux1-dev:summary"]
    assert "| Generation | 239.1 s · peak 10.4 GiB | 198.9 s · peak 7.6 GiB · 12 of 25 steps skipped |" in s
    # Literal, not re-derived from the entry: 239.14 / 198.9 = 1.20x wall; (239.14-6.0) / (198.9-6.5) = 1.21x
    # preview-subtracted (a_preview = 25 * 0.24 = 6.0 s, b_preview = 25 * 0.26 = 6.5 s).
    assert "On this run: 1.20× faster (1.21× with" in s
    assert "SSIM 0.97" in s and "(docs/comparison/flux1-dev.md)" in s
    assert "(_artifacts/v0.10.0_bench_flux1_dev.json)" in s


def test_machine_line_reports_the_macos_marketing_version_not_the_kernel() -> None:
    """Bug: the header prints 'macOS kernel Darwin 27.0.0' -- Darwin is the kernel name, not the macOS
    version a reader expects (e.g. 'macOS 27.0')."""
    assert (
        gen.render_blocks(_report())["machine"]
        == "Apple M1 Max, 32 GB unified memory, macOS 27.0, Python 3.12.9."
    )


def test_details_rows_are_per_condition() -> None:
    """Bug: a details row reads the same side twice."""
    d = gen.render_blocks(_report())["flux1-dev:details"]
    assert "| Model load (weights evaluated) | 21.0 s | 22.0 s |" in d
    assert "| Process footprint (macOS), peak | 14.8 GiB | 12.1 GiB |" in d
    assert "| Median skipped step | — | 1.2 s |" in d
    assert "0.20.0" in d


def test_subpage_images_are_relative_to_docs_comparison() -> None:
    """Bug: sub-page image links resolve from the repo root and break on GitHub."""
    assert (
        "(../../_artifacts/comparison/flux1-dev/steps-a.jpg)"
        in gen.render_blocks(_report())["flux1-dev:sheets"]
    )


def test_splice_replaces_between_markers_and_survives_backslashes() -> None:
    """Bug: re.sub replacement-string escapes corrupt a block containing a backslash."""
    text = "a\n<!-- COMPARISON:k START -->\nold\n<!-- COMPARISON:k END -->\nb\n"
    out = gen.splice(text, {"k": "new \\1 \\g<0>"}, required={"k"})
    assert "new \\1 \\g<0>" in out and "old" not in out and out.startswith("a\n") and out.endswith("b\n")


def test_splice_fails_on_missing_and_orphan_markers() -> None:
    """Bug: a page silently keeps stale numbers because its marker was renamed or dropped."""
    with pytest.raises(ValueError, match="missing"):
        gen.splice("no markers", {"k": "x"}, required={"k"})
    with pytest.raises(ValueError, match="orphan"):
        gen.splice(
            "<!-- COMPARISON:gone START -->\nx\n<!-- COMPARISON:gone END -->\n", {"k": "x"}, required=set()
        )


def test_page_blocks_split_main_page_from_subpages() -> None:
    """Bug: a sub-page is required to carry the main page's blocks, or vice versa."""
    assert gen.page_blocks(None, ["flux1-dev"]) == {"machine", "flux1-dev:summary"}
    assert gen.page_blocks("flux1-dev", ["flux1-dev"]) == {"flux1-dev:sheets", "flux1-dev:details"}


def test_committed_pages_match_the_committed_report() -> None:
    """Bug: a page was hand-edited after --write, or --write was never re-run before commit, so the
    committed prose no longer matches the committed report's numbers."""
    report = json.loads(gen.REPORT.read_text())
    blocks = gen.render_blocks(report)
    for page, required in gen._pages(report):
        original = page.read_text()
        assert gen.splice(original, blocks, required=required) == original, f"{page} is out of date"


def test_committed_report_image_and_bench_paths_all_exist() -> None:
    """Bug: an image or bench-report path in the committed report points at a file that was never
    committed, or was moved or deleted after the report was written."""
    report = json.loads(gen.REPORT.read_text())
    missing = []
    for slug, v in report["variants"].items():
        for key, rel in v["images"].items():
            if not (gen.REPO / rel).exists():
                missing.append(f"{slug}.images.{key} -> {rel}")
        if not (gen.REPO / v["bench_report"]).exists():
            missing.append(f"{slug}.bench_report -> {v['bench_report']}")
    assert not missing, f"report paths do not exist on disk: {missing}"
