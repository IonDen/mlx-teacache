"""Comparison page generator: numbers come from the report, prose stays hand-written (pure-core lane)."""

import json
import re
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
        "stamp": {"prompt_sha256": cr.prompt_sha256(cr.PROMPT)},
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
        mflux_compiles_on_this_chip=True,
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
    assert "(_artifacts/v0.18.0_bench_flux1_dev.json)" in s


def test_klein_base_pages_link_the_v0_12_0_bench_reports() -> None:
    """Bug: the Klein base 4B/9B summary rows and detail pages still citing the v0.10.0 bench reports
    after the v0.12.0 gate was re-benched, so the linked numbers disagree with the README table. Same for
    FLUX.1-dev, whose current multi-run record is v0.18.0 while the comparison run names v0.10.0."""
    blocks = gen.render_blocks(json.loads(gen.REPORT.read_text()))
    for slug, path in (
        ("klein-base-4b", "_artifacts/v0.12.0_bench_klein_base_4b.json"),
        ("klein-base-9b", "_artifacts/v0.12.0_bench_klein_base_9b.json"),
    ):
        assert f"[bench report]({path})" in blocks[f"{slug}:summary"]
        assert f"[bench report](../../{path})" in blocks[f"{slug}:details"]
    assert "[bench report](_artifacts/v0.18.0_bench_flux1_dev.json)" in blocks["flux1-dev:summary"]
    assert "[bench report](../../_artifacts/v0.18.0_bench_flux1_dev.json)" in blocks["flux1-dev:details"]


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


def test_details_peak_row_names_what_each_phase_boundary_actually_covers() -> None:
    """Bug: the row is labelled "MLX peak: load / encode / generation", but the sampled "encode" boundary
    sits after the transformer and VAE are evaluated too (_run_generation samples mlx_peak_encode_bytes
    once the denoiser is evaluated, not right after prompt encoding), so a reader takes the middle number
    as prompt-encoding-only memory when it also counts loading the model."""
    d = gen.render_blocks(_report())["flux1-dev:details"]
    assert "| MLX peak: encoders loaded / prompt encoded + model loaded / generation |" in d
    assert "| MLX peak: load / encode / generation |" not in d


def test_details_footer_has_a_blank_line_before_the_speedup_paragraph() -> None:
    """Bug: the footer starts with a single "\\n", so on GitHub the "Speedup on this run: ..." paragraph
    renders as an extra one-cell row of the table above it instead of its own paragraph."""
    d = gen.render_blocks(_report())["flux1-dev:details"]
    assert "|\n\nSpeedup on this run" in d


def _details_for(version: str, sha: str = "8c3fff5") -> str:
    report = _report()
    report["variants"]["flux1-dev"]["provenance"]["mlx_teacache_version"] = version
    report["variants"]["flux1-dev"]["provenance"]["git_sha_b"] = sha
    return gen.render_blocks(report)["flux1-dev:details"]


_NOT_CLEAN = "records the commit of each run but not whether each working tree was clean"
_UNCOMMITTED = "this run's version string marks uncommitted changes at install time"


def test_details_footer_prints_a_release_version_as_recorded_with_no_install_mark() -> None:
    """Bug: a plain release string is relabelled (decremented, or called install-time), or the footer claims
    uncommitted changes for a stamp that carries no ``.d<date>`` suffix."""
    d = _details_for("0.11.1", "bcaac6f")
    assert "mlx-teacache 0.11.1 (harness at commit `bcaac6f`)," in d
    assert f"{_NOT_CLEAN}. " in d
    assert "install" not in d


def test_details_footer_prints_a_dev_version_verbatim_as_install_time_without_a_date_mark() -> None:
    """Bug: a dev build's own string is replaced by the previous release ("0.11.1") instead of being shown as
    the run recorded it, or the uncommitted mark is keyed on ``.dev`` rather than on the ``.d<date>`` suffix."""
    d = _details_for("0.11.2.dev14+gbcaac6f1c")
    assert (
        "mlx-teacache 0.11.2.dev14+gbcaac6f1c (install-time version string; harness at commit `8c3fff5`),"
        in d
    )
    assert f"{_NOT_CLEAN}. " in d
    assert "uncommitted" not in d


def test_details_footer_says_a_dated_dev_stamp_marks_uncommitted_changes_at_install() -> None:
    """Bug: the page credits a ``.d<date>`` stamp to a release, or drops the note that the suffix marks
    uncommitted changes at install time."""
    d = _details_for("0.11.2.dev14+gbcaac6f1c.d20260925")
    assert (
        "mlx-teacache 0.11.2.dev14+gbcaac6f1c.d20260925 (install-time version string; harness at commit "
        "`8c3fff5`)," in d
    )
    assert f"{_NOT_CLEAN}; {_UNCOMMITTED}. " in d


def test_details_footer_does_not_take_a_dev_number_for_a_date_suffix() -> None:
    """Bug: the suffix rule matches ".dev" or a short ".d" run, so a stamp with no install date gets the
    uncommitted mark (the hatch-vcs date is exactly eight digits at the end of the string)."""
    assert "uncommitted" not in _details_for("0.11.2.dev20260925")
    assert "uncommitted" not in _details_for("0.11.2.dev1+g992128837.d2026092")
    assert _UNCOMMITTED in _details_for("0.11.2.dev1+g992128837.d20260925")


def test_details_footer_says_a_dirty_describe_string_marks_uncommitted_changes_at_install() -> None:
    """Bug: a git-describe string built from a dirty tree ("-dirty" at the end) gets no uncommitted mark, so the
    footer reads as if nothing marked that run's tree; or a "-dirty" in the middle of the string counts."""
    d = _details_for("v0.12.0-3-gabc1234-dirty")
    assert (
        "mlx-teacache v0.12.0-3-gabc1234-dirty (install-time version string; harness at commit `8c3fff5`),"
        in d
    )
    assert f"{_NOT_CLEAN}; {_UNCOMMITTED}. " in d
    assert "uncommitted" not in _details_for("v0.12.0-3-gabc1234")
    assert "uncommitted" not in _details_for("v0.12.0-dirtyfix-3-gabc1234")


def test_version_line_says_when_no_version_was_recorded() -> None:
    """Bug: a run with no ``mlx_teacache_version`` (missing, empty or null) prints "mlx-teacache  (install-time
    version string; …)": a double space, no version, and a label for a string that does not exist."""
    expected = "mlx-teacache version not recorded (harness at commit `bbb1111`)"
    assert gen.version_line({"git_sha_b": "bbb1111"}) == expected
    assert gen.version_line({"mlx_teacache_version": "", "git_sha_b": "bbb1111"}) == expected
    assert gen.version_line({"mlx_teacache_version": None, "git_sha_b": "bbb1111"}) == expected
    assert "uncommitted" not in _details_for("")


def test_committed_pages_print_each_runs_own_version_string() -> None:
    """Bug: a comparison page prints one fixed release for every run instead of the version string its own
    run recorded in the committed report."""
    report = json.loads(gen.REPORT.read_text())
    blocks = gen.render_blocks(report)
    expected = {
        "flux1-dev": ("0.11.1", False),
        "flux1-krea-dev": ("0.11.1", False),
        "klein-base-4b": ("0.11.2.dev1+g992128837.d20260925", True),
        "klein-base-9b": ("0.11.2.dev1+g992128837.d20260925", True),
        "z-image-base": ("0.11.2.dev1+g992128837.d20260925", True),
        "qwen-image": ("0.11.2.dev14+gbcaac6f1c.d20260925", True),
    }
    assert set(report["variants"]) == set(expected)
    for slug, (version, uncommitted) in expected.items():
        d = blocks[f"{slug}:details"]
        assert f"mlx-teacache {version} (" in d, slug
        assert (_UNCOMMITTED in d) is uncommitted, slug


def test_version_line_credits_the_b_run_commit_and_falls_back_to_the_a_run_commit() -> None:
    """Bug: the line credits the A run's commit when the B run recorded its own, or shows no commit (None)
    when only the A run recorded one."""
    both = gen.version_line(
        {"mlx_teacache_version": "0.11.1", "git_sha_a": "aaa0000", "git_sha_b": "bbb1111"}
    )
    assert both == "mlx-teacache 0.11.1 (harness at commit `bbb1111`)"
    only_a = gen.version_line({"mlx_teacache_version": "0.11.1", "git_sha_a": "ccc2222"})
    assert only_a == "mlx-teacache 0.11.1 (harness at commit `ccc2222`)"


def test_details_footer_says_text_encoder_freed_singular() -> None:
    """Bug: the footer says "text encoders freed" (plural) even though every variant that frees one
    frees exactly one (Klein 9B and Qwen each have a single `text_encoder` attribute, not separate
    clip/t5 encoders)."""
    report = _report()
    report["variants"]["flux1-dev"]["free_encoders"] = True
    d = gen.render_blocks(report)["flux1-dev:details"]
    assert "text encoder freed once the prompt is encoded" in d
    assert "text encoders freed" not in d


def test_details_footer_reads_the_seed_from_the_report() -> None:
    """Bug: the footer hard-codes "seed 42" instead of rendering report["seed"], so a page generated from
    a report recorded under a different seed would still claim seed 42."""
    report = _report()
    report["seed"] = 7
    d = gen.render_blocks(report)["flux1-dev:details"]
    assert "seed 7" in d
    assert "seed 42" not in d


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
    assert gen.page_blocks("flux1-dev", ["flux1-dev"]) == {
        "flux1-dev:prompt",
        "flux1-dev:sheets",
        "flux1-dev:details",
    }


def test_committed_pages_match_the_committed_report() -> None:
    """Bug: a page was hand-edited after --write, or --write was never re-run before commit, so the
    committed prose no longer matches the committed report's numbers."""
    report = json.loads(gen.REPORT.read_text())
    blocks = gen.render_blocks(report)
    for page, required in gen._pages(report):
        original = page.read_text()
        assert gen.splice(original, blocks, required=required) == original, f"{page} is out of date"


def test_committed_report_image_and_bench_paths_all_exist() -> None:
    """Bug: an image path in the committed report, or a bench-report link the generator renders from it
    onto the comparison pages, points at a file that was never committed, or was moved or deleted."""
    report = json.loads(gen.REPORT.read_text())
    blocks = gen.render_blocks(report)
    missing = []
    for slug, v in report["variants"].items():
        for key, rel in v["images"].items():
            if not (gen.REPO / rel).exists():
                missing.append(f"{slug}.images.{key} -> {rel}")
        for block, page_dir in ((f"{slug}:summary", gen.REPO), (f"{slug}:details", gen.SUB_DIR)):
            (link,) = re.findall(r"\[bench report\]\(([^)]+)\)", blocks[block])
            if not (page_dir / link).resolve().exists():
                missing.append(f"{block} bench report -> {link}")
    assert not missing, f"paths do not exist on disk: {missing}"


def _entry_with(prompt_text: str, **extra: object) -> dict:
    return {
        "steps": 50,
        "guidance": 4.0,
        "quantize": 8,
        "width": 864,
        "height": 1152,
        "prompt_sha256": cr.prompt_sha256(prompt_text),
        **extra,
    }


def test_prompt_block_shows_the_models_own_prompt_negative_and_settings() -> None:
    """Bug: a model that ran with its own prompt and a negative (Z-Image) gets the shared prompt on its
    page, the negative is left off, or the settings line shows another model's size."""
    report = {"prompt": "SHARED", "qwen_prompt_suffix": ", S.", "seed": 42}
    v = _entry_with(
        "a tennis player, soft glow",
        prompt="a tennis player, soft glow",
        negative_prompt="blotchy skin",
        negative_prompt_sha256=cr.prompt_sha256("blotchy skin"),
    )
    assert gen._prompt("z-image-base", v, report) == (
        "> a tennis player, soft glow\n\n"
        "Negative prompt:\n\n"
        "> blotchy skin\n\n"
        "Seed 42, 50 steps, guidance 4.0, q8, 864×1152."
    )


def test_prompt_block_falls_back_to_the_shared_prompt_and_qwen_suffix_for_older_entries() -> None:
    """Bug: an entry written before per-model prompts were recorded renders an empty or missing prompt, or
    Qwen's page drops the vendor suffix it actually ran with (or another model's page gains it)."""
    report = {"prompt": "SHARED", "qwen_prompt_suffix": ", S.", "seed": 42}
    assert gen._prompt("flux1-dev", _entry_with("SHARED"), report).startswith("> SHARED\n\nSeed 42,")
    assert gen._prompt("qwen-image", _entry_with("SHARED, S."), report).startswith("> SHARED, S.\n\nSeed 42,")


def test_prompt_block_refuses_a_prompt_that_is_not_the_one_measured() -> None:
    """Bug: the page shows a prompt whose hash differs from the prompt_sha256 the run recorded, so readers
    see different wording from what produced the images."""
    report = {"prompt": "SHARED", "qwen_prompt_suffix": ", S.", "seed": 42}
    with pytest.raises(ValueError, match="flux1-dev"):
        gen._prompt("flux1-dev", _entry_with("SOMETHING ELSE"), report)
    with pytest.raises(ValueError, match="z-image-base"):
        gen._prompt("z-image-base", _entry_with("measured", prompt="edited later"), report)


def test_prompt_block_refuses_a_negative_prompt_that_is_not_the_one_measured() -> None:
    """Bug: the page shows a negative prompt whose hash differs from negative_prompt_sha256 (or the field is
    missing entirely), so readers see negative wording that was not actually the one used to produce the
    images."""
    report = {"prompt": "SHARED", "qwen_prompt_suffix": ", S.", "seed": 42}
    edited = _entry_with(
        "measured",
        prompt="measured",
        negative_prompt="edited later",
        negative_prompt_sha256=cr.prompt_sha256("the real negative"),
    )
    with pytest.raises(ValueError, match="z-image-base"):
        gen._prompt("z-image-base", edited, report)

    missing_hash = _entry_with("measured", prompt="measured", negative_prompt="blotchy skin")
    with pytest.raises(ValueError, match="z-image-base"):
        gen._prompt("z-image-base", missing_hash, report)

    matching = _entry_with(
        "measured",
        prompt="measured",
        negative_prompt="blotchy skin",
        negative_prompt_sha256=cr.prompt_sha256("blotchy skin"),
    )
    assert "blotchy skin" in gen._prompt("z-image-base", matching, report)


def test_assemble_entry_records_the_prompt_and_negative_the_recipe_sent() -> None:
    """Bug: the report entry drops the per-model prompt/negative, so a model page falls back to the shared
    prompt even though that model ran with its own wording (Z-Image)."""
    z = cr.recipe_for("z-image-base")
    sha = {"stamp": {"prompt_sha256": cr.prompt_sha256(cr.prompt_for(z))}}
    entry = bc.assemble_entry(
        z, _result("a", **sha), _result("b", **sha), ssim=0.9, provenance={}, mflux_compiles_on_this_chip=True
    )
    assert entry["prompt"] == cr.Z_IMAGE_PROMPT
    assert entry["negative_prompt"] == cr.Z_IMAGE_NEGATIVE_PROMPT
    plain = bc.assemble_entry(
        cr.recipe_for("flux1-dev"),
        _result("a"),
        _result("b"),
        ssim=0.9,
        provenance={},
        mflux_compiles_on_this_chip=True,
    )
    assert plain["prompt"] == cr.PROMPT and plain["negative_prompt"] is None


def test_prompt_block_settings_line_says_when_the_text_encoder_was_freed() -> None:
    """Bug: the settings line under the prompt drops the freed-encoder note for the two recipes that free
    it (Klein 9B, Qwen), or adds it to a recipe that keeps its encoder."""
    report = {"prompt": "SHARED", "qwen_prompt_suffix": ", S.", "seed": 42}
    freed = gen._prompt("klein-base-9b", _entry_with("SHARED", free_encoders=True), report)
    kept = gen._prompt("flux1-dev", _entry_with("SHARED", free_encoders=False), report)
    assert freed.endswith("864×1152, text encoder freed once the prompt is encoded.")
    assert kept.endswith("864×1152.")
