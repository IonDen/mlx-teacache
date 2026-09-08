"""Pure-helper unit tests for ``scripts/bench_klein_base_vs_distilled.py``.

The script answers one question (mflux issue 113): is FLUX.2 Klein **Base +
TeaCache** worth running over the distilled Klein? It benches three conditions
on one prompt + seed — ``distilled`` (its default 4 steps), ``base`` (50 steps,
CFG), ``base-teacache`` (base + the wrapper) — subprocess per (condition, rep),
and prints a paste-ready markdown table with wall-clock, peak memory, and SSIM
of each condition against the distilled and the base image.

Everything here is the pure core: the recipe map, the per-rep metrics, the
chunk-resume layer, the SSIM helper, and the table renderer. The worker and
orchestrator import mflux / mlx / PIL / skimage only lazily, so this module
imports cleanly in the pure-core lane.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_klein_base_vs_distilled as bk  # noqa: E402

# --- condition_recipe: the three conditions map to the right model + recipe ---


def test_distilled_recipe_uses_the_distilled_config_and_four_steps() -> None:
    # Bug caught: pointing "distilled" at the base config, or running it at 50
    # steps, would make the whole comparison meaningless.
    r = bk.condition_recipe(size="4b", condition="distilled")
    assert r.model_config_name == "flux2_klein_4b"
    assert r.num_inference_steps == 4
    assert r.guidance == 1.0
    assert r.use_teacache is False


def test_base_recipe_uses_the_base_config_fifty_steps_cfg_no_wrapper() -> None:
    r = bk.condition_recipe(size="4b", condition="base")
    assert r.model_config_name == "flux2_klein_base_4b"
    assert r.num_inference_steps == 50
    assert r.guidance == 4.0
    assert r.use_teacache is False


def test_base_teacache_recipe_is_base_plus_the_wrapper() -> None:
    # Bug caught: forgetting to turn the wrapper on for the one condition that
    # is the whole point of the bench.
    r = bk.condition_recipe(size="4b", condition="base-teacache")
    assert r.model_config_name == "flux2_klein_base_4b"
    assert r.num_inference_steps == 50
    assert r.guidance == 4.0
    assert r.use_teacache is True


def test_recipe_threads_the_size_into_both_configs() -> None:
    assert bk.condition_recipe(size="9b", condition="distilled").model_config_name == "flux2_klein_9b"
    assert bk.condition_recipe(size="9b", condition="base").model_config_name == "flux2_klein_base_9b"


def test_recipe_rejects_an_unknown_size() -> None:
    with pytest.raises(ValueError, match="size"):
        bk.condition_recipe(size="7b", condition="base")


def test_recipe_rejects_an_unknown_condition() -> None:
    with pytest.raises(ValueError, match="condition"):
        bk.condition_recipe(size="4b", condition="wrapper")


def test_conditions_tuple_is_distilled_base_teacache_in_run_order() -> None:
    assert bk.CONDITIONS == ("distilled", "base", "base-teacache")


# --- parse_quantize: CLI string -> mflux quantize argument -------------------


def test_parse_quantize_maps_the_three_cli_choices() -> None:
    assert bk.parse_quantize("4") == 4
    assert bk.parse_quantize("8") == 8
    assert bk.parse_quantize("none") is None  # bf16, no quantization


def test_parse_quantize_rejects_a_bad_value() -> None:
    with pytest.raises(ValueError, match="quantize"):
        bk.parse_quantize("5")


# --- rep metrics + speedup ---------------------------------------------------


def test_rep_metrics_cold_is_first_warm_is_median_of_rest() -> None:
    assert bk.rep_metrics([12.0, 9.0, 7.0]) == {"cold": 12.0, "warm": 8.0, "median": 9.0}


def test_rep_metrics_one_rep_has_no_warm() -> None:
    assert bk.rep_metrics([5.0]) == {"cold": 5.0, "warm": None, "median": 5.0}


def test_headline_seconds_prefers_warm_when_present_else_cold() -> None:
    # The single "how long is one image" number: steady-state (warm) when we
    # have it, the cold rep for a one-rep preview.
    assert bk.headline_seconds([12.0, 9.0, 7.0]) == 8.0
    assert bk.headline_seconds([5.0]) == 5.0


def test_speedup_vs_base_is_base_over_condition() -> None:
    assert bk.speedup_vs_base(base_seconds=500.0, cond_seconds=400.0) == 1.25
    assert bk.speedup_vs_base(base_seconds=500.0, cond_seconds=500.0) == 1.0


def test_speedup_vs_base_is_none_on_missing_or_zero() -> None:
    assert bk.speedup_vs_base(base_seconds=None, cond_seconds=400.0) is None
    assert bk.speedup_vs_base(base_seconds=500.0, cond_seconds=None) is None
    assert bk.speedup_vs_base(base_seconds=500.0, cond_seconds=0.0) is None


# --- chunk persistence + resume (subprocess-per-(condition, rep)) -------------


def _fake_chunk(
    condition: str, rep: int, secs: float, *, quantize: int = 4, skipped: int = 0
) -> dict[str, Any]:
    return {
        "condition": condition,
        "rep": rep,
        "elapsed_s": secs,
        "quantize": quantize,
        "num_inference_steps": 50 if condition != "distilled" else 4,
        "peak_memory_gb": 9.0,
        "stats_summary": {"skipped_count": skipped, "computed_count": 48 - skipped} if skipped else {},
        "provenance": {"mlx_teacache_version": "0.11.0", "mflux_version": "0.18.0"},
    }


def test_chunk_path_is_condition_and_rep_keyed(tmp_path: Path) -> None:
    assert bk.chunk_path(tmp_path, "base-teacache", 2) == tmp_path / "base-teacache_rep2.json"


def test_pending_chunks_lists_every_pair_rep_outer_when_nothing_persisted(tmp_path: Path) -> None:
    # Rep-outer ordering (all conditions of rep 0, then rep 1...) so slow host
    # drift over a multi-hour run lands on every condition alike.
    assert bk.pending_chunks(list(bk.CONDITIONS), 2, tmp_path) == [
        ("distilled", 0),
        ("base", 0),
        ("base-teacache", 0),
        ("distilled", 1),
        ("base", 1),
        ("base-teacache", 1),
    ]


def test_pending_chunks_skips_a_persisted_chunk(tmp_path: Path) -> None:
    bk.persist_chunk(tmp_path, _fake_chunk("distilled", 0, 3.0))
    assert ("distilled", 0) not in bk.pending_chunks(list(bk.CONDITIONS), 1, tmp_path)
    assert ("base", 0) in bk.pending_chunks(list(bk.CONDITIONS), 1, tmp_path)


def test_persist_chunk_round_trips_and_creates_dirs(tmp_path: Path) -> None:
    result = _fake_chunk("base", 1, 500.0)
    written = bk.persist_chunk(tmp_path / "nested", result)
    assert written == bk.chunk_path(tmp_path / "nested", "base", 1)
    assert json.loads(written.read_text()) == result


def test_load_chunks_is_none_until_every_pair_exists(tmp_path: Path) -> None:
    bk.persist_chunk(tmp_path, _fake_chunk("distilled", 0, 3.0))
    bk.persist_chunk(tmp_path, _fake_chunk("base", 0, 500.0))
    assert bk.load_chunks(list(bk.CONDITIONS), 1, tmp_path) is None
    bk.persist_chunk(tmp_path, _fake_chunk("base-teacache", 0, 420.0, skipped=8))
    loaded = bk.load_chunks(list(bk.CONDITIONS), 1, tmp_path)
    assert loaded is not None
    assert loaded["base-teacache"][0]["stats_summary"]["skipped_count"] == 8


def test_verify_chunk_recipes_refuses_a_quantize_mismatch(tmp_path: Path) -> None:
    # Bug caught: a resume at a different --quantize silently aggregating a q4
    # timing under a q8 report header.
    bk.persist_chunk(tmp_path, _fake_chunk("distilled", 0, 3.0, quantize=4))
    with pytest.raises(SystemExit, match="quantize"):
        bk.verify_chunk_recipes(list(bk.CONDITIONS), 1, tmp_path, quantize=8, height=1024, width=768)


def test_verify_chunk_recipes_accepts_matching_quantize(tmp_path: Path) -> None:
    bk.persist_chunk(tmp_path, _fake_chunk("distilled", 0, 3.0, quantize=4))
    bk.verify_chunk_recipes(list(bk.CONDITIONS), 1, tmp_path, quantize=4, height=1024, width=768)  # no raise


def test_verify_chunk_recipes_refuses_a_resolution_mismatch(tmp_path: Path) -> None:
    # Bug caught: resuming at a different resolution and aggregating a 768x1024
    # timing under a 512x512 report header.
    chunk = {**_fake_chunk("distilled", 0, 3.0, quantize=4), "height": 1024, "width": 768}
    bk.persist_chunk(tmp_path, chunk)
    with pytest.raises(SystemExit, match="resolution|512"):
        bk.verify_chunk_recipes(list(bk.CONDITIONS), 1, tmp_path, quantize=4, height=512, width=512)


def test_verify_chunk_recipes_accepts_a_chunk_whose_resolution_matches(tmp_path: Path) -> None:
    # The equality half of the resolution guard: a chunk that records height/width
    # matching this invocation must NOT raise (mutating the guard to always-refuse is
    # caught here; the mismatch test alone leaves that path uncovered).
    chunk = {**_fake_chunk("distilled", 0, 3.0, quantize=4), "height": 512, "width": 512}
    bk.persist_chunk(tmp_path, chunk)
    bk.verify_chunk_recipes(list(bk.CONDITIONS), 1, tmp_path, quantize=4, height=512, width=512)  # no raise


# --- resolution in the tag (keeps 512x512 9B artifacts from colliding with 768x1024) ---


def test_chunk_tag_default_resolution_has_no_suffix() -> None:
    # The shared-portrait default (768x1024) keeps the plain tag, so the already
    # committed 4b_q4 artifacts are unaffected.
    assert bk.chunk_tag(size="4b", quantize=4, height=1024, width=768) == "4b_q4"


def test_chunk_tag_nondefault_resolution_appends_width_by_height() -> None:
    # Bug caught: a 512x512 9B run overwriting the 768x1024 artifacts because the
    # tag ignored resolution.
    assert bk.chunk_tag(size="9b", quantize=4, height=512, width=512) == "9b_q4_512x512"


def test_chunk_tag_bf16_with_nondefault_resolution() -> None:
    assert bk.chunk_tag(size="9b", quantize=None, height=512, width=512) == "9b_bf16_512x512"


# --- SSIM helper (real skimage, tiny arrays, no weights) ---------------------


def test_ssim_of_identical_images_is_one() -> None:
    pytest.importorskip("skimage")  # not in the pure-core CI env; runs in the mflux lanes
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    assert bk.ssim_from_arrays(img, img) == pytest.approx(1.0)


def test_ssim_of_structurally_different_images_is_near_zero() -> None:
    # Bug caught: an SSIM stuck at 1.0 (two unrelated images must read near 0), and a
    # dropped channel_axis (skimage then raises on the (32,32,3) array). A wrong
    # data_range is NOT caught here: for high-magnitude images its C1/C2 constants are
    # negligible next to the pixel variance, so no simple fixture distinguishes it.
    pytest.importorskip("skimage")
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    b = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)  # independent
    assert bk.ssim_from_arrays(a, b) < 0.1


def test_ssim_from_files_reads_images_and_converts_rgba(tmp_path: Path) -> None:
    # Covers load_rgb_array (its .convert("RGB") must drop an alpha channel) and
    # ssim_from_files: the same pixels via an RGB file and an RGBA file score 1.0.
    pytest.importorskip("skimage")
    from PIL import Image

    rng = np.random.default_rng(3)
    arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    rgb_path = tmp_path / "a.png"
    rgba_path = tmp_path / "b.png"
    Image.fromarray(arr, "RGB").save(rgb_path)
    rgba = np.concatenate([arr, np.full((32, 32, 1), 255, np.uint8)], axis=-1)
    Image.fromarray(rgba, "RGBA").save(rgba_path)
    assert bk.ssim_from_files(rgb_path, rgba_path) == pytest.approx(1.0)


def test_resolve_ssim_reuses_prior_when_pngs_are_gone(tmp_path: Path) -> None:
    # After the first full run converts the PNGs to webp, a re-run must reuse the SSIM
    # it already computed rather than recomputing from the lossy webp. With no PNGs on
    # disk, _resolve_ssim returns the prior value untouched (no skimage needed).
    (tmp_path / "base.webp").write_bytes(b"")  # a webp is present, the PNGs are not
    prior = {"base_vs_distilled": 0.5, "base_teacache_vs_base": 0.98}
    assert bk._resolve_ssim(tmp_path, prior) == prior


def test_resolve_ssim_recomputes_when_a_png_is_fresh(tmp_path: Path) -> None:
    # Bug caught (an all-or-nothing gate): after --only regenerates one condition's
    # PNG, a rebuild must recompute from the current images, not keep the stale prior.
    # With any PNG present, _resolve_ssim recomputes every pair.
    pytest.importorskip("skimage")
    from PIL import Image

    rng = np.random.default_rng(5)
    for cond in bk.CONDITIONS:
        arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(tmp_path / f"{cond}.png")
    stale = {"base_vs_distilled": 0.99, "base_teacache_vs_distilled": 0.99, "base_teacache_vs_base": 0.99}
    out = bk._resolve_ssim(tmp_path, stale)
    assert out != stale  # recomputed, not reused
    assert out["base_teacache_vs_base"] < 0.5  # three unrelated images score low, not 0.99


def test_prior_ssim_reads_a_valid_report_and_tolerates_a_corrupt_one(tmp_path: Path) -> None:
    # A killed run can leave a half-written report; the next rebuild must not crash.
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"ssim": {"base_teacache_vs_base": 0.97}}))
    assert bk._prior_ssim(good) == {"base_teacache_vs_base": 0.97}
    bad = tmp_path / "bad.json"
    bad.write_text('{"ssim": {"base_teacache_vs_base": 0.97')  # truncated JSON
    assert bk._prior_ssim(bad) == {}
    assert bk._prior_ssim(tmp_path / "missing.json") == {}


# --- markdown table (the paste-ready deliverable) ----------------------------


def _rows() -> list["bk.TableRow"]:
    return [
        bk.TableRow(
            condition="distilled",
            steps=4,
            wall_clock_s=15.0,
            speedup_vs_base=34.0,
            peak_gb=8.1,
            ssim_vs_distilled=1.0,
            ssim_vs_base=0.62,
            skipped=None,
        ),
        bk.TableRow(
            condition="base",
            steps=50,
            wall_clock_s=510.0,
            speedup_vs_base=1.0,
            peak_gb=9.0,
            ssim_vs_distilled=0.62,
            ssim_vs_base=1.0,
            skipped=None,
        ),
        bk.TableRow(
            condition="base-teacache",
            steps=50,
            wall_clock_s=420.0,
            speedup_vs_base=1.21,
            peak_gb=9.0,
            ssim_vs_distilled=0.61,
            ssim_vs_base=0.986,
            skipped=8,
        ),
    ]


def test_table_has_a_header_and_one_row_per_condition() -> None:
    table = bk.render_markdown_table(_rows())
    lines = [ln for ln in table.splitlines() if ln.strip()]
    assert lines[0].startswith("|") and "Wall-clock" in lines[0]
    assert "SSIM vs distilled" in lines[0] and "SSIM vs base" in lines[0]
    assert lines[1].count("---") >= 5  # the separator row
    assert len([ln for ln in lines[2:]]) == 3  # three data rows


def test_table_renders_each_column_in_the_right_position() -> None:
    # Bug caught: a column swap INSIDE render_markdown_table (e.g. ssim_vs_base vs
    # ssim_vs_distilled), or a dropped speedup/skip-annotation. Substring presence
    # alone misses a value that moved to the wrong column, so assert whole rows.
    table = bk.render_markdown_table(_rows())
    assert "| distilled | 4 | 15 | 34.00× | 8.1 | 1.000 | 0.620 |" in table
    assert "| base | 50 | 510 | 1.00× | 9.0 | 0.620 | 1.000 |" in table
    assert "| base+TeaCache | 50 (−8 skipped) | 420 | 1.21× | 9.0 | 0.610 | 0.986 |" in table


def test_table_rows_from_report_fills_speedup_and_ssim(tmp_path: Path) -> None:
    # A report where base=500s, base-teacache=400s, distilled=10s.
    report = {
        "conditions": {
            "distilled": {"headline_seconds": 10.0, "num_inference_steps": 4, "peak_memory_gb": 8.0},
            "base": {"headline_seconds": 500.0, "num_inference_steps": 50, "peak_memory_gb": 9.0},
            "base-teacache": {
                "headline_seconds": 400.0,
                "num_inference_steps": 50,
                "peak_memory_gb": 9.0,
                "skipped_median": 8,
            },
        },
        "ssim": {
            "base_vs_distilled": 0.6,
            "base_teacache_vs_distilled": 0.6,
            "base_teacache_vs_base": 0.98,
        },
    }
    rows = {r.condition: r for r in bk.table_rows_from_report(report)}
    assert rows["base"].speedup_vs_base == 1.0
    assert rows["base-teacache"].speedup_vs_base == pytest.approx(1.25)
    assert rows["distilled"].speedup_vs_base == pytest.approx(50.0)
    # self-SSIM on the diagonal, the measured pair off it
    assert rows["distilled"].ssim_vs_distilled == 1.0
    assert rows["base"].ssim_vs_base == 1.0
    assert rows["base-teacache"].ssim_vs_base == 0.98
    assert rows["base-teacache"].skipped == 8


# --- per-condition aggregation ------------------------------------------------


def test_aggregate_condition_takes_median_seconds_and_peak_max() -> None:
    reps = [
        _fake_chunk("base-teacache", 0, 430.0, skipped=8),
        _fake_chunk("base-teacache", 1, 410.0, skipped=8),
        _fake_chunk("base-teacache", 2, 420.0, skipped=8),
    ]
    reps[1]["peak_memory_gb"] = 9.5  # peak is the worst rep, not the last
    agg = bk.aggregate_condition(reps)
    assert agg["rep_seconds"] == [430.0, 410.0, 420.0]
    assert agg["headline_seconds"] == bk.headline_seconds([430.0, 410.0, 420.0])
    assert agg["peak_memory_gb"] == 9.5
    assert agg["skipped_median"] == 8
