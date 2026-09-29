"""Pure helpers of the schema-2 comparison harness (pure-core lane: bench_comparison imports mflux lazily)."""

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _comparison_recipes as cr  # noqa: E402
import bench_comparison as bc  # noqa: E402

GIB = 1024**3
V = {"mflux": "0.20.0"}


def _write_chunk(
    chunks: Path,
    raw: Path,
    slug: str,
    cond: str,
    *,
    frames: int,
    final: bool = True,
    stamp: dict | None = None,
    size: tuple[int, int] = (768, 1024),
) -> None:
    p = bc.chunk_path(chunks, slug, cond)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"condition": cond, "stamp": stamp or {}, "width": size[0], "height": size[1]}))
    fdir = bc.frames_dir_for(raw, slug, cond)
    fdir.mkdir(parents=True, exist_ok=True)
    for i in range(frames):
        (fdir / f"step_step{i:02d}.png").write_bytes(b"x")
    if final:
        (bc.raw_dir_for(raw, slug) / f"{cond}.png").write_bytes(b"x")


def test_chunk_complete_only_with_json_final_png_and_every_frame(tmp_path: Path) -> None:
    """Bug: a chunk with a missing frame or final image counts as done."""
    c, r = tmp_path / "c", tmp_path / "r"
    _write_chunk(c, r, "flux1-dev", "a", frames=25)
    assert bc.chunk_is_complete(c, r, "flux1-dev", "a", 25)
    assert not bc.chunk_is_complete(c, r, "flux1-dev", "a", 26)
    _write_chunk(c, r, "flux1-dev", "b", frames=25, final=False)
    assert not bc.chunk_is_complete(c, r, "flux1-dev", "b", 25)


def test_stamp_mismatch_refuses_and_names_the_field(tmp_path: Path) -> None:
    """Bug: a chunk measured at another resolution is reused."""
    r = cr.recipe_for("klein-base-9b")
    stored = {"stamp": cr.recipe_stamp(r, versions=V)}
    with pytest.raises(SystemExit, match="width"):
        bc.check_chunk_stamp(
            stored, cr.recipe_stamp(cr.with_resolution(r, 576, 768), versions=V), tmp_path / "a.json"
        )
    bc.check_chunk_stamp(stored, cr.recipe_stamp(r, versions=V), tmp_path / "a.json")
    with pytest.raises(SystemExit, match="no recipe stamp"):
        bc.check_chunk_stamp({}, {"slug": "x"}, tmp_path / "a.json")


def test_plan_conditions_checks_stamps_on_reuse_and_applies_the_budget(tmp_path: Path) -> None:
    """Bug: the reuse path skips the stamp check (old portrait chunk reused), or the budget is ignored."""
    rec = cr.recipe_for("flux1-dev")
    good = cr.recipe_stamp(rec, versions=V)
    c, r = tmp_path / "c", tmp_path / "r"
    assert bc.plan_conditions(c, r, "flux1-dev", 25, good, budget=1) == ["a"]
    assert bc.plan_conditions(c, r, "flux1-dev", 25, good, budget=-1) == ["a", "b"]
    _write_chunk(c, r, "flux1-dev", "a", frames=25, stamp=good)
    assert bc.plan_conditions(c, r, "flux1-dev", 25, good, budget=-1) == ["b"]
    _write_chunk(c, r, "flux1-dev", "b", frames=25, stamp={**good, "prompt_sha256": "portrait"})
    with pytest.raises(SystemExit, match="prompt_sha256"):
        bc.plan_conditions(c, r, "flux1-dev", 25, good, budget=-1)


def test_load_pair_requires_both_complete_matching_and_same_size(tmp_path: Path) -> None:
    """Bug: SSIM computed while one side is pending, or on A/B images of different sizes."""
    rec = cr.recipe_for("flux1-dev")
    good = cr.recipe_stamp(rec, versions=V)
    c, r = tmp_path / "c", tmp_path / "r"
    _write_chunk(c, r, "flux1-dev", "a", frames=25, stamp=good)
    with pytest.raises(SystemExit, match="pending"):
        bc.load_pair(c, r, "flux1-dev", 25, good)
    _write_chunk(c, r, "flux1-dev", "b", frames=25, stamp=good, size=(576, 768))
    with pytest.raises(SystemExit, match="size"):
        bc.load_pair(c, r, "flux1-dev", 25, good)


def test_load_pair_refuses_a_chunk_measured_under_a_different_stamp(tmp_path: Path) -> None:
    """Bug: load_pair skips the stamp check, so a chunk measured under a stale recipe/version silently
    feeds the SSIM comparison."""
    rec = cr.recipe_for("flux1-dev")
    good = cr.recipe_stamp(rec, versions=V)
    c, r = tmp_path / "c", tmp_path / "r"
    _write_chunk(c, r, "flux1-dev", "a", frames=25, stamp=good)
    _write_chunk(c, r, "flux1-dev", "b", frames=25, stamp={**good, "width": 576})
    with pytest.raises(SystemExit, match="width"):
        bc.load_pair(c, r, "flux1-dev", 25, good)


def test_parse_worker_line_prefers_an_abort_after_a_result() -> None:
    """Bug: a watchdog abort after the result line is ignored and the run counts."""
    s = bc.WORKER_RESULT_SENTINEL
    out = f"{s}{json.dumps({'condition': 'a'})}\n{s}{json.dumps({'aborted': 'x'})}\n"
    assert bc.parse_worker_line(out) == {"aborted": "x"}
    assert bc.parse_worker_line("no sentinel") is None


def _result(cond: str, compute: list[float], preview: list[float], gen: float, **extra: object) -> dict:
    base = {
        "condition": cond,
        "width": 768,
        "height": 1024,
        "load_seconds": 12.0 if cond == "a" else 13.0,
        "encode_seconds": 1.5 if cond == "a" else 1.6,
        "generation_seconds": gen,
        "compute_seconds": compute,
        "preview_seconds": preview,
        "mlx_peak_load_bytes": 9 * GIB,
        "mlx_peak_encode_bytes": 10 * GIB,
        "mlx_peak_generation_bytes": (11 if cond == "a" else 8) * GIB,
        "memory": {
            "peak_resident_bytes": 12 * GIB,
            "peak_footprint_bytes": 14 * GIB,
            "min_host_free_pct": 41.0,
            "phases": {},
        },
        "frames": len(compute),
        "released_encoders": [],
        "stamp": {"prompt_sha256": "p"},
        "git_sha": "abc1234",
        "mlx_teacache_version": "0.11.2.dev1",
    }
    base.update(extra)
    return base


def test_assemble_entry_derives_the_three_speedups_and_kind_medians() -> None:
    """Bug: a speedup uses the wrong side, step 0 included, or preview not subtracted."""
    r = cr.recipe_for("flux1-dev")
    a = _result("a", [30.0, 10.0, 10.0, 10.0], [0.5] * 4, gen=62.5)
    b = _result(
        "b",
        [30.0, 10.0, 1.0, 10.0],
        [0.5] * 4,
        gen=53.5,
        rel_l1_thresh=0.2,
        decision_kinds=["computed", "computed", "skipped", "computed"],
        skipped=1,
        computed=3,
        max_consecutive_skips=1,
        skip_pattern="CCSC",
    )
    e = bc.assemble_entry(
        r, a, b, ssim=0.97, provenance={"version_mflux": "0.20.0"}, mflux_compiles_on_this_chip=True
    )
    assert e["speedup_wall"] == pytest.approx(62.5 / 53.5)
    assert e["speedup_steady"] == pytest.approx(30.0 / 21.0)
    assert e["speedup_preview_subtracted"] == pytest.approx(60.5 / 51.5)
    assert e["b"]["step_medians"] == {"computed": 10.0, "skipped": 1.0}
    assert e["a"]["load_seconds"] == 12.0 and e["b"]["load_seconds"] == 13.0
    assert e["images"]["steps_b"] == "_artifacts/comparison/flux1-dev/steps-b.jpg"
    assert e["bench_report"] == r.bench_report and e["prompt_sha256"] == "p"


def test_assemble_entry_carries_the_negative_prompt_sha256_from_the_chunk_stamp() -> None:
    """Bug: assemble_entry drops negative_prompt_sha256 (or invents one instead of reading the literal value
    the worker actually stamped), so the page generator can never verify a rendered negative prompt against
    what was measured."""
    r = cr.recipe_for("z-image-base")
    a = _result(
        "a", [1.0, 1.0], [0.1, 0.1], gen=2.2, stamp={"prompt_sha256": "p", "negative_prompt_sha256": "np"}
    )
    b = _result(
        "b",
        [1.0, 1.0],
        [0.1, 0.1],
        gen=2.2,
        rel_l1_thresh=0.2,
        decision_kinds=["computed", "computed"],
        skipped=0,
        computed=2,
        max_consecutive_skips=0,
        skip_pattern="CC",
    )
    e = bc.assemble_entry(r, a, b, ssim=1.0, provenance={}, mflux_compiles_on_this_chip=True)
    assert e["negative_prompt_sha256"] == "np"


def test_assemble_entry_marks_a_compiled_predict_only_for_klein_and_z_image_on_a_compiling_chip() -> None:
    """Bug: FLUX.1 (no mx.compile'd predict step at all) is credited with a compiled predict step, or a
    Klein/Z-Image loader is marked compiled even on a chip where mflux doesn't compile it (M1/M2)."""
    a = _result("a", [1.0, 1.0], [0.1, 0.1], gen=2.2)
    b = _result(
        "b",
        [1.0, 1.0],
        [0.1, 0.1],
        gen=2.2,
        rel_l1_thresh=0.2,
        decision_kinds=["computed", "computed"],
        skipped=0,
        computed=2,
        max_consecutive_skips=0,
        skip_pattern="CC",
    )

    flux1 = bc.assemble_entry(
        cr.recipe_for("flux1-dev"), a, b, ssim=1.0, provenance={}, mflux_compiles_on_this_chip=True
    )
    assert flux1["a_compiled_predict"] is False

    klein = bc.assemble_entry(
        cr.recipe_for("klein-base-4b"), a, b, ssim=1.0, provenance={}, mflux_compiles_on_this_chip=True
    )
    assert klein["a_compiled_predict"] is True

    z_image = bc.assemble_entry(
        cr.recipe_for("z-image-base"), a, b, ssim=1.0, provenance={}, mflux_compiles_on_this_chip=True
    )
    assert z_image["a_compiled_predict"] is True

    klein_on_a_noncompiling_chip = bc.assemble_entry(
        cr.recipe_for("klein-base-4b"), a, b, ssim=1.0, provenance={}, mflux_compiles_on_this_chip=False
    )
    assert klein_on_a_noncompiling_chip["a_compiled_predict"] is False


def test_assemble_entry_refuses_mismatched_step_counts() -> None:
    """Bug: A and B step counts silently differ, misaligning the per-step tables."""
    r = cr.recipe_for("flux1-dev")
    a = _result("a", [1.0, 1.0], [0.1, 0.1], gen=2.2)
    with pytest.raises(ValueError, match="step counts"):
        bc.assemble_entry(
            r,
            a,
            _result("b", [1.0], [0.1], gen=1.1, decision_kinds=["computed"]),
            ssim=1.0,
            provenance={},
            mflux_compiles_on_this_chip=True,
        )


def test_assemble_entry_refuses_b_missing_decision_kinds() -> None:
    """Bug: B is treated as all-computed when its decision_kinds is missing (the KeyError must name
    decision_kinds itself, not an unrelated _B_KEYS entry that happens to be missing too)."""
    r = cr.recipe_for("flux1-dev")
    a = _result("a", [1.0, 1.0], [0.1, 0.1], gen=2.2)
    present = {k: 0 for k in bc._B_KEYS if k != "decision_kinds"}
    b = _result("b", [1.0, 1.0], [0.1, 0.1], gen=2.2, **present)
    with pytest.raises(KeyError, match="decision_kinds"):
        bc.assemble_entry(r, a, b, ssim=1.0, provenance={}, mflux_compiles_on_this_chip=True)


def test_merge_entry_replaces_one_slug_and_keeps_the_rest() -> None:
    """Bug: finishing one model drops the others from the report."""
    report = {"schema_version": 2, "variants": {"flux1-dev": {"x": 1}, "z-image-base": {"y": 2}}}
    out = bc.merge_entry(report, "flux1-dev", {"x": 3}, generated_at="2026-09-25T10:00Z")
    assert out["variants"] == {"flux1-dev": {"x": 3}, "z-image-base": {"y": 2}}
    assert out["generated_at"] == "2026-09-25T10:00Z" and report["variants"]["flux1-dev"] == {"x": 1}


def test_reset_condition_outputs_moves_stale_frames_and_final_to_the_trash(tmp_path: Path) -> None:
    """Bug: stale frames/final survive a retry (counted complete; mflux would write a_1.png beside a.png),
    or they are deleted instead of trashed (rule F)."""
    raw, trash = tmp_path / "raw", tmp_path / "trash"
    trash.mkdir()
    _write_chunk(tmp_path / "c", raw, "flux1-dev", "a", frames=3)
    moved = bc.reset_condition_outputs(raw, "flux1-dev", "a", trash=trash, tag="t")
    assert not (bc.raw_dir_for(raw, "flux1-dev") / "a.png").exists()
    assert not bc.frames_dir_for(raw, "flux1-dev", "a").exists()
    assert len(moved) == 2 and all(p.exists() and p.parent == trash for p in moved)
    assert bc.reset_condition_outputs(raw, "flux1-dev", "a", trash=trash, tag="t2") == []


def test_retire_chunk_moves_an_existing_chunk_and_returns_none_when_absent(tmp_path: Path) -> None:
    """Bug: a stale <cond>.json chunk from a failed re-run stays in place and pairs with a freshly-written
    image (mismatched provenance), instead of being retired to the Trash before the worker respawns."""
    chunks, trash = tmp_path / "c", tmp_path / "trash"
    path = bc.chunk_path(chunks, "flux1-dev", "a")
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    dest = bc.retire_chunk(path, trash=trash, tag="t")
    assert dest is not None and dest.exists() and dest.parent == trash
    assert not path.exists()
    assert bc.retire_chunk(path, trash=trash, tag="t2") is None


def test_now_tag_carries_microseconds_so_two_retires_in_one_second_never_collide() -> None:
    """Bug: two chunks retired within the same wall-clock second (a fast retry loop, or the aborted-marker
    retire landing in the same second as the chunk retire) get identical tags and one Trash destination
    silently overwrites the other."""
    import datetime

    class _FrozenNow(datetime.datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001
            return cls(2026, 9, 25, 10, 0, 0, 123456)

    real_datetime = bc.datetime
    bc.datetime = _FrozenNow
    try:
        tag = bc._now_tag()
    finally:
        bc.datetime = real_datetime
    assert tag == "2026-09-25-100000-123456"


def test_quality_probe_dir_nests_by_slug_then_name(tmp_path: Path) -> None:
    """Bug: the nesting order is reversed (name/slug), which breaks the documented layout
    tests/_artifacts/quality_probe/<slug>/<NAME>/."""
    assert bc.quality_probe_dir(tmp_path, "flux1-dev", "tighter-guidance") == (
        tmp_path / "flux1-dev" / "tighter-guidance"
    )


def test_quality_probe_dir_matches_run_generations_path_slug_nesting(tmp_path: Path) -> None:
    """Bug: quality_probe_dir and _run_generation's path_slug="<slug>/<name>" trick (raw_dir_for /
    frames_dir_for joined on a slug containing "/") stop agreeing on where the final PNG and frames land,
    so the worker's record.json ends up in a directory the image and frames are not actually written to."""
    out_dir = bc.quality_probe_dir(tmp_path, "flux1-dev", "tighter-guidance")
    path_slug = "flux1-dev/tighter-guidance"
    assert bc.raw_dir_for(tmp_path, path_slug) == out_dir
    assert bc.frames_dir_for(tmp_path, path_slug, "a") == out_dir / "frames" / "a"


def test_quality_probe_record_path_keeps_a_at_record_json_and_gives_b_its_own_file(tmp_path: Path) -> None:
    """Bug: condition B's record is written to record.json, overwriting condition A's record for the same
    probe NAME (or A moves off record.json and the existing A-only probes stop being found)."""
    out_dir = bc.quality_probe_dir(tmp_path, "z-image-base", "Z1")
    assert bc.quality_probe_record_path(tmp_path, "z-image-base", "Z1", "a") == out_dir / "record.json"
    assert bc.quality_probe_record_path(tmp_path, "z-image-base", "Z1", "b") == out_dir / "record.b.json"
    assert bc.quality_probe_record_path(tmp_path, "z-image-base", "Z1", "a", aborted=True) == (
        out_dir / "record.aborted.json"
    )
    assert bc.quality_probe_record_path(tmp_path, "z-image-base", "Z1", "b", aborted=True) == (
        out_dir / "record.b.aborted.json"
    )


def _populate_probe(root: Path, slug: str, name: str) -> Path:
    out_dir = bc.quality_probe_dir(root, slug, name)
    for cond in ("a", "b"):
        (out_dir / "frames" / cond).mkdir(parents=True)
        (out_dir / "frames" / cond / "step_step00.png").write_bytes(b"x")
        (out_dir / f"{cond}.png").write_bytes(cond.encode())
        bc.quality_probe_record_path(root, slug, name, cond).write_text(cond)
        bc.quality_probe_record_path(root, slug, name, cond, aborted=True).write_text(cond)
    return out_dir


def test_reset_quality_probe_outputs_retires_only_that_conditions_files(tmp_path: Path) -> None:
    """Bug: a condition-B probe run moves condition A's image, frames or record out of the NAME dir (the
    old whole-dir reset), so previewing TeaCache-on throws away the TeaCache-off image it is compared with;
    or B's own stale files survive and mix with the fresh run's outputs."""
    root, trash = tmp_path / "root", tmp_path / "trash"
    out_dir = _populate_probe(root, "z-image-base", "Z1")

    moved = bc.reset_quality_probe_outputs(root, "z-image-base", "Z1", "b", trash=trash, tag="t")

    assert (out_dir / "a.png").read_bytes() == b"a"
    assert (out_dir / "frames" / "a" / "step_step00.png").exists()
    assert (out_dir / "record.json").read_text() == "a"
    assert (out_dir / "record.aborted.json").read_text() == "a"
    for gone in ("b.png", "frames/b", "record.b.json", "record.b.aborted.json"):
        assert not (out_dir / gone).exists(), gone
    assert len(moved) == 4 and all(p.parent == trash and p.exists() for p in moved)


def test_reset_quality_probe_outputs_for_a_leaves_b_alone_and_is_a_no_op_when_empty(tmp_path: Path) -> None:
    """Bug: the per-condition reset ignores its condition argument (always retires "a", or both), or it
    fails on a probe NAME that has never run."""
    root, trash = tmp_path / "root", tmp_path / "trash"
    assert bc.reset_quality_probe_outputs(root, "z-image-base", "Z1", "a", trash=trash, tag="t0") == []
    out_dir = _populate_probe(root, "z-image-base", "Z1")

    bc.reset_quality_probe_outputs(root, "z-image-base", "Z1", "a", trash=trash, tag="t1")

    assert not (out_dir / "a.png").exists() and not (out_dir / "record.json").exists()
    assert (out_dir / "b.png").read_bytes() == b"b"
    assert (out_dir / "record.b.json").read_text() == "b"


def test_quality_probe_record_carries_name_overrides_prompt_negative_and_peaks() -> None:
    """Bug: the record drops one of the required fields (name, the resolved-recipe stamp, overrides, prompt,
    negative_prompt, negative_used, width/height/steps/guidance/quantize, generation seconds, peaks, memory
    phases, min host free), so a candidate setting can't actually be judged from it."""
    recipe = cr.apply_overrides(cr.recipe_for("flux1-dev"), guidance=7.0, prompt="a red bicycle")
    result = {
        "generation_seconds": 12.5,
        "mlx_peak_load_bytes": 1 * GIB,
        "mlx_peak_encode_bytes": 2 * GIB,
        "mlx_peak_generation_bytes": 3 * GIB,
        "memory": {
            "peak_resident_bytes": 4 * GIB,
            "peak_footprint_bytes": 5 * GIB,
            "min_host_free_pct": 42.0,
            "phases": {"load": {"peak_resident_bytes": 1}},
        },
    }
    record = bc.quality_probe_record(
        "tighter-guidance",
        recipe,
        overrides={"guidance": 7.0, "prompt": "a red bicycle"},
        result=result,
        versions=V,
    )
    assert record["name"] == "tighter-guidance"
    assert record["slug"] == "flux1-dev"
    assert record["stamp"] == cr.recipe_stamp(recipe, versions=V)
    assert record["overrides"] == {"guidance": 7.0, "prompt": "a red bicycle"}
    assert (record["width"], record["height"], record["steps"]) == (recipe.width, recipe.height, recipe.steps)
    assert (record["guidance"], record["quantize"]) == (7.0, recipe.quantize)
    assert record["prompt"] == "a red bicycle"
    assert record["negative_prompt"] is None
    assert record["negative_used"] is False
    assert record["generation_seconds"] == 12.5
    assert record["min_host_free_pct"] == 42.0
    assert record["memory_phases"] == {"load": {"peak_resident_bytes": 1}}
    assert record["peaks"] == {
        "mlx_peak_load_bytes": 1 * GIB,
        "mlx_peak_encode_bytes": 2 * GIB,
        "mlx_peak_generation_bytes": 3 * GIB,
        "peak_resident_bytes": 4 * GIB,
        "peak_footprint_bytes": 5 * GIB,
    }


def test_quality_probe_record_carries_the_condition_and_b_skip_telemetry() -> None:
    """Bug: a condition-B probe record doesn't say it is B, or drops the skip count / pattern / threshold,
    so a TeaCache-on face can't be tied to how many steps were actually skipped."""
    recipe = cr.recipe_for("z-image-base")
    base = {
        "generation_seconds": 1.0,
        "mlx_peak_load_bytes": 0,
        "mlx_peak_encode_bytes": 0,
        "mlx_peak_generation_bytes": 0,
        "memory": {"peak_resident_bytes": 0, "peak_footprint_bytes": 0, "min_host_free_pct": 50.0},
    }
    b_result = {
        **base,
        "condition": "b",
        "rel_l1_thresh": 0.25,
        "skipped": 14,
        "computed": 36,
        "max_consecutive_skips": 1,
        "skip_pattern": "CCSC",
    }
    b = bc.quality_probe_record("Z1", recipe, overrides={}, result=b_result, versions=V)
    assert b["condition"] == "b"
    assert b["teacache"] == {
        "rel_l1_thresh": 0.25,
        "skipped": 14,
        "computed": 36,
        "max_consecutive_skips": 1,
        "skip_pattern": "CCSC",
    }
    a = bc.quality_probe_record("Z1", recipe, overrides={}, result={**base, "condition": "a"}, versions=V)
    assert a["condition"] == "a"
    assert a["teacache"] is None


def test_quality_probe_record_carries_b_per_step_gate_trace() -> None:
    """Bug: the probe record keeps only the S/C pattern, so a skip that happens at every threshold cannot be
    traced back to the rel_l1 the gate measured on that step without re-running the model."""
    recipe = cr.recipe_for("klein-base-4b")
    base = {
        "generation_seconds": 1.0,
        "mlx_peak_load_bytes": 0,
        "mlx_peak_encode_bytes": 0,
        "mlx_peak_generation_bytes": 0,
        "memory": {"peak_resident_bytes": 0, "peak_footprint_bytes": 0, "min_host_free_pct": 50.0},
    }
    trace = [
        {"step": 0, "timestep": 1000.0, "kind": "forced", "rel_l1": None, "accumulated_distance": 0.0},
        {"step": 1, "timestep": 997.2, "kind": "skipped", "rel_l1": 0.0071, "accumulated_distance": 0.049},
    ]
    b_result = {
        **base,
        "condition": "b",
        "rel_l1_thresh": 0.08,
        "skipped": 1,
        "computed": 1,
        "max_consecutive_skips": 1,
        "skip_pattern": "CS",
        "gate_trace": trace,
    }
    b = bc.quality_probe_record("K1t08", recipe, overrides={}, result=b_result, versions=V)
    assert b["gate_trace"] == [
        {"step": 0, "timestep": 1000.0, "kind": "forced", "rel_l1": None, "accumulated_distance": 0.0},
        {"step": 1, "timestep": 997.2, "kind": "skipped", "rel_l1": 0.0071, "accumulated_distance": 0.049},
    ]
    a = bc.quality_probe_record("K1t08", recipe, overrides={}, result={**base, "condition": "a"}, versions=V)
    assert a["gate_trace"] is None


def test_quality_probe_record_negative_used_true_for_a_real_negative_that_reaches_the_model() -> None:
    """Bug: negative_used is hardcoded False, or true even when mflux would drop the negative."""
    recipe = cr.apply_overrides(cr.recipe_for("z-image-base"), negative_prompt="watermark")
    result = {
        "generation_seconds": 1.0,
        "mlx_peak_load_bytes": 0,
        "mlx_peak_encode_bytes": 0,
        "mlx_peak_generation_bytes": 0,
        "memory": {"peak_resident_bytes": 0, "peak_footprint_bytes": 0, "min_host_free_pct": 50.0},
    }
    record = bc.quality_probe_record("n", recipe, overrides={}, result=result, versions=V)
    assert record["negative_used"] is True
    assert record["memory_phases"] == {}  # absent "phases" key defaults to {}, not a KeyError


def test_quality_probe_record_negative_used_false_at_guidance_one_even_with_a_real_negative() -> None:
    """Bug: negative_used stays True at guidance 1.0, where mflux's own > 1.0 check (and Qwen's own neg +
    guidance * (pos - neg) combine, a no-op at guidance 1.0) means the negative never reaches the image."""
    recipe = cr.apply_overrides(cr.recipe_for("z-image-base"), negative_prompt="watermark", guidance=1.0)
    result = {
        "generation_seconds": 1.0,
        "mlx_peak_load_bytes": 0,
        "mlx_peak_encode_bytes": 0,
        "mlx_peak_generation_bytes": 0,
        "memory": {"peak_resident_bytes": 0, "peak_footprint_bytes": 0, "min_host_free_pct": 50.0},
    }
    record = bc.quality_probe_record("n", recipe, overrides={}, result=result, versions=V)
    assert record["negative_used"] is False


def test_max_workers_must_be_positive() -> None:
    """Bug: --max-workers 0 makes every invocation exit 3 and run-units re-invokes forever."""
    import argparse

    assert bc.positive_int("1") == 1
    with pytest.raises(argparse.ArgumentTypeError):
        bc.positive_int("0")


def test_smoke_recipe_leaves_a_step_outside_teacaches_always_computed_window() -> None:
    """Bug: smoke steps leave no step outside TeaCache's always-computed first/last window, so condition
    B raises."""
    smoke = bc._smoke_recipe(cr.recipe_for("flux1-dev"))
    assert smoke.steps >= 3
    assert smoke.width == 256 and smoke.height == 256
    assert smoke.free_encoders is True


def test_soft_cap_gb_is_capped_by_the_working_set_when_wired_plus_one_would_exceed_it() -> None:
    """Bug: soft_gb = wired_cap_gb + 1 alone puts Z-Image's soft cap (25 GiB) above the 24.96 GiB working set
    on this machine, defeating the point of a soft memory guideline."""
    working_set_bytes = int(24.96 * GIB)
    got = bc.soft_cap_gb(wired_cap_gb=24, cache_gb=2.0, working_set_bytes=working_set_bytes)
    assert got == pytest.approx(24.96 - 2)
    assert got < 24 + 1  # strictly under the naive wired_cap_gb + 1


def test_soft_cap_gb_uses_wired_plus_one_when_the_working_set_has_headroom() -> None:
    """Bug: the working-set term always wins, so soft_gb never actually reflects wired_cap_gb + 1 on a
    machine with a generous working set."""
    got = bc.soft_cap_gb(wired_cap_gb=22, cache_gb=2.0, working_set_bytes=40 * GIB)
    assert got == pytest.approx(23)


def test_probe_record_from_worker_maps_phase_peaks_and_footprint(tmp_path: Path) -> None:
    """Bug: the mapping from a worker's raw result to a probe record drops the process footprint, or
    misreads one of the three phase peaks -- a load-only overflow must still fail the probe."""
    r = cr.recipe_for("klein-base-9b")
    result = {
        "mlx_peak_load_bytes": 23 * GIB,
        "mlx_peak_encode_bytes": 2 * GIB,
        "mlx_peak_generation_bytes": 3 * GIB,
        "memory": {"min_host_free_pct": 40.0, "peak_footprint_bytes": 20 * GIB, "phases": {}},
    }
    rec = bc.probe_record_from_worker(r, result, working_set_bytes=24 * GIB, versions=V)
    assert rec["pass"] is False  # load-only overflow: 23 + 2 (cache_gb) >= 24 GiB working set
    assert rec["active_peak_bytes"] == 23 * GIB
    assert rec["peak_footprint_bytes"] == 20 * GIB
    assert rec["memory"] == result["memory"]


def test_probe_record_from_worker_passes_a_measurement_within_every_budget(tmp_path: Path) -> None:
    """Bug: a genuinely-fitting measurement is still failed -- e.g. the footprint or phase-peak gate is
    inverted, so nothing can ever pass."""
    r = cr.recipe_for("klein-base-9b")  # cache_gb=2.0, working_set 24 GiB below
    result = {
        "mlx_peak_load_bytes": 15 * GIB,
        "mlx_peak_encode_bytes": 10 * GIB,
        "mlx_peak_generation_bytes": 12 * GIB,
        "memory": {"min_host_free_pct": 40.0, "peak_footprint_bytes": 20 * GIB, "phases": {}},
    }
    rec = bc.probe_record_from_worker(r, result, working_set_bytes=24 * GIB, versions=V)
    assert rec["pass"] is True


def test_probe_record_from_worker_treats_an_aborted_payload_as_a_failing_measurement() -> None:
    """Bug: an aborted worker payload (killed by the watchdog) is mistaken for a real measurement and can
    pass the probe."""
    r = cr.recipe_for("klein-base-9b")
    rec = bc.probe_record_from_worker(
        r, {"aborted": "active-memory watchdog"}, working_set_bytes=24 * GIB, versions=V
    )
    assert rec["pass"] is False
    assert rec["aborted"] == "active-memory watchdog"


def test_merge_probe_results_takes_the_worse_reading_per_phase_and_footprint() -> None:
    """Bug: B holds cached TeaCache residuals on top of everything A holds, so a probe judged on A alone
    misses the +0.5-0.8 GiB B actually measures; merging must keep the worse side per field."""
    a = {
        "mlx_peak_load_bytes": 23 * GIB,
        "mlx_peak_encode_bytes": 5 * GIB,
        "mlx_peak_generation_bytes": 6 * GIB,
        "memory": {"min_host_free_pct": 45.0, "peak_footprint_bytes": 18 * GIB},
    }
    b = {
        "mlx_peak_load_bytes": 10 * GIB,
        "mlx_peak_encode_bytes": 5 * GIB,
        "mlx_peak_generation_bytes": 20 * GIB,  # B's cached residual makes the generation peak worse
        "memory": {"min_host_free_pct": 30.0, "peak_footprint_bytes": 22 * GIB},
    }
    merged = bc.merge_probe_results(a, b)
    assert merged["mlx_peak_load_bytes"] == 23 * GIB  # A was worse here
    assert merged["mlx_peak_generation_bytes"] == 20 * GIB  # B was worse here
    assert merged["memory"]["peak_footprint_bytes"] == 22 * GIB
    assert merged["memory"]["min_host_free_pct"] == 30.0  # the lower (worse) of the two


def test_merge_probe_results_propagates_an_abort_from_either_side() -> None:
    """Bug: B aborting mid-probe is swallowed because the merge only ever looks at A's payload."""
    a_ok = {
        "mlx_peak_load_bytes": 1,
        "mlx_peak_encode_bytes": 1,
        "mlx_peak_generation_bytes": 1,
        "memory": {"min_host_free_pct": 50.0, "peak_footprint_bytes": 1},
    }
    b_aborted = {"aborted": "active-memory watchdog"}
    assert bc.merge_probe_results(a_ok, b_aborted) == b_aborted
    assert bc.merge_probe_results(b_aborted, a_ok) == b_aborted


def test_merged_probe_result_can_fail_a_probe_that_condition_a_alone_would_pass() -> None:
    """Bug: the probe wiring still judges on A alone even though a merge helper exists -- this pins the
    end-to-end shape (merge_probe_results feeding probe_record_from_worker) the K ruling requires."""
    r = cr.recipe_for("klein-base-9b")
    a = {
        "mlx_peak_load_bytes": 15 * GIB,
        "mlx_peak_encode_bytes": 10 * GIB,
        "mlx_peak_generation_bytes": 12 * GIB,
        "memory": {"min_host_free_pct": 40.0, "peak_footprint_bytes": 20 * GIB},
    }
    b = {
        "mlx_peak_load_bytes": 15 * GIB,
        "mlx_peak_encode_bytes": 10 * GIB,
        "mlx_peak_generation_bytes": 23 * GIB,  # B's residual tips this over the 24 GiB working set
        "memory": {"min_host_free_pct": 40.0, "peak_footprint_bytes": 20 * GIB},
    }
    rec = bc.probe_record_from_worker(r, bc.merge_probe_results(a, b), working_set_bytes=24 * GIB, versions=V)
    assert rec["pass"] is False
    assert rec["active_peak_bytes"] == 23 * GIB


def _fake_versions() -> dict[str, str]:
    return {"mflux": "0.20.0", "mlx": "0.32.2", "mlx_taef": "0.8.3"}


def _fake_spawn_writes_chunk(raw: Path):  # noqa: ANN201
    def spawn(recipe: cr.Recipe, condition: str, *, probe: bool, smoke: bool) -> dict:
        raw_dir = bc.raw_dir_for(raw, recipe.slug)
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{condition}.png").write_bytes(b"x")
        frames = bc.frames_dir_for(raw, recipe.slug, condition)
        frames.mkdir(parents=True, exist_ok=True)
        for i in range(recipe.steps):
            (frames / f"step_step{i:02d}.png").write_bytes(b"x")
        return {
            "condition": condition,
            "width": recipe.width,
            "height": recipe.height,
            "stamp": cr.recipe_stamp(recipe, versions=_fake_versions()),
        }

    return spawn


def test_orchestrate_persists_a_chunk_per_budgeted_call_and_reports_what_remains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: a chunk is reported complete before its PNG/frames exist, or a budget of 1 never lets a second
    call finish the run (exit code stays 3 forever)."""
    monkeypatch.setattr(bc, "_versions", _fake_versions)
    chunks, raw, trash = tmp_path / "chunks", tmp_path / "raw", tmp_path / "trash"
    spawn = _fake_spawn_writes_chunk(raw)

    first = bc._orchestrate(
        "flux1-dev", budget=1, smoke=False, spawn=spawn, trash=trash, chunks_dir=chunks, raw_dir=raw
    )
    assert first == 3
    assert bc.chunk_path(chunks, "flux1-dev", "a").exists()
    assert not bc.chunk_path(chunks, "flux1-dev", "b").exists()

    second = bc._orchestrate(
        "flux1-dev", budget=-1, smoke=False, spawn=spawn, trash=trash, chunks_dir=chunks, raw_dir=raw
    )
    assert second == 0
    assert bc.chunk_path(chunks, "flux1-dev", "b").exists()


def test_orchestrate_records_an_abort_and_persists_no_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: an aborted worker payload is mistaken for a real chunk and persisted, or no .aborted.json
    artifact is written for the caller to inspect."""
    monkeypatch.setattr(bc, "_versions", _fake_versions)
    chunks, raw, trash = tmp_path / "chunks", tmp_path / "raw", tmp_path / "trash"

    def spawn(recipe: cr.Recipe, condition: str, *, probe: bool, smoke: bool) -> dict:
        return {"aborted": "active-memory watchdog"}

    result = bc._orchestrate(
        "flux1-dev", budget=-1, smoke=False, spawn=spawn, trash=trash, chunks_dir=chunks, raw_dir=raw
    )
    assert result == 4
    assert bc.chunk_path(chunks, "flux1-dev", "a").with_suffix(".aborted.json").exists()
    assert not bc.chunk_path(chunks, "flux1-dev", "a").exists()


def test_orchestrate_retires_a_stale_aborted_marker_before_respawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: a stale <cond>.aborted.json from a previous killed run survives a successful retry, so anything
    globbing for abort artifacts (the heavy-runs ABORT_GLOB) still sees this run as aborted."""
    monkeypatch.setattr(bc, "_versions", _fake_versions)
    chunks, raw, trash = tmp_path / "chunks", tmp_path / "raw", tmp_path / "trash"
    stale = bc.chunk_path(chunks, "flux1-dev", "a").with_suffix(".aborted.json")
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    spawn = _fake_spawn_writes_chunk(raw)

    result = bc._orchestrate(
        "flux1-dev", budget=-1, smoke=False, spawn=spawn, trash=trash, chunks_dir=chunks, raw_dir=raw
    )

    assert result == 0
    assert not stale.exists()
    assert any(trash.iterdir())


def test_orchestrate_refuses_a_mismatched_stamp_and_persists_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: a worker result measured under a different recipe/version is written to the chunk file instead
    of being refused."""
    monkeypatch.setattr(bc, "_versions", _fake_versions)
    chunks, raw, trash = tmp_path / "chunks", tmp_path / "raw", tmp_path / "trash"

    def spawn(recipe: cr.Recipe, condition: str, *, probe: bool, smoke: bool) -> dict:
        stamp = {**cr.recipe_stamp(recipe, versions=_fake_versions()), "width": 1}
        return {"condition": condition, "width": recipe.width, "height": recipe.height, "stamp": stamp}

    with pytest.raises(SystemExit, match="width"):
        bc._orchestrate(
            "flux1-dev", budget=-1, smoke=False, spawn=spawn, trash=trash, chunks_dir=chunks, raw_dir=raw
        )
    assert not bc.chunk_path(chunks, "flux1-dev", "a").exists()


def test_quality_probe_name_rejects_path_traversal_and_accepts_plain_names() -> None:
    """Bug: an unvalidated NAME lets --quality-probe ../x (or a/b, or a bare "." / ".." that resolves to the
    current/parent directory through quality_probe_dir's plain path join) escape QUALITY_PROBE_ROOT, writing
    or Trash-moving outside tests/_artifacts/quality_probe/."""
    with pytest.raises(argparse.ArgumentTypeError):
        bc.quality_probe_name("../x")
    with pytest.raises(argparse.ArgumentTypeError):
        bc.quality_probe_name("a/b")
    with pytest.raises(argparse.ArgumentTypeError):
        bc.quality_probe_name("")
    with pytest.raises(argparse.ArgumentTypeError):
        bc.quality_probe_name(".")
    with pytest.raises(argparse.ArgumentTypeError):
        bc.quality_probe_name("..")
    with pytest.raises(argparse.ArgumentTypeError):
        bc.quality_probe_name("...")
    assert bc.quality_probe_name("Z1") == "Z1"
    assert bc.quality_probe_name("K1t12") == "K1t12"
    assert bc.quality_probe_name("tighter-guidance") == "tighter-guidance"
    assert bc.quality_probe_name("a.b") == "a.b"
    assert bc.quality_probe_name("tighter-guidance_v2.1") == "tighter-guidance_v2.1"


def test_read_override_file_rejects_empty_and_strips_whitespace(tmp_path: Path) -> None:
    """Bug: a prompt/negative file that is empty (or all whitespace) after stripping is silently applied as
    an empty-string override instead of being rejected with a clear message."""
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n\t")
    with pytest.raises(SystemExit, match="empty"):
        bc._read_override_file(empty, label="prompt")

    real = tmp_path / "real.txt"
    real.write_text("  a bicycle in the rain  \n")
    assert bc._read_override_file(real, label="prompt") == "a bicycle in the rain"


def _probe_overrides_ns(**over: object) -> argparse.Namespace:
    base: dict[str, object] = dict(
        width=None,
        height=None,
        guidance=None,
        quantize=None,
        rel_l1_thresh=None,
        prompt_file=None,
        negative_file=None,
    )
    return argparse.Namespace(**{**base, **over})


def test_probe_overrides_reads_and_strips_prompt_and_negative_files_without_swapping_them(
    tmp_path: Path,
) -> None:
    """Bug: prompt and negative_prompt get swapped while assembling the overrides dict, so a probe's real
    prompt is silently rendered as its negative prompt (and vice versa)."""
    p = tmp_path / "p.txt"
    p.write_text("  a red bicycle \n")
    n = tmp_path / "n.txt"
    n.write_text("blurry\n")
    args = _probe_overrides_ns(width=512, rel_l1_thresh=0.12, prompt_file=p, negative_file=n)
    assert bc._probe_overrides(args) == {
        "width": 512,
        "rel_l1_thresh": 0.12,
        "prompt": "a red bicycle",
        "negative_prompt": "blurry",
    }


def test_probe_overrides_with_nothing_given_returns_an_empty_dict() -> None:
    """Bug: an override the caller never set (e.g. width defaulting to something other than None) still ends
    up in the dict handed to apply_overrides."""
    assert bc._probe_overrides(_probe_overrides_ns()) == {}


def test_quantize_choices_are_restricted_to_supported_bit_depths(capsys: pytest.CaptureFixture[str]) -> None:
    """Bug: --quantize accepts any int (e.g. 7), which mlx-lm/mflux quantization doesn't support."""
    parser = bc._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--only", "flux1-dev", "--quantize", "7"])
    capsys.readouterr()
    ns = parser.parse_args(["--only", "flux1-dev", "--quantize", "4"])
    assert ns.quantize == 4


def test_quality_probe_is_mutually_exclusive_with_probe_smoke_and_finalize(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bug: --quality-probe can be combined with --probe/--smoke/--finalize on one command line, silently
    running the wrong mode instead of refusing the ambiguous combination."""
    parser = bc._build_parser()
    for other in ("--probe", "--smoke", "--finalize"):
        with pytest.raises(SystemExit):
            parser.parse_args(["--only", "flux1-dev", "--quality-probe", "n", other])
        capsys.readouterr()
    # sanity: --quality-probe alone parses fine
    ns = parser.parse_args(["--only", "flux1-dev", "--quality-probe", "n"])
    assert ns.quality_probe == "n"


def test_quality_probe_records_an_abort_and_returns_exit_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: a watchdog abort during --quality-probe is thrown away as a bare RuntimeError (and the process
    exits non-zero with no artifact) instead of being recorded like a regular worker's chunk abort, with the
    same exit-4 convention _orchestrate uses."""
    root = tmp_path / "qp"
    monkeypatch.setattr(bc, "QUALITY_PROBE_ROOT", root)

    def fake_run_worker(cmd: list[str], label: str) -> dict:
        return {"aborted": "active-memory watchdog"}

    args = argparse.Namespace(
        only="flux1-dev",
        quality_probe="tighter-guidance",
        condition="a",
        width=None,
        height=None,
        guidance=None,
        quantize=None,
        prompt_file=None,
        negative_file=None,
        rel_l1_thresh=None,
    )
    result = bc._quality_probe(args, run_worker=fake_run_worker)

    assert result == 4
    record = bc.quality_probe_dir(root, "flux1-dev", "tighter-guidance") / "record.aborted.json"
    assert json.loads(record.read_text()) == {"aborted": "active-memory watchdog"}


def test_quality_probe_returns_zero_and_writes_no_abort_file_on_a_normal_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: a normal (non-aborted) worker result is mistaken for a failure, or a spurious abort file is
    written even though nothing aborted."""
    root = tmp_path / "qp"
    monkeypatch.setattr(bc, "QUALITY_PROBE_ROOT", root)

    def fake_run_worker(cmd: list[str], label: str) -> dict:
        return {"name": "tighter-guidance"}

    args = argparse.Namespace(
        only="flux1-dev",
        quality_probe="tighter-guidance",
        condition="a",
        width=None,
        height=None,
        guidance=None,
        quantize=None,
        prompt_file=None,
        negative_file=None,
        rel_l1_thresh=None,
    )
    result = bc._quality_probe(args, run_worker=fake_run_worker)

    assert result == 0
    assert not (bc.quality_probe_dir(root, "flux1-dev", "tighter-guidance") / "record.aborted.json").exists()


def test_quality_probe_forwards_only_the_overrides_actually_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: an override the user never set on the command line (e.g. width) is still forwarded to the worker
    with some default/None-derived value, making the worker's record look like the user asked for something
    they didn't."""
    root = tmp_path / "qp"
    monkeypatch.setattr(bc, "QUALITY_PROBE_ROOT", root)
    seen: dict[str, list[str]] = {}

    def fake_run_worker(cmd: list[str], label: str) -> dict:
        seen["cmd"] = cmd
        return {"name": "n"}

    args = argparse.Namespace(
        only="flux1-dev",
        quality_probe="n",
        condition="a",
        width=512,
        height=None,
        guidance=7.0,
        quantize=None,
        prompt_file=None,
        negative_file=None,
        rel_l1_thresh=None,
    )
    bc._quality_probe(args, run_worker=fake_run_worker)

    cmd = seen["cmd"]
    assert cmd[cmd.index("--width") + 1] == "512"
    assert cmd[cmd.index("--guidance") + 1] == "7.0"
    assert "--height" not in cmd
    assert "--quantize" not in cmd


def _qp_args(**kw: object) -> argparse.Namespace:
    base: dict[str, object] = dict(
        only="z-image-base",
        quality_probe="Z1",
        condition="a",
        width=None,
        height=None,
        guidance=None,
        quantize=None,
        prompt_file=None,
        negative_file=None,
        rel_l1_thresh=None,
    )
    return argparse.Namespace(**{**base, **kw})


def test_quality_probe_forwards_the_condition_to_the_worker() -> None:
    """Bug: --quality-probe drops --condition, so asking for a TeaCache-on (B) preview silently renders
    condition A again."""
    seen: list[list[str]] = []

    def fake_run_worker(cmd: list[str], label: str) -> dict:
        seen.append(cmd)
        return {"name": "Z1"}

    bc._quality_probe(_qp_args(condition="b"), run_worker=fake_run_worker)
    bc._quality_probe(_qp_args(condition="a"), run_worker=fake_run_worker)

    b_cmd, a_cmd = seen
    assert b_cmd[b_cmd.index("--condition") + 1] == "b"
    assert a_cmd[a_cmd.index("--condition") + 1] == "a"


def test_quality_probe_refuses_a_negative_file_for_flux1_before_spawning_the_worker(tmp_path: Path) -> None:
    """Bug: FLUX.1/Krea's --negative-file reaches the worker (which then raises deep inside
    precompute_prompt only after the whole model has loaded) instead of being refused up front, before any
    worker subprocess is spawned."""
    calls: list[list[str]] = []

    def fake_run_worker(cmd: list[str], label: str) -> dict:
        calls.append(cmd)
        return {"name": "n"}

    neg = tmp_path / "n.txt"
    neg.write_text("blurry")

    with pytest.raises(SystemExit):
        bc._quality_probe(_qp_args(only="flux1-dev", negative_file=neg), run_worker=fake_run_worker)
    assert calls == []

    bc._quality_probe(_qp_args(only="z-image-base", negative_file=neg), run_worker=fake_run_worker)
    assert len(calls) == 1


def test_quality_probe_b_abort_is_recorded_without_touching_as_abort_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: a condition-B watchdog abort is written to record.aborted.json, which is condition A's file, so
    a B abort reads as if A had aborted (or overwrites A's real abort record)."""
    root = tmp_path / "qp"
    monkeypatch.setattr(bc, "QUALITY_PROBE_ROOT", root)

    def fake_run_worker(cmd: list[str], label: str) -> dict:
        return {"aborted": "active-memory watchdog"}

    assert bc._quality_probe(_qp_args(condition="b"), run_worker=fake_run_worker) == 4

    out_dir = bc.quality_probe_dir(root, "z-image-base", "Z1")
    assert json.loads((out_dir / "record.b.aborted.json").read_text()) == {
        "aborted": "active-memory watchdog"
    }
    assert not (out_dir / "record.aborted.json").exists()


def test_quality_probe_forwards_rel_l1_thresh_only_when_given() -> None:
    """Bug: --rel-l1-thresh is dropped on the way to the worker (B runs at the default threshold while the
    probe is named for a lower one), or a threshold the user never set is forwarded."""
    seen: list[list[str]] = []

    def fake_run_worker(cmd: list[str], label: str) -> dict:
        seen.append(cmd)
        return {"name": "Z1"}

    bc._quality_probe(_qp_args(condition="b", rel_l1_thresh=0.12), run_worker=fake_run_worker)
    bc._quality_probe(_qp_args(condition="b", rel_l1_thresh=None), run_worker=fake_run_worker)

    with_t, without_t = seen
    assert with_t[with_t.index("--rel-l1-thresh") + 1] == "0.12"
    assert "--rel-l1-thresh" not in without_t


def test_rel_l1_thresh_flag_parses_as_a_float() -> None:
    """Bug: --rel-l1-thresh is missing from the parser or parsed as a string."""
    ns = bc._build_parser().parse_args(
        ["--only", "klein-base-4b", "--quality-probe", "t", "--rel-l1-thresh", "0.1"]
    )
    assert ns.rel_l1_thresh == 0.1


def _probe_only_ns(**over: object) -> argparse.Namespace:
    base: dict[str, object] = dict(
        quality_probe=None,
        condition="a",
        guidance=None,
        quantize=None,
        rel_l1_thresh=None,
        prompt_file=None,
        negative_file=None,
    )
    return argparse.Namespace(**{**base, **over})


def test_probe_only_flags_misused_is_empty_when_nothing_is_set() -> None:
    """Bug: the misuse check fires even when the caller set none of the probe-only flags."""
    assert bc.probe_only_flags_misused(_probe_only_ns()) == []


def test_probe_only_flags_misused_names_a_probe_only_flag_given_without_the_probe() -> None:
    """Bug: --rel-l1-thresh (or another probe-only flag) silently does nothing when given without
    --quality-probe, instead of being refused and named."""
    assert bc.probe_only_flags_misused(_probe_only_ns(rel_l1_thresh=0.1)) == ["--rel-l1-thresh"]


def test_probe_only_flags_misused_names_every_probe_only_flag() -> None:
    """Bug: one of the five probe-only flags is missing from the list, so it silently does nothing again
    outside --quality-probe."""
    ns = _probe_only_ns(
        guidance=7.0, quantize=4, rel_l1_thresh=0.1, prompt_file="p.txt", negative_file="n.txt"
    )
    assert bc.probe_only_flags_misused(ns) == [
        "--guidance",
        "--quantize",
        "--rel-l1-thresh",
        "--prompt-file",
        "--negative-file",
    ]


def test_probe_only_flags_misused_is_empty_when_the_probe_is_also_given() -> None:
    """Bug: a legitimate --quality-probe X --rel-l1-thresh 0.1 combination is refused."""
    assert bc.probe_only_flags_misused(_probe_only_ns(quality_probe="X", rel_l1_thresh=0.1)) == []


def test_parse_and_validate_refuses_a_probe_only_flag_given_without_quality_probe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bug: a probe-only flag without --quality-probe parses fine and silently does nothing (the orchestrator
    runs the recipe's defaults) instead of being refused up front. --guidance is used because the separate
    condition-A rule never looks at it, so only the misuse check can refuse this command."""
    with pytest.raises(SystemExit):
        bc._parse_and_validate(["--only", "z-image-base", "--guidance", "7"])
    assert "requires --quality-probe" in capsys.readouterr().err


def test_parse_and_validate_refuses_rel_l1_thresh_on_condition_a(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bug: --quality-probe X --rel-l1-thresh 0.1 on condition A (the default) is accepted even though
    condition A never applies TeaCache, so the threshold could never take effect."""
    with pytest.raises(SystemExit):
        bc._parse_and_validate(["--only", "z-image-base", "--quality-probe", "X", "--rel-l1-thresh", "0.1"])
    capsys.readouterr()


def test_parse_and_validate_allows_rel_l1_thresh_on_condition_b() -> None:
    """Bug: the condition-A refusal is too broad and also rejects the legitimate condition-B case."""
    ns = bc._parse_and_validate(
        ["--only", "z-image-base", "--quality-probe", "X", "--rel-l1-thresh", "0.1", "--condition", "b"]
    )
    assert ns.condition == "b" and ns.rel_l1_thresh == 0.1


def test_quality_probe_forwards_prompt_and_negative_files_and_the_command_round_trips(
    tmp_path: Path,
) -> None:
    """Bug: --prompt-file/--negative-file are dropped on the way to the worker, or the forwarded command line
    doesn't actually parse back to the same values (a quoting/ordering bug that would only surface once a
    real subprocess re-parses argv)."""
    p, n = tmp_path / "p.txt", tmp_path / "n.txt"
    p.write_text("a red bicycle")
    n.write_text("blurry")
    seen: list[list[str]] = []

    def fake_run_worker(cmd: list[str], label: str) -> dict:
        seen.append(cmd)
        return {"name": "Z1"}

    args = _qp_args(condition="b", prompt_file=p, negative_file=n, rel_l1_thresh=0.12, width=512)
    bc._quality_probe(args, run_worker=fake_run_worker)

    cmd = seen[0]
    assert cmd[cmd.index("--prompt-file") + 1] == str(p)
    assert cmd[cmd.index("--negative-file") + 1] == str(n)

    ns = bc._build_parser().parse_args(
        cmd[2:]
    )  # cmd[2:]: skip the interpreter and script path, keep --worker
    assert ns.prompt_file == p
    assert ns.negative_file == n
    assert ns.condition == "b"
    assert ns.rel_l1_thresh == 0.12
    assert ns.width == 512
    assert ns.quality_probe == "Z1"
    assert ns.only == "z-image-base"


def test_run_worker_treats_a_sigabrt_exit_with_no_result_line_as_an_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug: a worker killed by macOS's GPU watchdog (SIGABRT, no parseable result line) is thrown away as a
    bare RuntimeError instead of being recorded as an abort, the way a real memory-watchdog abort already
    is."""
    monkeypatch.setattr(bc, "_stream_worker", lambda cmd: (-6, "no result line\n"))
    result = bc._run_worker(["x"], "flux1-dev/a")
    assert "aborted" in result
    assert result["returncode"] == -6


def test_run_worker_still_raises_on_a_plain_non_zero_exit_with_no_result_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug: the new SIGABRT handling swallows an ordinary crash too, hiding a real failure behind a fake
    "aborted" record instead of raising."""
    monkeypatch.setattr(bc, "_stream_worker", lambda cmd: (1, ""))
    with pytest.raises(RuntimeError):
        bc._run_worker(["x"], "flux1-dev/a")


def test_run_worker_raises_for_a_sigkill_rather_than_calling_it_the_gpu_time_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug: the SIGABRT check widens to any signal death, so a jetsam SIGKILL (-9) is recorded as "likely the
    macOS GPU time limit" instead of raising."""
    monkeypatch.setattr(bc, "_stream_worker", lambda cmd: (-9, ""))
    with pytest.raises(RuntimeError):
        bc._run_worker(["x"], "flux1-dev/a")


def test_run_worker_treats_the_shell_form_of_sigabrt_as_an_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug: only -6 is recognised, so a worker whose SIGABRT reaches us through a shell (exit 134) raises
    instead of recording the abort."""
    monkeypatch.setattr(bc, "_stream_worker", lambda cmd: (134, "no result line\n"))
    payload = bc._run_worker(["x"], "flux1-dev/a")
    assert payload["returncode"] == 134 and "SIGABRT" in payload["aborted"]


_SIGABRT_REASON = "SIGABRT (likely the macOS GPU time limit: Impacting Interactivity)"


def test_orchestrate_prints_the_aborted_payloads_own_reason_not_a_hardcoded_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bug: the printed abort message always says "the memory watchdog" even when the payload's own
    ``aborted`` reason is something else (a SIGABRT from the macOS GPU time limit), misleading whoever reads
    the log about why the run stopped."""
    monkeypatch.setattr(bc, "_versions", _fake_versions)
    chunks, raw, trash = tmp_path / "chunks", tmp_path / "raw", tmp_path / "trash"

    def spawn(recipe: cr.Recipe, condition: str, *, probe: bool, smoke: bool) -> dict:
        return {"aborted": _SIGABRT_REASON}

    bc._orchestrate(
        "flux1-dev", budget=-1, smoke=False, spawn=spawn, trash=trash, chunks_dir=chunks, raw_dir=raw
    )
    assert _SIGABRT_REASON in capsys.readouterr().out


def test_quality_probe_prints_the_aborted_payloads_own_reason_not_a_hardcoded_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bug: the printed abort message on a quality-probe abort always says "the memory watchdog" even when
    the payload's own reason is a SIGABRT."""
    root = tmp_path / "qp"
    monkeypatch.setattr(bc, "QUALITY_PROBE_ROOT", root)

    def fake_run_worker(cmd: list[str], label: str) -> dict:
        return {"aborted": _SIGABRT_REASON}

    bc._quality_probe(_qp_args(condition="a"), run_worker=fake_run_worker)
    assert _SIGABRT_REASON in capsys.readouterr().out
