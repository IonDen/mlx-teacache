"""Pure helpers of the schema-2 comparison harness (pure-core lane: bench_comparison imports mflux lazily)."""

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
    e = bc.assemble_entry(r, a, b, ssim=0.97, provenance={"version_mflux": "0.20.0"})
    assert e["speedup_wall"] == pytest.approx(62.5 / 53.5)
    assert e["speedup_steady"] == pytest.approx(30.0 / 21.0)
    assert e["speedup_preview_subtracted"] == pytest.approx(60.5 / 51.5)
    assert e["b"]["step_medians"] == {"computed": 10.0, "skipped": 1.0}
    assert e["a"]["load_seconds"] == 12.0 and e["b"]["load_seconds"] == 13.0
    assert e["images"]["steps_b"] == "_artifacts/comparison/flux1-dev/steps-b.jpg"
    assert e["bench_report"] == r.bench_report and e["prompt_sha256"] == "p"


def test_assemble_entry_refuses_mismatched_step_counts() -> None:
    """Bug: A and B step counts silently differ, misaligning the per-step tables."""
    r = cr.recipe_for("flux1-dev")
    a = _result("a", [1.0, 1.0], [0.1, 0.1], gen=2.2)
    with pytest.raises(ValueError, match="step counts"):
        bc.assemble_entry(
            r, a, _result("b", [1.0], [0.1], gen=1.1, decision_kinds=["computed"]), ssim=1.0, provenance={}
        )


def test_assemble_entry_refuses_b_missing_decision_kinds() -> None:
    """Bug: B is treated as all-computed when its decision_kinds is missing (the KeyError must name
    decision_kinds itself, not an unrelated _B_KEYS entry that happens to be missing too)."""
    r = cr.recipe_for("flux1-dev")
    a = _result("a", [1.0, 1.0], [0.1, 0.1], gen=2.2)
    present = {k: 0 for k in bc._B_KEYS if k != "decision_kinds"}
    b = _result("b", [1.0, 1.0], [0.1, 0.1], gen=2.2, **present)
    with pytest.raises(KeyError, match="decision_kinds"):
        bc.assemble_entry(r, a, b, ssim=1.0, provenance={})


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
    assert rec["memory"] == result["memory"]


def test_probe_record_from_worker_treats_an_aborted_payload_as_a_failing_measurement() -> None:
    """Bug: an aborted worker payload (killed by the watchdog) is mistaken for a real measurement and can
    pass the probe."""
    r = cr.recipe_for("klein-base-9b")
    rec = bc.probe_record_from_worker(
        r, {"aborted": "active-memory watchdog"}, working_set_bytes=24 * GIB, versions=V
    )
    assert rec["pass"] is False
    assert rec["aborted"] == "active-memory watchdog"


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
