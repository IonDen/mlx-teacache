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


def test_assemble_entry_refuses_mismatched_steps_and_b_without_decisions() -> None:
    """Bug: B treated as all-computed when its decisions are missing; A/B step counts differ."""
    r = cr.recipe_for("flux1-dev")
    a = _result("a", [1.0, 1.0], [0.1, 0.1], gen=2.2)
    with pytest.raises(ValueError, match="step counts"):
        bc.assemble_entry(
            r, a, _result("b", [1.0], [0.1], gen=1.1, decision_kinds=["computed"]), ssim=1.0, provenance={}
        )
    with pytest.raises(KeyError):
        bc.assemble_entry(r, a, _result("b", [1.0, 1.0], [0.1, 0.1], gen=2.2), ssim=1.0, provenance={})


def test_merge_entry_replaces_one_slug_and_keeps_the_rest() -> None:
    """Bug: finishing one model drops the others from the report."""
    report = {"schema_version": 2, "variants": {"flux1-dev": {"x": 1}, "z-image-base": {"y": 2}}}
    out = bc.merge_entry(report, "flux1-dev", {"x": 3}, generated_at="2026-09-25T10:00Z")
    assert out["variants"] == {"flux1-dev": {"x": 3}, "z-image-base": {"y": 2}}
    assert out["generated_at"] == "2026-09-25T10:00Z" and report["variants"]["flux1-dev"] == {"x": 1}
