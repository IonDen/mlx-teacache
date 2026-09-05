"""Pure-helper unit tests for ``scripts/bench_speedup.py``.

Covers the per-chunk persistence + resume layer added for the v0.10.0 bench
re-run: every (condition, rep) worker result is written to disk the moment the
worker returns, and a re-invocation skips the chunks whose file already exists.
This is what lets one three-way bench be split into several short, finite
jobs (one per condition) instead of a single 60-90 min monolith.

``bench_speedup`` imports mflux / mlx only lazily (inside the worker), so this
module imports cleanly in the pure-core lane.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _bench_telemetry as bt  # noqa: E402
import bench_speedup as bs  # noqa: E402

_CONDITIONS = ["vanilla", "wrapper_nogate", "wrapper"]


def _fake_result(condition: str, rep: int, elapsed: float = 1.0) -> dict[str, object]:
    return {
        "variant": "flux1-dev",
        "condition": condition,
        "rep": rep,
        "elapsed_s": elapsed,
        "peak_memory_gb": 1.5,
        "stats_summary": {"skipped_count": 6, "computed_count": 19},
    }


# --- _chunk_path -----------------------------------------------------------


def test_chunk_path_is_condition_and_rep_keyed(tmp_path: Path) -> None:
    assert bs._chunk_path(tmp_path, "wrapper_nogate", 2) == tmp_path / "wrapper_nogate_rep2.json"


# --- _pending_chunks -------------------------------------------------------


def test_pending_chunks_lists_every_pair_in_run_order_when_nothing_persisted(tmp_path: Path) -> None:
    pending = bs._pending_chunks(_CONDITIONS, 2, tmp_path)
    assert pending == [
        ("vanilla", 0),
        ("wrapper_nogate", 0),
        ("wrapper", 0),
        ("vanilla", 1),
        ("wrapper_nogate", 1),
        ("wrapper", 1),
    ]


def test_pending_chunks_skips_pairs_whose_file_exists(tmp_path: Path) -> None:
    bs._persist_chunk(tmp_path, _fake_result("vanilla", 0))
    bs._persist_chunk(tmp_path, _fake_result("wrapper", 1))
    pending = bs._pending_chunks(_CONDITIONS, 2, tmp_path)
    assert ("vanilla", 0) not in pending
    assert ("wrapper", 1) not in pending
    assert pending == [("wrapper_nogate", 0), ("wrapper", 0), ("vanilla", 1), ("wrapper_nogate", 1)]


def test_pending_chunks_tolerates_missing_results_dir(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    assert bs._pending_chunks(["vanilla"], 1, missing) == [("vanilla", 0)]


# --- _persist_chunk / _load_chunks ----------------------------------------


def test_persist_chunk_round_trips_the_worker_result(tmp_path: Path) -> None:
    result = _fake_result("wrapper", 0, elapsed=83.25)
    written = bs._persist_chunk(tmp_path, result)
    assert written == bs._chunk_path(tmp_path, "wrapper", 0)
    assert json.loads(written.read_text()) == result


def test_persist_chunk_creates_the_results_dir(tmp_path: Path) -> None:
    nested = tmp_path / "bench_chunks" / "flux1-dev"
    bs._persist_chunk(nested, _fake_result("vanilla", 0))
    assert nested.is_dir()


def test_load_chunks_returns_none_while_any_chunk_is_missing(tmp_path: Path) -> None:
    bs._persist_chunk(tmp_path, _fake_result("vanilla", 0))
    bs._persist_chunk(tmp_path, _fake_result("vanilla", 1))
    bs._persist_chunk(tmp_path, _fake_result("wrapper", 0))
    assert bs._load_chunks(["vanilla", "wrapper"], 2, tmp_path) is None


def test_load_chunks_returns_results_keyed_by_condition_ordered_by_rep(tmp_path: Path) -> None:
    # Persist out of order to prove ordering comes from the rep index, not mtime.
    bs._persist_chunk(tmp_path, _fake_result("wrapper", 1, elapsed=71.0))
    bs._persist_chunk(tmp_path, _fake_result("vanilla", 1, elapsed=104.0))
    bs._persist_chunk(tmp_path, _fake_result("vanilla", 0, elapsed=103.0))
    bs._persist_chunk(tmp_path, _fake_result("wrapper", 0, elapsed=70.0))
    loaded = bs._load_chunks(["vanilla", "wrapper"], 2, tmp_path)
    assert loaded is not None
    assert [r["elapsed_s"] for r in loaded["vanilla"]] == [103.0, 104.0]
    assert [r["elapsed_s"] for r in loaded["wrapper"]] == [70.0, 71.0]


# --- _image_path_for -------------------------------------------------------


def test_image_path_only_for_rep_zero(tmp_path: Path) -> None:
    assert bs._image_path_for(tmp_path, "vanilla", 1, three_way=True) is None
    assert bs._image_path_for(tmp_path, "vanilla", 0, three_way=True) == tmp_path / "vanilla.png"


def test_image_path_names_follow_the_three_way_flag(tmp_path: Path) -> None:
    assert (
        bs._image_path_for(tmp_path, "wrapper_nogate", 0, three_way=True) == tmp_path / "wrapper_nogate.png"
    )
    assert bs._image_path_for(tmp_path, "wrapper", 0, three_way=True) == tmp_path / "wrapper_gated.png"
    assert bs._image_path_for(tmp_path, "wrapper", 0, three_way=False) == tmp_path / "wrapper.png"


# --- skip-streak telemetry -------------------------------------------------


def test_skip_pattern_marks_only_skipped_decisions_as_S() -> None:
    kinds = ["forced", "computed", "skipped", "skipped", "numerical-miss", "computed", "skipped"]
    assert bt.skip_pattern(kinds) == "CCSSCCS"


def test_max_skip_streak_is_the_longest_run_of_S() -> None:
    assert bt.max_skip_streak("CCSSCCS") == 2
    assert bt.max_skip_streak("CSSSSCSS") == 4
    assert bt.max_skip_streak("CCCC") == 0
    assert bt.max_skip_streak("") == 0


def test_streak_telemetry_reads_the_last_committed_generation() -> None:
    from mlx_teacache._kernel.stats import StepDecision, TeaCacheStats

    stats = TeaCacheStats()
    kinds = ["forced", "skipped", "skipped", "skipped", "computed", "skipped", "computed"]
    for i, kind in enumerate(kinds):
        stats.record(
            StepDecision(
                step_idx=i, timestep=1.0 - i / 10, rel_l1=0.1, accumulated_distance=0.2, decision=kind
            )  # type: ignore[arg-type]
        )
    stats.finalize_last_generation(num_inference_steps=len(kinds), cfg_was_active=False)
    assert bs._streak_telemetry(stats) == {"skip_pattern": "CSSSCSC", "max_consecutive_skips": 3}


def test_streak_telemetry_is_empty_before_any_committed_generation() -> None:
    from mlx_teacache._kernel.stats import TeaCacheStats

    assert bs._streak_telemetry(TeaCacheStats()) == {"skip_pattern": "", "max_consecutive_skips": 0}


def test_wrapper_streak_arrays_are_per_rep_in_order() -> None:
    results = [
        {**_fake_result("wrapper", 0), "stats_summary": {"skip_pattern": "CSSC", "max_consecutive_skips": 2}},
        {**_fake_result("wrapper", 1), "stats_summary": {"skip_pattern": "CSCS", "max_consecutive_skips": 1}},
    ]
    assert bs._wrapper_streak_arrays(results) == {
        "skip_patterns": ["CSSC", "CSCS"],
        "max_consecutive_skips": [2, 1],
    }


def test_wrapper_streak_arrays_tolerate_pre_telemetry_chunks() -> None:
    # Chunks persisted before the telemetry fields existed carry neither key.
    results = [_fake_result("wrapper", 0)]
    assert bs._wrapper_streak_arrays(results) == {"skip_patterns": [""], "max_consecutive_skips": [0]}


# --- recipe stamp: a persisted chunk must match the invocation that reuses it ---


def test_verify_chunk_recipes_accepts_matching_stamps(tmp_path: Path) -> None:
    bs._persist_chunk(tmp_path, {**_fake_result("vanilla", 0), "num_inference_steps": 25, "guidance": 3.5})
    bs._verify_chunk_recipes(["vanilla", "wrapper"], 1, tmp_path, num_inference_steps=25, guidance=3.5)


def test_verify_chunk_recipes_refuses_a_chunk_from_a_different_recipe(tmp_path: Path) -> None:
    bs._persist_chunk(tmp_path, {**_fake_result("vanilla", 0), "num_inference_steps": 20, "guidance": 3.5})
    with pytest.raises(SystemExit, match="vanilla_rep0.json.*num_inference_steps=20.*25"):
        bs._verify_chunk_recipes(["vanilla"], 1, tmp_path, num_inference_steps=25, guidance=3.5)


def test_verify_chunk_recipes_refuses_an_unstamped_chunk(tmp_path: Path) -> None:
    bs._persist_chunk(tmp_path, _fake_result("wrapper", 2))  # no recipe keys at all
    with pytest.raises(SystemExit, match="wrapper_rep2.json"):
        bs._verify_chunk_recipes(["wrapper"], 3, tmp_path, num_inference_steps=25, guidance=3.5)


# --- wired-cap clamp (shared scripts/_mlx_caps.py) --------------------------


def test_clamped_wired_bytes_keeps_a_request_below_the_device_ceiling() -> None:
    import _mlx_caps

    gib = 1024**3
    # 20 GB requested on a 32 GB M1 Max (recommended working set ~25 GB): the request wins.
    assert _mlx_caps.clamped_wired_bytes(20, 25 * gib) == 20 * gib


def test_clamped_wired_bytes_clamps_to_a_fraction_of_a_small_device_ceiling() -> None:
    import _mlx_caps

    gib = 1024**3
    # 22 GB requested on a 16 GB Mac (recommended ~10.6 GB): clamp to 85% of the ceiling, not raise.
    assert _mlx_caps.clamped_wired_bytes(22, int(10.6 * gib)) == int(int(10.6 * gib) * 0.85)


def test_clamped_wired_bytes_rejects_nonpositive_requests() -> None:
    import _mlx_caps

    with pytest.raises(ValueError):
        _mlx_caps.clamped_wired_bytes(0, 25 * 1024**3)


# --- Qwen-Image bench support -------------------------------------------------
# Qwen was absent from the bench harness entirely, so its max consecutive-skip
# streak had no committed source and docs/calibration.md carried an empty row.


def test_qwen_is_a_supported_bench_variant():
    """RED if the harness cannot bench Qwen at all (its slug is unmapped),
    which is why the v0.10.0 streak table had no qwen-image row."""
    assert bs._VARIANT_SLUG_TO_ID["qwen"] == "qwen-image"
    assert bs._VARIANT_RECIPE["qwen"] == {"num_inference_steps": 50, "guidance": 4.0}
    assert bs._VARIANT_QUANTIZE["qwen"] == 4


def test_qwen_benches_at_its_pinned_768_resolution():
    """RED if Qwen benched at the shared 512x512 recipe. Its DEFAULT_THRESH=0.30
    was calibrated and swept at 768x768, so a 512x512 bench would report a skip
    pattern for an operating point the threshold was never tuned against."""
    assert bs._resolution_for("qwen") == (768, 768)


@pytest.mark.parametrize("variant", ["flux1-dev", "klein-base-4b", "klein-base-9b", "z-image"])
def test_non_qwen_variants_keep_the_shared_512_recipe(variant):
    """RED if adding Qwen's override silently moved every other variant's
    resolution, which would invalidate every committed bench JSON."""
    assert bs._resolution_for(variant) == (512, 512)


# --- v0.10.1: rep-outer order, memory fields, abort artifact, git stamp ---


def test_pending_chunks_are_rep_outer_so_conditions_interleave(tmp_path: Path) -> None:
    # bug caught: condition-outer order (all vanilla, then all no-gate, then all gated)
    assert bs._pending_chunks(_CONDITIONS, 2, tmp_path) == [
        ("vanilla", 0),
        ("wrapper_nogate", 0),
        ("wrapper", 0),
        ("vanilla", 1),
        ("wrapper_nogate", 1),
        ("wrapper", 1),
    ]


def test_memory_fields_separate_load_and_loop_and_keep_the_legacy_peak() -> None:
    # bug caught: reporting the process-lifetime peak as the loop peak
    gib = 1024**3
    f = bs._memory_fields(load_peak_bytes=20 * gib, loop_peak_bytes=26 * gib, cache_bytes=gib)
    assert f == {
        "peak_memory_gb": 26.0,
        "load_peak_memory_gb": 20.0,
        "loop_peak_memory_gb": 26.0,
        "cache_memory_gb": 1.0,
    }
    g = bs._memory_fields(load_peak_bytes=26 * gib, loop_peak_bytes=20 * gib, cache_bytes=0)
    assert g["peak_memory_gb"] == 26.0  # legacy field is the max of the two


def test_abort_artifact_is_persisted_beside_the_chunk_and_never_counts_as_done(tmp_path: Path) -> None:
    # bug caught: writing the abort payload to the chunk path (a resume would reuse it as a result)
    payload = {"aborted": "active-memory watchdog", "condition": "wrapper", "rep": 1, "resident_bytes": 1}
    written = bs._persist_abort(tmp_path, payload)
    assert written == tmp_path / "wrapper_rep1.aborted.json"
    assert json.loads(written.read_text()) == payload
    assert ("wrapper", 1) in bs._pending_chunks(["wrapper"], 2, tmp_path)


def test_parse_worker_line_finds_the_sentinel_payload() -> None:
    line = f"noise\n{bs.WORKER_RESULT_SENTINEL}{json.dumps({'a': 1})}\nmore"
    assert bs._parse_worker_line(line) == {"a": 1}
    assert bs._parse_worker_line("no sentinel here") is None


def test_git_revision_reports_commit_and_dirty_flag(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        check=True,
    )
    rev = bs._git_revision(tmp_path)
    assert isinstance(rev["git_commit"], str) and len(rev["git_commit"]) == 40
    assert rev["git_dirty"] is False
    (tmp_path / "f").write_text("x")
    assert bs._git_revision(tmp_path)["git_dirty"] is True


def test_git_revision_outside_a_repo_is_none(tmp_path: Path) -> None:
    assert bs._git_revision(tmp_path) == {"git_commit": None, "git_dirty": None}
