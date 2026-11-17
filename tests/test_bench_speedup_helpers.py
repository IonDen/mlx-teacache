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
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
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


def test_memory_fields_schema_4() -> None:
    # bug caught: the ~0 load peak comes back (mflux materialises weights inside the first
    # generation, so a peak read right after the load measured almost nothing)
    gib = 1024**3
    f = bs._memory_fields(peak_bytes=26 * gib, cache_bytes=gib)
    assert f == {"peak_memory_gb": 26.0, "cache_memory_gb": 1.0}
    assert bs._memory_fields(peak_bytes=3 * gib // 2, cache_bytes=0) == {
        "peak_memory_gb": 1.5,
        "cache_memory_gb": 0.0,
    }


def _schema4_chunk(condition: str, rep: int, elapsed: float, peak: float, cache: float) -> dict[str, object]:
    return {
        "variant": "flux1-dev",
        "condition": condition,
        "rep": rep,
        "num_inference_steps": 25,
        "guidance": 3.5,
        "elapsed_s": elapsed,
        "started_at": "2026-11-10T10:00:00+00:00",
        "finished_at": "2026-11-10T10:01:00+00:00",
        "peak_memory_gb": peak,
        "cache_memory_gb": cache,
        "stats_summary": (
            {"skipped_count": 6, "computed_count": 19, "skip_pattern": "CCS", "max_consecutive_skips": 1}
            if condition != "vanilla"
            else {}
        ),
    }


_NO_START_READING = {
    "git_commit_at_start": None,
    "git_dirty_at_start": None,
    "git_dirty_files_at_start": None,
}


def _canned_report(
    *,
    three_way: bool,
    hardware: dict[str, object] | None = None,
    run_start: dict[str, object] | None = None,
) -> dict[str, object]:
    conditions = ["vanilla", "wrapper_nogate", "wrapper"] if three_way else ["vanilla", "wrapper"]
    elapsed = {"vanilla": 100.0, "wrapper_nogate": 80.0, "wrapper": 50.0}
    peak = {"vanilla": 20.0, "wrapper_nogate": 20.5, "wrapper": 21.0}
    all_results = {
        c: [_schema4_chunk(c, r, elapsed[c] + r, peak[c] + r / 10, 0.5 + r) for r in range(2)]
        for c in conditions
    }
    return bs._build_report(
        all_results=all_results,
        variant="flux1-dev",
        num_inference_steps=25,
        guidance=3.5,
        reps=2,
        three_way=three_way,
        bench_dir=bs.REPO_ROOT / "tests" / "_artifacts" / "bench_images" / "flux1-dev",
        hardware=hardware if hardware is not None else {"chip": "test"},
        run_start=run_start if run_start is not None else _NO_START_READING,
    )


def test_report_schema_version_is_4() -> None:
    # bug caught: the report drops the load and loop peaks but still says schema 3,
    # so a reader of schema 3 looks for fields that are no longer there
    report = _canned_report(three_way=False)
    assert report["schema_version"] == 4
    assert {k for k in report if "memory_gb" in k} == {
        "vanilla_peak_memory_gb",
        "wrapper_peak_memory_gb",
        "vanilla_cache_memory_gb",
        "wrapper_cache_memory_gb",
    }
    assert report["vanilla_peak_memory_gb"] == [20.0, 20.1]
    assert report["wrapper_cache_memory_gb"] == [0.5, 1.5]
    assert report["vanilla_median"] == 100.5
    assert report["speedup_median"] == 100.5 / 50.5


def test_three_way_report_schema_4_memory_keys() -> None:
    # bug caught: the no-gate condition keeps its load and loop peak arrays
    report = _canned_report(three_way=True)
    assert {k for k in report if "memory_gb" in k} == {
        "vanilla_peak_memory_gb",
        "wrapper_peak_memory_gb",
        "nogate_peak_memory_gb",
        "vanilla_cache_memory_gb",
        "wrapper_cache_memory_gb",
        "nogate_cache_memory_gb",
    }
    assert report["nogate_peak_memory_gb"] == [20.5, 20.6]
    assert report["nogate_median"] == 80.5


# --- the checkpoint each condition loaded ------------------------------------

_QWEN_ORIGINAL = {
    "model_name": "Qwen/Qwen-Image",
    "base_model": "Qwen/Qwen-Image-2512",
    "model_revision": None,
}


def _report_with_checkpoints(checkpoints: dict[tuple[str, int], dict[str, object]]) -> dict[str, object]:
    """A three-way, two-rep report whose chunk (condition, rep) carries checkpoints[(condition, rep)]."""
    all_results = {
        c: [{**_schema4_chunk(c, r, 10.0 + r, 20.0, 0.5), **checkpoints[(c, r)]} for r in range(2)]
        for c in _CONDITIONS
    }
    return bs._build_report(
        all_results=all_results,
        variant="qwen",
        num_inference_steps=25,
        guidance=3.5,
        reps=2,
        three_way=True,
        bench_dir=bs.REPO_ROOT / "tests" / "_artifacts" / "bench_images" / "qwen",
        hardware={"chip": "test"},
        run_start=_NO_START_READING,
    )


def test_report_names_the_checkpoint_every_chunk_loaded() -> None:
    # bug caught: the report drops the checkpoint, so a Qwen-Image record cannot say
    # whether the original or the 2512 weights were measured
    pinned = {
        "model_name": "black-forest-labs/FLUX.1-dev",
        "base_model": None,
        "model_revision": "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
    }
    for checkpoint in (_QWEN_ORIGINAL, pinned):
        report = _report_with_checkpoints({(c, r): checkpoint for c in _CONDITIONS for r in range(2)})
        assert report["model_name"] == checkpoint["model_name"]
        assert report["base_model"] == checkpoint["base_model"]
        assert report["model_revision"] == checkpoint["model_revision"]


@pytest.mark.parametrize(
    "field, other",
    [
        ("model_name", "Qwen/Qwen-Image-2512"),
        ("base_model", "Qwen/Qwen-Image"),
        ("model_revision", "25468b98e3276ca6700de15c6628e51b7de54a26"),
    ],
)
def test_report_refuses_chunks_that_loaded_different_checkpoints(field: str, other: str) -> None:
    # bug caught: one report silently mixes timings from two checkpoints (a resumed run
    # whose later chunks loaded other weights), or compares only some of the fields
    checkpoints: dict[tuple[str, int], dict[str, object]] = {
        (c, r): _QWEN_ORIGINAL for c in _CONDITIONS for r in range(2)
    }
    checkpoints[("wrapper", 1)] = {**_QWEN_ORIGINAL, field: other}
    with pytest.raises(SystemExit, match="checkpoint") as excinfo:
        _report_with_checkpoints(checkpoints)
    assert other in str(excinfo.value)
    assert "wrapper rep 1" in str(excinfo.value)


@pytest.mark.mflux
def test_checkpoint_fields_come_from_the_loaded_model_config() -> None:
    # bug caught: the revision is looked up for the base model instead of the repo that was
    # loaded (Qwen/Qwen-Image is unpinned on mflux 0.24; its base, Qwen-Image-2512, is pinned),
    # or the name is taken from somewhere other than the loaded model's config
    from types import SimpleNamespace

    qwen = SimpleNamespace(
        model_config=SimpleNamespace(model_name="Qwen/Qwen-Image", base_model="Qwen/Qwen-Image-2512")
    )
    assert bs._checkpoint_fields(qwen) == _QWEN_ORIGINAL
    dev = SimpleNamespace(
        model_config=SimpleNamespace(model_name="black-forest-labs/FLUX.1-dev", base_model=None)
    )
    assert bs._checkpoint_fields(dev) == {
        "model_name": "black-forest-labs/FLUX.1-dev",
        "base_model": None,
        "model_revision": "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
    }


def test_klein_report_schema_version_is_2() -> None:
    # bug caught: Klein chunks drop the load and loop peaks but the report still says schema 1
    import bench_klein_base_vs_distilled as bk

    chunk = {
        "condition": "base",
        "rep": 0,
        "elapsed_s": 500.0,
        "num_inference_steps": 50,
        "peak_memory_gb": 9.0,
        "cache_memory_gb": 0.5,
        "stats_summary": {},
    }
    report = bk._build_report(
        size="4b",
        quantize=4,
        reps=1,
        conditions=["base"],
        loaded={"base": [chunk]},
        ssim={},
        images_dir=bk.REPO_ROOT / "tests" / "_artifacts" / "klein_images" / "4b",
        hardware={},
        height=512,
        width=512,
    )
    assert report["schema_version"] == 2
    assert report["conditions"]["base"]["peak_memory_gb"] == 9.0


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


_REPO_LOCATION_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY")


def _isolated_git_env() -> dict[str, str]:
    """The process environment without the variables that point git at another repository."""
    return {k: v for k, v in os.environ.items() if k not in _REPO_LOCATION_VARS}


def test_isolated_git_env_drops_the_four_repo_location_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    # bug caught: a name missing from the removal list lets an exported work tree, index or
    # object store redirect the temp repo's git calls
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"):
        monkeypatch.setenv(name, "/decoy")
    env = _isolated_git_env()
    assert {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"} & env.keys() == set()
    assert env["PATH"] == os.environ["PATH"]


def _init_repo_with_commit(path: Path) -> None:
    env = _isolated_git_env()
    subprocess.run(["git", "init", "-q", str(path)], env=env, check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
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
        env=env,
        check=True,
    )


def _head_and_count(repo: Path) -> tuple[str, str]:
    env = _isolated_git_env()
    git = ["git", "-C", str(repo)]
    head = subprocess.run([*git, "rev-parse", "HEAD"], env=env, capture_output=True, text=True, check=True)
    count = subprocess.run(
        [*git, "rev-list", "--count", "HEAD"], env=env, capture_output=True, text=True, check=True
    )
    return head.stdout.strip(), count.stdout.strip()


def test_repo_setup_ignores_an_inherited_git_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # bug caught: the setup's git calls inherit GIT_DIR (exported by `git rebase -x`) and commit
    # into the repository being rebased instead of the temporary one
    decoy = tmp_path / "decoy"
    env = _isolated_git_env()
    subprocess.run(["git", "init", "-q", str(decoy)], env=env, check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(decoy),
            "-c",
            "user.email=d@d",
            "-c",
            "user.name=d",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "d",
        ],
        env=env,
        check=True,
    )
    head_before, count_before = _head_and_count(decoy)
    assert count_before == "1"

    monkeypatch.setenv("GIT_DIR", str(decoy / ".git"))
    _init_repo_with_commit(tmp_path / "repo")

    assert _head_and_count(decoy) == (head_before, "1")


def _drop_inherited_repo_location(monkeypatch: pytest.MonkeyPatch) -> None:
    # `_git_revision` runs git with the process env; under `git rebase -x` it would read the rebased repo
    for name in _REPO_LOCATION_VARS:
        monkeypatch.delenv(name, raising=False)


def test_git_revision_reports_commit_and_dirty_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _drop_inherited_repo_location(monkeypatch)
    _init_repo_with_commit(tmp_path)
    rev = bs._git_revision(tmp_path)
    assert isinstance(rev["git_commit"], str) and len(rev["git_commit"]) == 40
    # bug caught: the reported sha comes from an inherited GIT_DIR's repository, not the temp repo
    assert rev["git_commit"] == _head_and_count(tmp_path)[0]
    assert rev["git_dirty"] is False
    (tmp_path / "f").write_text("x")
    assert bs._git_revision(tmp_path)["git_dirty"] is True


def test_git_revision_outside_a_repo_is_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _drop_inherited_repo_location(monkeypatch)
    assert bs._git_revision(tmp_path) == {"git_commit": None, "git_dirty": None, "git_dirty_files": None}


def test_memory_arrays_report_an_unknown_cache_pool_as_none() -> None:
    # bug caught: reporting a pre-v0.10.1 chunk's process peak as its cache-pool size
    old = {"peak_memory_gb": 26.0}
    new = {"peak_memory_gb": 26.2, "cache_memory_gb": 0.9}
    assert bs._memory_arrays([old, new], "cache_memory_gb") == [None, 0.9]


def test_parse_worker_line_prefers_an_abort_payload_over_an_earlier_result() -> None:
    # bug caught: taking the first sentinel line, so an abort that fires after the
    # result was printed (during image.save) reads as "worker failed: exit 3"
    ok = f"{bs.WORKER_RESULT_SENTINEL}{json.dumps({'elapsed_s': 1.0})}"
    ab = f"{bs.WORKER_RESULT_SENTINEL}{json.dumps({'aborted': 'active-memory watchdog'})}"
    assert bs._parse_worker_line(f"{ok}\n{ab}\n") == {"aborted": "active-memory watchdog"}


def test_krea_dev_slug_maps_to_the_variant_and_its_model_card_recipe() -> None:
    # bug caught: benching krea at flux1-dev's 25 / 3.5 instead of its own 28 / 4.5
    assert bs._VARIANT_SLUG_TO_ID["krea-dev"] == "flux1-krea-dev"
    assert bs._VARIANT_RECIPE["krea-dev"] == {"num_inference_steps": 28, "guidance": 4.5}
    assert bs._VARIANT_QUANTIZE["krea-dev"] == 4


def test_memory_saver_is_enabled_only_for_the_variant_that_needs_it() -> None:
    # bug caught: freeing text encoders on every variant (changes nothing for FLUX.1-sized runs
    # but makes their reports incomparable with the committed ones)
    assert bs._memory_saver_for("qwen") is True
    assert bs._memory_saver_for("flux1-dev") is False
    assert bs._memory_saver_for("krea-dev") is False


def test_memory_saver_is_built_with_the_load_bearing_kwargs() -> None:
    """bug caught: dropping cache_limit_bytes=None (MemorySaver's default 1 GB branch
    overrides our cache cap, resets the peak counter before the load peak is read and
    switches the VAE to tiled decode, changing output pixels) or keep_transformer=True."""
    seen: dict = dict()

    class _FakeSaver:
        def __init__(self, **kw):
            seen.update(kw)

    flux = object()
    bs._make_memory_saver(flux, _FakeSaver)
    assert seen == dict(model=flux, keep_transformer=True, cache_limit_bytes=None, num_seeds=1)


# --- complete bench records: dirty-file list, report date, commit at run start ---


_STATUS_ARGV = ["status", "--porcelain", "--untracked-files=normal"]


class _FakeGit:
    """Stands in for ``subprocess.run`` at the process boundary: answers ``git rev-parse HEAD``
    and ``git status --porcelain --untracked-files=normal`` from its fields, and fails every other
    command the way a missing binary does (so sysctl reads in ``_detect_hardware`` fall back to None).
    A status call without ``--untracked-files=normal`` is an error: its dirty flag would follow the
    user's ``status.showUntrackedFiles`` setting."""

    def __init__(self, *, head: str, porcelain: str) -> None:
        self.head = head
        self.porcelain = porcelain

    def __call__(self, cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "git" and cmd[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{self.head}\n", stderr="")
        if cmd[0] == "git" and "status" in cmd:
            assert cmd[-3:] == _STATUS_ARGV, f"git status must pin the untracked-files mode: {cmd}"
            return subprocess.CompletedProcess(cmd, 0, stdout=self.porcelain, stderr="")
        raise FileNotFoundError(cmd[0])


_HEAD_A = "a" * 40
_HEAD_B = "b" * 40


def test_git_revision_lists_every_dirty_path_from_porcelain(monkeypatch: pytest.MonkeyPatch) -> None:
    # bug caught: the porcelain output is thrown away after the bool (only `git_dirty` is kept),
    # or the whole output is stripped first, so the leading space of " M path" is lost and the
    # first path loses its first letter
    porcelain = " M scripts/bench_speedup.py\n?? notes.txt\nR  old.py -> new.py\n"
    monkeypatch.setattr(bs.subprocess, "run", _FakeGit(head=_HEAD_A, porcelain=porcelain))
    assert bs._git_revision(bs.REPO_ROOT) == {
        "git_commit": _HEAD_A,
        "git_dirty": True,
        "git_dirty_files": ["scripts/bench_speedup.py", "notes.txt", "old.py -> new.py"],
    }


def test_git_revision_dirty_files_are_an_empty_list_on_a_clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    # bug caught: a clean tree reports [""] (splitting the empty output on "\n") or None, which a
    # reader cannot tell apart from "not a checkout"
    monkeypatch.setattr(bs.subprocess, "run", _FakeGit(head=_HEAD_A, porcelain=""))
    assert bs._git_revision(bs.REPO_ROOT) == {
        "git_commit": _HEAD_A,
        "git_dirty": False,
        "git_dirty_files": [],
    }


def test_hardware_block_of_the_report_carries_the_dirty_files(monkeypatch: pytest.MonkeyPatch) -> None:
    # bug caught: `_detect_hardware` builds its git fields by hand and leaves the new list out,
    # so the written report still has no dirty-file list
    monkeypatch.setattr(bs.subprocess, "run", _FakeGit(head=_HEAD_A, porcelain="?? notes.txt\n"))
    monkeypatch.setattr(bs, "_mlx_teacache_version", lambda: "x")
    report = _canned_report(three_way=False, hardware=bs._detect_hardware(quantize=4))
    hardware = report["hardware"]
    assert isinstance(hardware, dict)
    assert hardware["git_dirty_files"] == ["notes.txt"]
    assert hardware["git_dirty"] is True


def test_report_has_a_utc_generated_at_from_when_it_was_built() -> None:
    # bug caught: no report-level date, a naive local time, or a date copied from a chunk
    # (the canned chunks started on 2026-11-10) instead of the moment the report is written
    before = datetime.now(UTC)
    report = _canned_report(three_way=False)
    after = datetime.now(UTC)
    generated = datetime.fromisoformat(str(report["generated_at"]))
    assert generated.utcoffset() == timedelta(0)
    assert before <= generated <= after


def test_start_reading_survives_a_resume_after_head_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # bug caught: the start reading is taken again on every invocation (a resume overwrites it
    # with the later HEAD), or the report writes the report-time reading as the start one
    git = _FakeGit(head=_HEAD_A, porcelain=" M src/mlx_teacache/x.py\n")
    monkeypatch.setattr(bs.subprocess, "run", git)
    first = bs._run_start_reading(tmp_path, resumed=False)
    assert first == {
        "git_commit_at_start": _HEAD_A,
        "git_dirty_at_start": True,
        "git_dirty_files_at_start": ["src/mlx_teacache/x.py"],
    }

    git.head, git.porcelain = _HEAD_B, ""  # a commit landed between two invocations
    resumed = bs._run_start_reading(tmp_path, resumed=True)
    assert resumed == first

    report = _canned_report(three_way=False, hardware=bs._git_revision(bs.REPO_ROOT), run_start=resumed)
    hardware = report["hardware"]
    assert isinstance(hardware, dict)
    assert (hardware["git_commit_at_start"], hardware["git_commit"]) == (_HEAD_A, _HEAD_B)
    assert (hardware["git_dirty_at_start"], hardware["git_dirty"]) == (True, False)
    assert (hardware["git_dirty_files_at_start"], hardware["git_dirty_files"]) == (
        ["src/mlx_teacache/x.py"],
        [],
    )


def test_a_fresh_run_replaces_a_start_reading_left_by_a_run_with_no_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # bug caught: the stored reading is reused whenever the file exists, so a run whose first
    # worker aborted (no chunk kept) stamps every later chunk with that older HEAD
    git = _FakeGit(head=_HEAD_A, porcelain="")
    monkeypatch.setattr(bs.subprocess, "run", git)
    bs._run_start_reading(tmp_path, resumed=False)
    git.head = _HEAD_B
    assert bs._run_start_reading(tmp_path, resumed=False)["git_commit_at_start"] == _HEAD_B


def test_a_resume_with_no_stored_start_reading_says_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # bug caught: chunks persisted before the start reading existed get the current HEAD written
    # as their start commit, which is a claim the record cannot back
    monkeypatch.setattr(bs.subprocess, "run", _FakeGit(head=_HEAD_B, porcelain=""))
    assert bs._run_start_reading(tmp_path, resumed=True) == _NO_START_READING


def test_a_resumed_start_reading_cannot_override_measured_hardware_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # bug caught: the resume returns run_start.json verbatim, so a stale or edited file that carries
    # report-time keys (git_commit, mlx_version) replaces the measured values in the hardware block
    monkeypatch.setattr(bs.subprocess, "run", _FakeGit(head=_HEAD_B, porcelain=""))
    stored = {
        "git_commit_at_start": _HEAD_A,
        "git_dirty_at_start": False,
        "git_dirty_files_at_start": [],
        "git_commit": _HEAD_A,
        "mlx_version": "0.0.1",
    }
    (tmp_path / "run_start.json").write_text(json.dumps(stored))
    resumed = bs._run_start_reading(tmp_path, resumed=True)
    assert resumed == {
        "git_commit_at_start": _HEAD_A,
        "git_dirty_at_start": False,
        "git_dirty_files_at_start": [],
    }
    measured = {"chip": "test", "mlx_version": "0.32.2", "git_commit": _HEAD_B}
    hardware = _canned_report(three_way=False, hardware=measured, run_start=resumed)["hardware"]
    assert isinstance(hardware, dict)
    assert (hardware["git_commit"], hardware["mlx_version"]) == (_HEAD_B, "0.32.2")
    assert hardware["git_commit_at_start"] == _HEAD_A


def test_a_resumed_start_reading_missing_a_field_reports_it_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # bug caught: filtering to the stored keys drops a start field an older or edited file lacks,
    # so the report's hardware block loses the key instead of saying unknown (None)
    monkeypatch.setattr(bs.subprocess, "run", _FakeGit(head=_HEAD_B, porcelain=""))
    (tmp_path / "run_start.json").write_text(json.dumps({"git_commit_at_start": _HEAD_A}))
    assert bs._run_start_reading(tmp_path, resumed=True) == {
        "git_commit_at_start": _HEAD_A,
        "git_dirty_at_start": None,
        "git_dirty_files_at_start": None,
    }


class _HangingGit:
    """``subprocess.run`` where one git subcommand hangs: with a timeout it raises the
    ``TimeoutExpired`` the real call raises when the timeout runs out; without one it fails the test,
    since the real call would block the bench forever. Other git calls answer as ``_FakeGit``."""

    def __init__(self, hangs: str) -> None:
        self.hangs = hangs
        self.answer = _FakeGit(head=_HEAD_A, porcelain="")

    def __call__(self, cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "git" and self.hangs in cmd:
            assert kwargs.get("timeout") == 5, f"git {self.hangs} has no 5 s timeout: {kwargs}"
            raise subprocess.TimeoutExpired(cmd, 5)
        return self.answer(cmd, **kwargs)


@pytest.mark.parametrize("hangs", ["rev-parse", "status"])
def test_git_revision_gives_up_on_a_hung_git_call(hangs: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # bug caught: a git call without a timeout blocks the bench forever, or the TimeoutExpired is not
    # caught (only FileNotFoundError / CalledProcessError are) and aborts the report
    monkeypatch.setattr(bs.subprocess, "run", _HangingGit(hangs))
    assert bs._git_revision(bs.REPO_ROOT) == {"git_commit": None, "git_dirty": None, "git_dirty_files": None}
