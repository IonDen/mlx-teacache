"""Doc-to-artifact consistency for the committed FLUX.1-dev benchmark.

CLAUDE.md requires every user-facing speedup claim to be backed by a committed
benchmark. This module pins the README's FLUX.1-dev headline row (speedup, skip
count, per-condition seconds) to the committed ``bench_speedup.py`` report so the
number can never drift from the artifact that produced it.

Guards the v0.6.x design-review H1 finding: the FLUX.1-dev headline was repeated
in several places and called "the reproducible ``bench_speedup.py`` number" while
no committed JSON actually produced it.

The numeric/rounding rules live in the pure ``bench_headline`` helper so they are
unit-testable without touching the filesystem; the consistency tests read the
real committed files.
"""

import json
import statistics
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FLUX1_DEV_BENCH = _REPO_ROOT / "_artifacts" / "v0.10.0_bench_flux1_dev.json"
_README = _REPO_ROOT / "README.md"

_RUN_HINT = (
    "uv run python scripts/bench_speedup.py --variant flux1-dev "
    "--three-way --reps 3 --report _artifacts/v0.10.0_bench_flux1_dev.json"
)


# --- pure helper -------------------------------------------------------------


def bench_headline(report: dict) -> dict:
    """Canonical headline figures derived from a ``bench_speedup.py`` report.

    Pure so the rounding/median rules are testable without the filesystem.
    Returns the exact values the README prints:
      ``speedup_x``  median wall-clock ratio, 2 dp (the ``1.46`` in ``1.46×``)
      ``skipped``    median per-rep skipped-step count (the ``6`` in ``6 / 25``)
      ``steps``      ``num_inference_steps`` (the ``25`` denominator)
      ``vanilla_s``  median vanilla seconds, 1 dp (the ``103.8`` in ``103.8s``)
      ``wrapper_s``  median wrapper seconds, 1 dp (the ``71.0`` in ``71.0s``)
    """
    return {
        "speedup_x": round(float(report["speedup_median"]), 2),
        "skipped": int(statistics.median(report["skipped_counts"])),
        "steps": int(report["num_inference_steps"]),
        "vanilla_s": round(float(report["vanilla_median"]), 1),
        "wrapper_s": round(float(report["wrapper_median"]), 1),
    }


# --- file readers ------------------------------------------------------------


def _load_bench() -> dict:
    assert _FLUX1_DEV_BENCH.exists(), (
        f"Committed FLUX.1-dev bench missing: {_FLUX1_DEV_BENCH}\n  Run: {_RUN_HINT}"
    )
    return json.loads(_FLUX1_DEV_BENCH.read_text())


def _flux1_dev_benchmark_row() -> list[str]:
    """Cells of the README Benchmarks-table row for ``flux1-dev``.

    Disambiguated from the Supported-models table row (same leading cell) by
    requiring the 7-column shape with a numeric Steps cell.
    """
    for line in _README.read_text().splitlines():
        s = line.strip()
        if s.startswith("| `flux1-dev`"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if len(cells) >= 7 and cells[1].isdigit():
                return cells
    raise AssertionError("No `flux1-dev` row found in the README Benchmarks table")


# --- pure-helper unit tests (RED first: bench_headline is undefined) ---------


def test_bench_headline_rounds_speedup_to_two_dp():
    report = {
        "speedup_median": 1.4438,
        "skipped_counts": [6, 6, 6],
        "num_inference_steps": 25,
        "vanilla_median": 103.74,
        "wrapper_median": 71.81,
    }
    assert bench_headline(report) == {
        "speedup_x": 1.44,
        "skipped": 6,
        "steps": 25,
        "vanilla_s": 103.7,
        "wrapper_s": 71.8,
    }


def test_bench_headline_uses_median_skip_not_max():
    # [5, 6, 9]: median 6, max 9 — a buggy max() would return 9, not 6.
    report = {
        "speedup_median": 1.40,
        "skipped_counts": [5, 6, 9],
        "num_inference_steps": 25,
        "vanilla_median": 100.0,
        "wrapper_median": 71.4,
    }
    assert bench_headline(report)["skipped"] == 6


def test_bench_headline_median_skip_picks_middle_value():
    report = {
        "speedup_median": 1.10,
        "skipped_counts": [4, 5, 9],
        "num_inference_steps": 25,
        "vanilla_median": 100.0,
        "wrapper_median": 90.9,
    }
    assert bench_headline(report)["skipped"] == 5


# --- consistency tests (RED until the artifact exists + README reconciled) ---


def test_flux1_dev_bench_artifact_is_committed_and_valid():
    report = _load_bench()
    assert report["schema_version"] == 2
    assert report["variant"] == "flux1-dev"
    assert report["num_inference_steps"] == 25
    reps = report["reps"]
    assert reps >= 3, f"need >=3 reps for a credible median, got {reps}"
    assert len(report["vanilla_seconds"]) == reps
    assert len(report["wrapper_seconds"]) == reps
    assert len(report["skipped_counts"]) == reps, "per-rep skip telemetry required"


def test_readme_benchmark_row_matches_committed_artifact():
    h = bench_headline(_load_bench())
    cells = _flux1_dev_benchmark_row()
    # cells: [variant, steps, vanilla, wrapper, speedup, skipped, mechanism]
    assert int(cells[1]) == h["steps"]
    assert cells[2].endswith("s") and cells[2][:-1] == f"{h['vanilla_s']:.1f}"
    assert cells[3].endswith("s") and cells[3][:-1] == f"{h['wrapper_s']:.1f}"
    assert cells[4].replace("*", "").replace("×", "") == f"{h['speedup_x']:.2f}"
    assert cells[5].replace("*", "") == f"{h['skipped']} / {h['steps']}"


def test_flux1_dev_bench_artifact_is_meaningful():
    """Schema validity (the sibling test) is not enough — a corrupt artifact with
    a sub-1 speedup or all-zero skips would pass that. Pin the *meaning*: the
    headline speedup is real and the cache actually engages.

    Note: ``skipped + computed`` is 23, not 25 — the forced first/last-window
    steps are in neither count and there is no ``forced_counts`` field, so we
    assert the sound bound (sum cannot exceed total steps) rather than an
    identity we cannot reconstruct from this artifact.
    """
    report = _load_bench()
    steps = int(report["num_inference_steps"])
    skipped = report["skipped_counts"]
    computed = report["computed_counts"]

    assert float(report["speedup_median"]) > 1.0, "headline speedup must be > 1x to be a speedup"
    assert 0 < statistics.median(skipped) < steps, (
        "cache must engage (median skips > 0) without skipping every step"
    )
    assert len(computed) == len(skipped), "per-rep computed/skipped telemetry must align"
    for i, (sk, cp) in enumerate(zip(skipped, computed, strict=True)):
        assert sk >= 0 and cp > 0, f"rep {i}: counts must be non-negative / computed>0 ({sk}, {cp})"
        assert sk + cp <= steps, f"rep {i}: skipped+computed {sk + cp} exceeds {steps} steps"


# --- qwen-image ---------------------------------------------------------------
# Qwen joined the bench harness in v0.10.0 and was re-benched in v0.11.0 on the
# guarded harness with the text encoders freed. Its README row carries the largest
# speedup in the table, so it gets the same artifact pin as the flux1-dev row.

_QWEN_BENCH = _REPO_ROOT / "_artifacts" / "v0.11.0_bench_qwen_image.json"


def _load_qwen_bench() -> dict:
    return json.loads(_QWEN_BENCH.read_text())


def _qwen_benchmark_row() -> list[str]:
    """Cells of the README Benchmarks-table row for ``qwen-image``.

    Disambiguated from the Supported-models row (same leading cell) by the
    7-column shape with a numeric Steps cell.
    """
    for line in _README.read_text().splitlines():
        s = line.strip()
        if s.startswith("| `qwen-image`"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if len(cells) >= 7 and cells[1].isdigit():
                return cells
    raise AssertionError("No `qwen-image` row found in the README Benchmarks table")


def test_qwen_bench_artifact_is_committed_and_valid():
    """RED if the qwen report is missing or was written with fewer than three
    reps — the headline would then be a single-run number."""
    report = _load_qwen_bench()
    assert report["schema_version"] == 3
    assert report["variant"] == "qwen"
    assert report["chunk_order"] == "rep-outer"
    assert report["memory_saver"] is True, "the README's 24.1 GB peak is the MemorySaver figure"
    assert report["num_inference_steps"] == 50
    assert report["height"] == 768 and report["width"] == 768, (
        "qwen benches at its pinned 768x768, not the shared 512x512 recipe"
    )
    reps = report["reps"]
    assert reps >= 3, f"need >=3 reps for a credible median, got {reps}"
    assert len(report["vanilla_seconds"]) == reps
    assert len(report["wrapper_seconds"]) == reps
    assert len(report["skipped_counts"]) == reps


def test_readme_qwen_row_matches_committed_artifact():
    """RED if the README's qwen numbers drift from the committed report."""
    h = bench_headline(_load_qwen_bench())
    cells = _qwen_benchmark_row()
    assert int(cells[1]) == h["steps"]
    assert cells[2].endswith("s") and cells[2][:-1] == f"{h['vanilla_s']:.1f}"
    assert cells[3].endswith("s") and cells[3][:-1] == f"{h['wrapper_s']:.1f}"
    assert cells[4].replace("*", "").replace("×", "") == f"{h['speedup_x']:.2f}"
    assert cells[5].replace("*", "") == f"{h['skipped']} / {h['steps']}"


def test_qwen_streak_stays_under_the_runaway_cap():
    """The release documents the cap as never engaging at a shipped default.
    Qwen has the longest streak of any variant, so it is the row that would
    falsify that claim first. RED if a future re-measure reaches the cap."""
    from mlx_teacache._kernel.gate import MAX_CONSECUTIVE_SKIPS

    streaks = _load_qwen_bench()["max_consecutive_skips"]
    assert max(streaks) < MAX_CONSECUTIVE_SKIPS, (
        f"qwen streak {max(streaks)} reached the cap {MAX_CONSECUTIVE_SKIPS}; "
        "the documented 'cap never engages at a shipped default' claim is stale"
    )


def test_qwen_bench_artifact_is_meaningful():
    """Schema validity is not enough: a corrupt artifact with all-zero skips and
    a sub-1 speedup, with the README synced to match, passes every other qwen
    test. Pin the *meaning* — the headline is a real speedup and the cache
    actually engages. RED on a dormant-cache or slower-than-vanilla artifact."""
    report = _load_qwen_bench()
    steps = int(report["num_inference_steps"])
    skipped = report["skipped_counts"]
    computed = report["computed_counts"]
    assert report["speedup_median"] > 1.0, "qwen headline must be a real speedup"
    med = statistics.median(skipped)
    assert 0 < med < steps, f"median skips {med} must be strictly inside (0, {steps})"
    for s_, c_ in zip(skipped, computed, strict=True):
        assert s_ + c_ <= steps, f"skipped {s_} + computed {c_} exceeds {steps} steps"


# ---------------------------------------------------------------------------
# flux1-krea-dev — v0.11.0 three-way bench (schema 3: load/loop/cache peaks, rep-outer order)
# ---------------------------------------------------------------------------

_KREA_BENCH = _REPO_ROOT / "_artifacts" / "v0.11.0_bench_krea_dev.json"


def _load_krea_bench() -> dict:
    return json.loads(_KREA_BENCH.read_text())


def _krea_benchmark_row() -> list[str]:
    """Cells of the README Benchmarks-table row for ``flux1-krea-dev`` (7 columns,
    numeric Steps cell; the Supported-models row shares the leading cell)."""
    for line in _README.read_text().splitlines():
        s = line.strip()
        if s.startswith("| `flux1-krea-dev`"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if len(cells) >= 7 and cells[1].isdigit():
                return cells
    raise AssertionError("No `flux1-krea-dev` row found in the README Benchmarks table")


def test_krea_bench_artifact_is_committed_and_valid():
    report = _load_krea_bench()
    assert report["schema_version"] == 3
    assert report["variant"] == "krea-dev"
    assert report["num_inference_steps"] == 28 and report["guidance"] == 4.5
    assert report["chunk_order"] == "rep-outer"
    reps = report["reps"]
    assert reps >= 3
    assert (
        len(report["vanilla_seconds"])
        == len(report["wrapper_seconds"])
        == len(report["nogate_seconds"])
        == reps
    )
    assert len(report["wrapper_loop_peak_memory_gb"]) == reps


def test_readme_krea_row_matches_committed_artifact():
    h = bench_headline(_load_krea_bench())
    cells = _krea_benchmark_row()
    assert int(cells[1]) == h["steps"]
    assert cells[2].endswith("s") and cells[2][:-1] == f"{h['vanilla_s']:.1f}"
    assert cells[3].endswith("s") and cells[3][:-1] == f"{h['wrapper_s']:.1f}"
    assert cells[4].replace("*", "").replace("×", "") == f"{h['speedup_x']:.2f}"
    assert cells[5].replace("*", "") == f"{h['skipped']} / {h['steps']}"


def test_krea_bench_artifact_is_meaningful():
    """Real speedup, cache engaged, every rep skips the same count with no streak
    above 1 (the sweep knee's operating point), and the no-gate condition skips nothing."""
    report = _load_krea_bench()
    assert report["speedup_median"] > 1.0
    assert report["skipped_counts"] == [10, 10, 10]
    assert max(report["max_consecutive_skips"]) == 1
    assert report["gating_ratio"] > report["compile_avoidance_ratio"], (
        "the win must come from gating on FLUX.1"
    )
