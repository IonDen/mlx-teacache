"""Pure-helper unit tests for ``scripts/bench_comparison.py``.

Covers two harness defects:

  - ``_condition_metrics`` — cold = rep 1, warm = median of reps 2+, with the
    one-rep images-only preview (``--reps 1``) returning ``warm=None`` instead of
    crashing on ``statistics.median([])``.
  - ``_speedup`` — vanilla/wrapper ratio with None/zero-denominator guards.
  - ``_merge_variant_into_report`` — a ``--only <slug>`` resume refreshes the
    report's top-level provenance to the current run and stamps per-variant
    provenance, so a resumed report can no longer keep a prior run's stale
    ``generated_at`` / version.

``bench_comparison`` imports mflux / PIL / mlx only lazily (inside the worker
functions), so this module imports cleanly in the pure-core lane.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_comparison as bc  # noqa: E402

# --- Finding 2: cold/warm split, one-rep preview is crash-free ---------------


def test_condition_metrics_three_reps_cold_first_warm_median_of_rest() -> None:
    assert bc._condition_metrics([12.0, 9.0, 7.0]) == {"cold": 12.0, "warm": 8.0}


def test_condition_metrics_one_rep_has_no_warm() -> None:
    # --reps 1 (images-only preview): no warm measurement, must NOT raise on median([]).
    assert bc._condition_metrics([5.0]) == {"cold": 5.0, "warm": None}


def test_condition_metrics_two_reps_warm_is_single_rest() -> None:
    assert bc._condition_metrics([10.0, 8.0]) == {"cold": 10.0, "warm": 8.0}


def test_speedup_is_vanilla_over_wrapper() -> None:
    assert bc._speedup(10.0, 8.0) == 1.25


def test_speedup_is_none_when_either_side_missing() -> None:
    assert bc._speedup(10.0, None) is None
    assert bc._speedup(None, 8.0) is None


def test_speedup_is_none_on_zero_denominator() -> None:
    assert bc._speedup(10.0, 0.0) is None


# --- Finding 3: resume refreshes top-level provenance, stamps per-variant ----


def _prov() -> dict[str, str]:
    return {
        "generated_at": "2026-06-18T08:09Z",
        "mlx_teacache_version": "0.9.0",
        "mflux_version": "0.17.5",
    }


def test_merge_stamps_per_variant_provenance() -> None:
    out = bc._merge_variant_into_report(
        {"schema_version": 1, "variants": {}}, "qwen-image", {"speedup_warm": 1.74}, _prov()
    )
    assert out["variants"]["qwen-image"]["provenance"] == _prov()
    assert out["variants"]["qwen-image"]["speedup_warm"] == 1.74


def test_merge_refreshes_stale_top_level_on_resume() -> None:
    # The resume scenario: a --only run reloads a report whose top-level was last
    # written by an earlier run (here 2026-05-18 / 0.1.0).
    stale = {
        "schema_version": 1,
        "generated_at": "2026-05-18T07:02Z",
        "hardware": {
            "chip": "Apple M1 Max",
            "ram_gb": 32,
            "mlx_teacache_version": "0.1.0",
            "mflux_version": "0.17.5",
        },
        "prompt": "portrait",
        "variants": {"flux1-dev": {"speedup_warm": 1.18}},
    }
    out = bc._merge_variant_into_report(stale, "qwen-image", {"speedup_warm": 1.74}, _prov())

    assert out["generated_at"] == "2026-06-18T08:09Z"
    assert out["hardware"]["mlx_teacache_version"] == "0.9.0"
    assert out["hardware"]["mflux_version"] == "0.17.5"
    # stable machine fields, unrelated keys, and pre-existing rows are preserved;
    # the older row gets NO fabricated provenance (its true version is unknown).
    assert out["hardware"]["chip"] == "Apple M1 Max"
    assert out["hardware"]["ram_gb"] == 32
    assert out["prompt"] == "portrait"
    assert out["variants"]["flux1-dev"] == {"speedup_warm": 1.18}
    assert out["variants"]["qwen-image"]["provenance"] == _prov()


def test_merge_is_pure_and_does_not_mutate_input() -> None:
    report = {"variants": {}, "generated_at": "old", "hardware": {"mlx_teacache_version": "0.1.0"}}
    bc._merge_variant_into_report(report, "x", {"a": 1}, _prov())
    assert report["generated_at"] == "old"
    assert report["variants"] == {}
    assert report["hardware"]["mlx_teacache_version"] == "0.1.0"


# --- skip-streak telemetry (shared with bench_speedup via scripts/_bench_telemetry) ---


def test_streak_telemetry_reads_the_last_committed_generation() -> None:
    from mlx_teacache._kernel.stats import StepDecision, TeaCacheStats

    stats = TeaCacheStats()
    kinds = ["forced", "skipped", "skipped", "computed", "skipped", "skipped", "skipped", "computed"]
    for i, kind in enumerate(kinds):
        stats.record(
            StepDecision(
                step_idx=i, timestep=1.0 - i / 10, rel_l1=0.1, accumulated_distance=0.2, decision=kind
            )  # type: ignore[arg-type]
        )
    stats.finalize_last_generation(num_inference_steps=len(kinds), cfg_was_active=False)
    assert bc._streak_telemetry(stats) == {"skip_pattern": "CSSCSSSC", "max_consecutive_skips": 3}
