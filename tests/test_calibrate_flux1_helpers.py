"""Pure-helper tests for scripts/calibrate_flux1.py: pair extraction, R² scoring
of a given tuple, chunk bookkeeping, the sweep summary, and the per-model
recipes. No weights, no mflux import at module top."""

import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_flux1 as cf  # noqa: E402


def test_pairs_from_capture_are_consecutive_rel_l1_of_signal_and_body() -> None:
    # bug caught: pairing step t with step t-2, or swapping x and y
    mod = [mx.ones((4,)), mx.ones((4,)) * 2.0, mx.ones((4,)) * 4.0]
    body = [mx.ones((4,)), mx.ones((4,)) * 1.5, mx.ones((4,)) * 3.0]
    pairs = cf._pairs_from_capture(mod, body)
    assert len(pairs) == 2
    assert pairs[0] == (1.0, 0.5)  # |2-1|/1 , |1.5-1|/1
    assert pairs[1] == (1.0, 1.0)  # |4-2|/2 , |3-1.5|/1.5


def test_r2_score_is_one_for_the_generating_polynomial_and_low_for_a_wrong_one() -> None:
    # bug caught: computing SS_res against the mean, or returning 1 - SS_tot/SS_res
    coeffs = (0.0, 0.0, 2.0, 1.0, 0.5)  # 2x^2 + x + 0.5
    xs = [0.1 * i for i in range(10)]
    ys = [2 * x * x + x + 0.5 for x in xs]
    assert abs(cf._r2_score(coeffs, xs, ys) - 1.0) < 1e-12
    assert cf._r2_score((0.0, 0.0, 0.0, 0.0, 0.5), xs, ys) < 0.5


def test_chunk_bookkeeping_is_per_model_and_resumes(tmp_path: Path) -> None:
    # bug caught: dev and krea-dev chunks sharing a filename
    assert cf._chunk_filename("krea-dev", 3) != cf._chunk_filename("dev", 3)
    (tmp_path / cf._chunk_filename("krea-dev", 0)).write_text("{}")
    assert cf._pending_prompt_indices(tmp_path, "krea-dev", 3) == [1, 2]
    assert cf._pending_prompt_indices(tmp_path, "dev", 3) == [0, 1, 2]


def test_aggregate_chunks_concatenates_pairs_in_prompt_order(tmp_path: Path) -> None:
    for idx, pairs in ((1, [[0.3, 0.4]]), (0, [[0.1, 0.2], [0.2, 0.3]])):
        (tmp_path / cf._chunk_filename("dev", idx)).write_text(json.dumps({"pairs": pairs}))
    xs, ys = cf._aggregate_pairs(tmp_path, "dev", 2)
    assert xs == [0.1, 0.2, 0.3]
    assert ys == [0.2, 0.3, 0.4]


def test_build_sweep_summary_sorts_and_carries_streak_telemetry() -> None:
    chunks = [
        {
            "threshold": 0.25,
            "wrapper_seconds": 5.0,
            "skipped": 6,
            "computed": 22,
            "ssim_vs_vanilla": 0.95,
            "skip_pattern": "CSC",
            "max_consecutive_skips": 1,
        },
        {"threshold": 0.15, "wrapper_seconds": 8.0, "skipped": 2, "computed": 26, "ssim_vs_vanilla": 0.99},
    ]
    summary = cf._build_sweep_summary(chunks, vanilla_seconds=10.0, model="krea-dev")
    rows = summary["thresholds"]
    assert [r["threshold"] for r in rows] == [0.15, 0.25]
    assert rows[0]["max_consecutive_skips"] == 0 and rows[1]["skip_pattern"] == "CSC"
    assert rows[1]["speedup_vs_vanilla_single_rep"] == 2.0
    assert summary["model"] == "krea-dev"


def test_recipes_follow_each_model_card() -> None:
    assert cf.RECIPES["dev"] == {"num_inference_steps": 25, "guidance": 3.5}
    assert cf.RECIPES["krea-dev"] == {"num_inference_steps": 28, "guidance": 4.5}


def test_dry_run_sweep_chunk_uses_the_models_step_count(tmp_path: Path) -> None:
    """bug caught: the dry-run sweep chunk hard-coding Krea's 28 steps for every
    model, so a `--model dev --dry-run --sweep` chunk (25-step recipe) carries a
    skip pattern and computed count that do not add up to dev's schedule."""
    cf._sweep_worker("dev", "t0.200", chunk_dir=tmp_path, dry_run=True)
    chunk = json.loads((tmp_path / cf._sweep_chunk_filename("dev", "t0.200")).read_text())
    steps = cf.RECIPES["dev"]["num_inference_steps"]
    assert len(chunk["skip_pattern"]) == steps
    assert chunk["skipped"] + chunk["computed"] == steps
