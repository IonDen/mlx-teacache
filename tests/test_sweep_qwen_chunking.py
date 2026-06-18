"""Unit tests for the chunked/resumable sweep plumbing in
`scripts/sweep_threshold_qwen.py` — unit naming, run order (vanilla first),
resume (pending), and summary aggregation. Pure functions: no weights, no
generation.

Importing the script module pulls scikit-image (SSIM) at module top, so this runs
in the mflux lane (where scikit-image is installed).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sweep_threshold_qwen as sw  # noqa: E402

pytestmark = pytest.mark.mflux


def test_threshold_name_three_decimals() -> None:
    assert sw._threshold_name(0.05) == "t0.050"
    assert sw._threshold_name(0.4) == "t0.400"


def test_units_vanilla_first() -> None:
    units = sw._units()
    assert units[0] == "vanilla"
    assert units[1:] == [sw._threshold_name(t) for t in sw.THRESHOLDS]
    assert len(units) == 1 + len(sw.THRESHOLDS)


def test_pending_units_preserves_run_order_and_skips_existing(tmp_path: Path) -> None:
    units = sw._units()
    (tmp_path / sw._chunk_filename("vanilla")).write_text("{}")
    (tmp_path / sw._chunk_filename(units[1])).write_text("{}")
    pending = sw._pending_units(tmp_path, units)
    assert "vanilla" not in pending
    assert units[1] not in pending
    assert pending == units[2:]  # remaining thresholds, order preserved


def test_pending_all_done_is_empty(tmp_path: Path) -> None:
    units = sw._units()
    for u in units:
        (tmp_path / sw._chunk_filename(u)).write_text("{}")
    assert sw._pending_units(tmp_path, units) == []


def test_build_summary_sorts_by_threshold_and_derives_speedup() -> None:
    chunks = [
        {"threshold": 0.25, "wrapper_seconds": 5.0, "skipped": 6, "computed": 44, "ssim_vs_vanilla": 0.99},
        {"threshold": 0.05, "wrapper_seconds": 8.0, "skipped": 1, "computed": 49, "ssim_vs_vanilla": 0.999},
    ]
    summary = sw._build_summary(chunks, vanilla_seconds=10.0)
    ts = summary["thresholds"]
    assert [r["threshold"] for r in ts] == [0.05, 0.25]  # ascending
    assert ts[0]["speedup_vs_vanilla_single_rep"] == 10.0 / 8.0
    assert ts[1]["speedup_vs_vanilla_single_rep"] == 10.0 / 5.0
    assert summary["vanilla_seconds"] == 10.0
    assert summary["signal"] == "A"
    assert summary["num_inference_steps"] == sw.STEPS
