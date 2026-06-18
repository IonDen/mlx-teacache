"""Unit tests for the chunked/resumable calibration plumbing in
`scripts/calibrate_qwen.py` — the resume (pending), fit/held split (n_fit),
aggregation (accumulate), and the dry-run/clobber-guard logic. Pure functions:
no weights, no generation, no MLX state mutation.

Importing the script module pulls mflux (via the variant integration import), so
this runs in the mflux lane.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_qwen as cq  # noqa: E402

pytestmark = pytest.mark.mflux


def test_chunk_filename_zero_padded() -> None:
    assert cq._chunk_filename(0) == "prompt_00.json"
    assert cq._chunk_filename(7) == "prompt_07.json"
    assert cq._chunk_filename(10) == "prompt_10.json"


def test_pending_indices_empty_dir_all_pending(tmp_path: Path) -> None:
    assert cq._pending_prompt_indices(tmp_path, 5) == [0, 1, 2, 3, 4]


def test_pending_indices_skips_existing(tmp_path: Path) -> None:
    (tmp_path / cq._chunk_filename(0)).write_text("{}")
    (tmp_path / cq._chunk_filename(2)).write_text("{}")
    assert cq._pending_prompt_indices(tmp_path, 4) == [1, 3]


def test_pending_indices_all_done_is_empty(tmp_path: Path) -> None:
    for i in range(3):
        (tmp_path / cq._chunk_filename(i)).write_text("{}")
    assert cq._pending_prompt_indices(tmp_path, 3) == []


def test_n_fit_normal_and_small() -> None:
    assert cq._n_fit(10, 3) == 7  # the real 7-fit / 3-held split
    assert cq._n_fit(2, 3) == 1  # held-out shrinks first; always >= 1 fit
    assert cq._n_fit(1, 3) == 1


def test_accumulate_splits_fit_and_held_by_idx() -> None:
    chunks = [
        {"idx": 0, "signal_A": {"xs": [0.1], "ys": [0.2]}, "signal_B": {"xs": [1.1], "ys": [1.2]}},
        {"idx": 1, "signal_A": {"xs": [0.3], "ys": [0.4]}, "signal_B": {"xs": [1.3], "ys": [1.4]}},
        {"idx": 2, "signal_A": {"xs": [0.5], "ys": [0.6]}, "signal_B": {"xs": [1.5], "ys": [1.6]}},
    ]
    acc = cq._accumulate_chunks(chunks, n_fit=2)  # idx 0,1 -> fit ; idx 2 -> held
    assert acc["A"]["fit_x"] == [0.1, 0.3]
    assert acc["A"]["fit_y"] == [0.2, 0.4]
    assert acc["A"]["held_x"] == [0.5]
    assert acc["A"]["held_y"] == [0.6]
    assert acc["B"]["fit_x"] == [1.1, 1.3]
    assert acc["B"]["held_x"] == [1.5]


def test_accumulate_sorts_by_idx_regardless_of_input_order() -> None:
    chunks = [
        {"idx": 2, "signal_A": {"xs": [0.5], "ys": [0.6]}, "signal_B": {"xs": [1.5], "ys": [1.6]}},
        {"idx": 0, "signal_A": {"xs": [0.1], "ys": [0.2]}, "signal_B": {"xs": [1.1], "ys": [1.2]}},
        {"idx": 1, "signal_A": {"xs": [0.3], "ys": [0.4]}, "signal_B": {"xs": [1.3], "ys": [1.4]}},
    ]
    acc = cq._accumulate_chunks(chunks, n_fit=2)
    assert acc["A"]["fit_x"] == [0.1, 0.3]  # sorted: idx0, then idx1
    assert acc["A"]["held_x"] == [0.5]


def test_aggregate_path_real_default_is_the_committed_json() -> None:
    p = cq._aggregate_path(cq.CHUNK_DIR_DEFAULT, dry_run=False)
    assert p.name == cq.OUTPUT_JSON
    assert p.parent == Path(cq.__file__).parent  # scripts/_calibration_qwen.json


def test_aggregate_path_dry_run_never_clobbers_committed(tmp_path: Path) -> None:
    # a custom chunk dir writes beside its chunks
    assert cq._aggregate_path(tmp_path, dry_run=True) == tmp_path / cq.OUTPUT_JSON
    # even the DEFAULT dir under dry-run must not point at the committed file
    assert cq._aggregate_path(cq.CHUNK_DIR_DEFAULT, dry_run=True) == cq.CHUNK_DIR_DEFAULT / cq.OUTPUT_JSON
