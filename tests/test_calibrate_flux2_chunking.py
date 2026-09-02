"""Unit tests for the chunked/resumable calibration plumbing in
`scripts/calibrate_flux2.py` — chunk naming, resume (pending), the CFG
fit-branch-policy selection, aggregation, the polynomial fit, and the
dry-run/clobber-guard logic. Pure functions: no weights, no generation, no
MLX state mutation.

Importing the script module pulls mflux (via the variant integration import),
so this runs in the mflux lane.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_flux2 as cf  # noqa: E402

pytestmark = pytest.mark.mflux


def test_chunk_filename_zero_padded() -> None:
    assert cf._chunk_filename("klein-4b", 0) == "klein-4b_prompt00.json"
    assert cf._chunk_filename("klein-base-4b", 7) == "klein-base-4b_prompt07.json"
    assert cf._chunk_filename("klein-9b", 10) == "klein-9b_prompt10.json"


def test_pending_indices_empty_dir_all_pending(tmp_path: Path) -> None:
    assert cf._pending_prompt_indices(tmp_path, "klein-4b", 5) == [0, 1, 2, 3, 4]


def test_pending_indices_skips_existing(tmp_path: Path) -> None:
    (tmp_path / cf._chunk_filename("klein-4b", 0)).write_text("{}")
    (tmp_path / cf._chunk_filename("klein-4b", 2)).write_text("{}")
    assert cf._pending_prompt_indices(tmp_path, "klein-4b", 4) == [1, 3]


def test_pending_indices_all_done_is_empty(tmp_path: Path) -> None:
    for i in range(3):
        (tmp_path / cf._chunk_filename("klein-4b", i)).write_text("{}")
    assert cf._pending_prompt_indices(tmp_path, "klein-4b", 3) == []


def test_pending_indices_is_per_variant(tmp_path: Path) -> None:
    # A chunk for one variant must not mask a pending index for another
    # variant sharing the same chunk directory.
    (tmp_path / cf._chunk_filename("klein-4b", 0)).write_text("{}")
    assert cf._pending_prompt_indices(tmp_path, "klein-9b", 1) == [0]


def test_select_y_worst_average_positive_negative() -> None:
    assert cf._select_y(0.3, 0.5, policy="worst") == 0.5
    assert cf._select_y(0.3, 0.5, policy="average") == 0.4
    assert cf._select_y(0.3, 0.5, policy="positive") == 0.3
    assert cf._select_y(0.3, 0.5, policy="negative") == 0.5


def test_select_y_unknown_policy_raises() -> None:
    with pytest.raises(ValueError):
        cf._select_y(0.1, 0.2, policy="bogus")


def test_accumulate_non_cfg_concatenates_in_idx_order() -> None:
    chunks = [
        {"idx": 1, "xs": [0.3], "ys": [0.4]},
        {"idx": 0, "xs": [0.1], "ys": [0.2]},
    ]
    acc = cf._accumulate_chunks(chunks, cfg=False)
    assert acc["xs"] == [0.1, 0.3]
    assert acc["ys"] == [0.2, 0.4]
    assert "ys_pos" not in acc


def test_accumulate_cfg_applies_policy_per_pair() -> None:
    chunks = [
        {"idx": 0, "xs": [0.1, 0.2], "ys_pos": [0.3, 0.5], "ys_neg": [0.7, 0.1]},
    ]
    acc = cf._accumulate_chunks(chunks, cfg=True, fit_branch_policy="worst")
    assert acc["xs"] == [0.1, 0.2]
    assert acc["ys_pos"] == [0.3, 0.5]
    assert acc["ys_neg"] == [0.7, 0.1]
    assert acc["ys"] == [0.7, 0.5]  # max(0.3,0.7), max(0.5,0.1)


def test_accumulate_cfg_average_policy() -> None:
    chunks = [{"idx": 0, "xs": [0.1], "ys_pos": [0.2], "ys_neg": [0.4]}]
    acc = cf._accumulate_chunks(chunks, cfg=True, fit_branch_policy="average")
    assert acc["ys"] == pytest.approx([0.3])


def test_fit_polynomial_origin_forces_zero_intercept() -> None:
    xs = [0.0, 0.1, 0.2, 0.3, 0.4]
    ys = [0.0, 0.2, 0.4, 0.6, 0.8]  # y = 2x, exactly fits an origin-constrained line
    coeffs, r2 = cf._fit_polynomial(xs, ys, fit_mode="origin")
    assert len(coeffs) == 5
    assert coeffs[-1] == 0.0  # c0 forced to zero
    assert r2 == pytest.approx(1.0, abs=1e-6)


def test_fit_polynomial_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        cf._fit_polynomial([0.1], [0.2], fit_mode="bogus")


def test_aggregate_path_real_default_is_the_committed_json() -> None:
    p = cf._aggregate_path(cf.CHUNK_DIR_DEFAULT, "klein-4b", dry_run=False)
    assert p.name == cf._VARIANTS["klein-4b"]["output_json"]
    assert p.parent == Path(cf.__file__).parent


def test_aggregate_path_dry_run_never_clobbers_committed(tmp_path: Path) -> None:
    output_json = cf._VARIANTS["klein-4b"]["output_json"]
    assert cf._aggregate_path(tmp_path, "klein-4b", dry_run=True) == tmp_path / output_json
    assert (
        cf._aggregate_path(cf.CHUNK_DIR_DEFAULT, "klein-4b", dry_run=True)
        == cf.CHUNK_DIR_DEFAULT / output_json
    )


def test_aggregate_path_custom_chunk_dir_never_clobbers_committed(tmp_path: Path) -> None:
    output_json = cf._VARIANTS["klein-base-4b"]["output_json"]
    assert cf._aggregate_path(tmp_path, "klein-base-4b", dry_run=False) == tmp_path / output_json
