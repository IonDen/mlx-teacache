"""Unit tests for the chunked/resumable calibration plumbing in
`scripts/calibrate_qwen.py` — the resume (pending), fit/held split (n_fit),
aggregation (accumulate), and the dry-run/clobber-guard logic. Pure functions:
no weights, no generation, no MLX state mutation.

Importing the script module pulls mflux (via the variant integration import), so
this runs in the mflux lane.
"""

import ast
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from mlx_teacache._kernel.gate import mean_abs_rel_l1

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


def _held_arrays(obj: object) -> int:
    """Count the mx.arrays an object keeps alive through its attributes."""

    def walk(v: object) -> int:
        if isinstance(v, mx.array):
            return 1
        if isinstance(v, dict):
            return sum(walk(x) for x in v.values())
        if isinstance(v, (list, tuple)):
            return sum(walk(x) for x in v)
        return 0

    return walk(vars(obj))


def _normal(seed: int) -> mx.array:
    return mx.random.normal((1, 16, 8), key=mx.random.key(seed))


def _rel(arrays: list[mx.array]) -> list[float]:
    return [mean_abs_rel_l1(arrays[t], arrays[t - 1]) for t in range(1, len(arrays))]


def _steps(n: int = 6) -> dict[str, list[mx.array]]:
    return {
        "A": [_normal(i) for i in range(n)],
        "B": [_normal(100 + i) for i in range(n)],
        "pos": [_normal(200 + i) for i in range(n)],
        # neg wins the worst-branch max on some steps only
        "neg": [_normal(300 + i) * (2.0 if i % 2 else 0.5) for i in range(n)],
    }


def _assert_matches_retain_then_reduce(reducer: Any, seq: dict[str, list[mx.array]]) -> None:
    expected_y = [max(p, n) for p, n in zip(_rel(seq["pos"]), _rel(seq["neg"]), strict=True)]
    for sig in ("A", "B"):
        xs, ys = reducer.series(sig)
        assert len(xs) == len(ys) == len(seq["A"]) - 1
        assert max(abs(g - e) for g, e in zip(xs, _rel(seq[sig]), strict=True)) <= 1e-7
        assert max(abs(g - e) for g, e in zip(ys, expected_y, strict=True)) <= 1e-7


def test_online_reducer_pairs_match_retain_then_reduce() -> None:
    """bug caught: the online reducer dropping the negative branch from the worst-branch
    target, taking a signal from the wrong array, or pairing step t with a stale step,
    versus the old path that kept all 50 steps (8 arrays each) and reduced afterwards."""
    seq = _steps()
    reducer = cq._QwenPairReducer()
    for a, b, p, n in zip(seq["A"], seq["B"], seq["pos"], seq["neg"], strict=True):
        reducer.positive(signal_A=a, signal_B=b, body_out=p)
        reducer.negative(body_out=n)
    reducer.finish()
    _assert_matches_retain_then_reduce(reducer, seq)
    assert reducer.steps == 6


def test_online_reducer_holds_one_step_of_arrays() -> None:
    """bug caught: the reducer keeping every step's activations until generate_image
    returns (~11 GB at 768x768 over 50 steps, which trips the 28 GiB watchdog)."""
    seq = _steps()
    reducer = cq._QwenPairReducer()
    for a, b, p, n in zip(seq["A"], seq["B"], seq["pos"], seq["neg"], strict=True):
        reducer.positive(signal_A=a, signal_B=b, body_out=p)
        reducer.negative(body_out=n)
    assert _held_arrays(reducer) == 4  # step t-1's signal A, signal B, body_out pos and neg


def test_online_reducer_step_without_negative_branch_targets_the_positive_one() -> None:
    """bug caught: a positive-only step (no CFG call) being dropped instead of closed by
    the next positive call or finish(); the old path fell back to the positive body_out."""
    pos = [_normal(200 + i) for i in range(4)]
    reducer = cq._QwenPairReducer()
    for i in range(4):
        reducer.positive(signal_A=_normal(i), signal_B=_normal(100 + i), body_out=pos[i])
    reducer.finish()
    _, ys = reducer.series("A")
    assert reducer.steps == 4
    assert max(abs(g - e) for g, e in zip(ys, _rel(pos), strict=True)) <= 1e-7


def test_capturing_transformer_routes_the_two_calls_per_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """bug caught: the branch-parity counter flipped or not advanced, so the negative
    call's body_out lands in the positive slot (mflux calls the transformer twice per
    step: positive prompt, then negative)."""
    seq = _steps(4)
    calls = iter(
        [
            {
                "signal_A": seq["A"][i],
                "signal_B": seq["B"][i],
                "body_out": seq["pos"][i],
                "noise": mx.ones((2,)),
            }
            if branch == "pos"
            else {
                "signal_A": _normal(900 + i),
                "signal_B": _normal(950 + i),
                "body_out": seq["neg"][i],
                "noise": mx.ones((2,)),
            }
            for i in range(4)
            for branch in ("pos", "neg")
        ]
    )
    monkeypatch.setattr(cq, "_qwen_capture_branch", lambda inner, **kw: next(calls))
    reducer = cq._QwenPairReducer()
    wrapper = cq._CapturingTransformer(lambda **kw: mx.ones((2,)), reducer, {"done": False})
    for _ in range(8):
        wrapper(
            t=0,
            config=None,
            hidden_states=mx.zeros((1,)),
            encoder_hidden_states=mx.zeros((1,)),
            encoder_hidden_states_mask=mx.zeros((1,)),
        )
    reducer.finish()
    _assert_matches_retain_then_reduce(reducer, seq)
    assert reducer.steps == 4


def test_memory_saver_frees_the_text_encoders_with_the_load_bearing_kwargs() -> None:
    """bug caught: the calibration worker not registering mflux's MemorySaver, or building
    it with the default cache_limit_bytes (1 GB branch: overrides our cache cap, resets
    the peak counter) or keep_transformer=False (deletes the transformer mid-capture)."""
    registered: list[Any] = []

    class _Callbacks:
        def register(self, cb: Any) -> None:
            registered.append(cb)

    class _Flux:
        callbacks = _Callbacks()

    class _FakeSaver:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw

    flux = _Flux()
    cq._register_memory_saver(flux, _FakeSaver)
    assert len(registered) == 1
    assert registered[0].kw == dict(model=flux, keep_transformer=True, cache_limit_bytes=None, num_seeds=1)


def test_every_install_caps_call_bounds_the_cache_pool_to_one_gib() -> None:
    """bug caught: the Qwen calibrator keeping the 2 GiB default pool; at ~26 GiB active the
    1 GiB pool is what keeps it under the 28 GiB watchdog ceiling (bench and sweep use 1.0)."""
    tree = ast.parse(Path(cq.__file__).read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "install_caps"
    ]
    assert len(calls) == 2  # the worker and the memory probe
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert isinstance(kw.get("cache_gb"), ast.Constant) and kw["cache_gb"].value == 1.0


def test_worker_chunk_records_the_peak_memory() -> None:
    """bug caught: the chunk JSON losing peak_memory_gb, the only record of what the
    online-reducer capture actually peaks at on real weights."""
    reducer = cq._QwenPairReducer()
    for i in range(3):
        reducer.positive(signal_A=_normal(i), signal_B=_normal(100 + i), body_out=_normal(200 + i))
        reducer.negative(body_out=_normal(300 + i))
    chunk = cq._chunk_from_reducer(reducer, prompt_idx=4, prompt="p", steps=3)
    assert chunk["idx"] == 4 and chunk["prompt"] == "p" and chunk["num_captures"] == 3 and chunk["steps"] == 3
    assert len(chunk["signal_A"]["xs"]) == 2 and len(chunk["signal_B"]["ys"]) == 2
    assert isinstance(chunk["peak_memory_gb"], float) and chunk["peak_memory_gb"] > 0.0
