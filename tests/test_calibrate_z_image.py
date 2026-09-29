"""Unit tests for the pure helpers in scripts/calibrate_z_image.py.

Pure-core (mflux-free): imports the calibration script (mlx + numpy + the
model-agnostic _kernel.gate only; mflux is imported lazily inside main()).
NOT added to conftest._MFLUX_FILES — runs in the pure-core lane.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_teacache._kernel.gate import mean_abs_rel_l1
from tests.test_forward_z_image_fake import _CAP_SEQ, _DIM, _X_SEQ, _FakeZImageTransformer

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import calibrate_z_image as cz  # noqa: E402  (after sys.path setup)
from calibrate_z_image import fit_signal  # noqa: E402


def test_fit_signal_free_recovers_known_degree4_polynomial():
    true = [2.0, -0.5, 1.0, 3.0, 0.7]  # c4..c0
    xs = list(np.linspace(0.0, 1.0, 50))
    ys = [float(np.poly1d(true)(x)) for x in xs]
    out = fit_signal(xs, ys, fit_mode="free")
    assert out["fit_r_squared"] > 0.9999
    for got, exp in zip(out["coefficients_c4_to_c0"], true, strict=True):
        assert abs(got - exp) < 1e-3


def test_fit_signal_origin_forces_c0_zero():
    xs = list(np.linspace(0.05, 1.0, 40))
    ys = [float(0.8 * x + 0.3 * x**2) for x in xs]
    out = fit_signal(xs, ys, fit_mode="origin")
    assert out["coefficients_c4_to_c0"][-1] == 0.0  # c0 forced through the origin
    assert len(out["coefficients_c4_to_c0"]) == 5


def test_fit_signal_reports_curve_range_and_pairs():
    # >=5 points: a degree-4 fit on fewer is rank-deficient (RankWarning -> error
    # under the repo's filterwarnings=error). Real calibration has hundreds.
    xs = list(np.linspace(0.1, 0.4, 10))
    ys = list(np.linspace(0.0, 0.9, 10))
    out = fit_signal(xs, ys, fit_mode="free")
    assert out["x_min"] == pytest.approx(0.1)
    assert out["x_max"] == pytest.approx(0.4)
    assert out["y_min"] == pytest.approx(0.0)
    assert out["y_max"] == pytest.approx(0.9)
    assert out["n_pairs"] == 10


def test_fit_signal_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown fit_mode"):
        fit_signal([0.1, 0.2], [0.1, 0.2], fit_mode="bogus")


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


def test_online_reducer_pairs_match_retain_then_reduce():
    """bug caught: the online reducer fitting y on the positive branch only (dropping the
    worst-branch max), taking signal B from the wrong array, or pairing step t with a
    stale step, versus the old path that kept every step and reduced afterwards."""
    sig_a = [_normal(i) for i in range(6)]
    sig_b = [_normal(100 + i) for i in range(6)]
    pos = [_normal(200 + i) for i in range(6)]
    neg = [_normal(300 + i) * (2.0 if i % 2 else 0.5) for i in range(6)]  # neg wins on some steps only
    reducer = cz._ZImagePairReducer()
    for a, b, p, n in zip(sig_a, sig_b, pos, neg, strict=True):
        reducer.push(signal_A=a, signal_B=b, main_out_pos=p, main_out_neg=n)
    expected_y = [max(yp, yn) for yp, yn in zip(_rel(pos), _rel(neg), strict=True)]
    for sig, arrays in (("A", sig_a), ("B", sig_b)):
        xs, ys = reducer.series(sig)
        assert max(abs(g - e) for g, e in zip(xs, _rel(arrays), strict=True)) <= 1e-7
        assert max(abs(g - e) for g, e in zip(ys, expected_y, strict=True)) <= 1e-7
        assert len(xs) == len(ys) == 5
    assert reducer.steps == 6


def test_online_reducer_holds_one_step_of_arrays():
    """bug caught: the reducer keeping every step's signals and main outputs alive until
    the generation ends (the whole-prompt retention the online path removes)."""
    reducer = cz._ZImagePairReducer()
    for i in range(6):
        reducer.push(
            signal_A=_normal(i),
            signal_B=_normal(100 + i),
            main_out_pos=_normal(200 + i),
            main_out_neg=_normal(300 + i),
        )
    assert _held_arrays(reducer) == 4  # step t-1's signal A, signal B, main_out pos and neg


def test_online_reducer_without_negative_branch_targets_the_positive_one():
    """bug caught: a no-CFG step (main_out_neg None) storing None as the negative branch,
    so the next step's rel-L1 crashes; the old path fell back to the positive main_out."""
    pos = [_normal(200 + i) for i in range(4)]
    reducer = cz._ZImagePairReducer()
    for i in range(4):
        reducer.push(signal_A=_normal(i), signal_B=_normal(100 + i), main_out_pos=pos[i])
    _, ys = reducer.series("A")
    assert max(abs(g - e) for g, e in zip(ys, _rel(pos), strict=True)) <= 1e-7


def test_capture_forward_taps_on_the_runtime_prelude():
    """bug caught: signal A sliced from the caption part of the unified stream, signal B
    measured from the full body instead of layer 0, or main_out missing a main layer.

    The fake prelude is the identity, so unified_in = concat(latents=1, caption=7);
    layers add 1, 10, 100 (see tests/test_forward_z_image_fake.py)."""
    transformer = _FakeZImageTransformer()
    out = cz._zimage_capture_forward(
        transformer,
        mx.full((_X_SEQ, _DIM), 1.0),
        mx.array([0.5]),
        mx.full((_CAP_SEQ, _DIM), 7.0),
    )
    assert mx.array_equal(out["signal_A"], mx.full((1, _X_SEQ, _DIM), 1.0)).item()
    assert mx.array_equal(out["signal_B"], mx.full((1, _X_SEQ + _CAP_SEQ, _DIM), 1.0)).item()
    main_expected = mx.concatenate(
        [mx.full((1, _X_SEQ, _DIM), 112.0), mx.full((1, _CAP_SEQ, _DIM), 118.0)], axis=1
    )
    assert mx.array_equal(out["main_out"], main_expected).item()
    assert mx.array_equal(out["noise"], mx.full((_X_SEQ, _DIM), -112.0)).item()


def test_calibrators_do_not_reimplement_the_forward_or_the_gate_signal():
    """bug caught: a calibrator walking the Z-Image refiners itself, or computing its own
    sum-ratio rel-L1, so a recalibration fits a forward or a signal the runtime does not
    use. The runtime helpers and mean_abs_rel_l1 are imported instead."""
    offenders = []
    for path in sorted(_SCRIPTS.glob("calibrate_*.py")):
        text = path.read_text()
        for needle in ("transformer.noise_refiner", "transformer.context_refiner", "mx.sum(mx.abs("):
            if needle in text:
                offenders.append(f"{path.name}: {needle}")
    assert offenders == []
