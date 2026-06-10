# tests/test_gate.py
"""Unit tests for gate.py — the structured GateDecision contract.

Per spec §5.3:
- Hard short-circuit at rel_l1_thresh <= 0 → always 'computed', no polynomial eval.
- Forced windows → 'forced', should_compute=True, should_update_cache=False.
- nan/inf in mod_in → 'numerical-miss', should_compute=True, should_update_cache=False.
- First step with previous_mod_input is None → 'computed', should_update_cache=True.
- Skip path: predicted_distance clamped at 0 (monotonic non-decreasing accumulator)."""

import mlx.core as mx
import pytest

from mlx_teacache.cache import TeaCacheState
from mlx_teacache.gate import gate_step, mean_abs_rel_l1, poly_eval

COEFFS = (1.0, -1.0, 0.5, 2.0, 0.0)  # x⁴ - x³ + 0.5x² + 2x


def test_poly_eval_zero_intercept():
    assert poly_eval(COEFFS, 0.0) == 0.0


def test_poly_eval_known_value():
    # at x=1: 1 - 1 + 0.5 + 2 + 0 = 2.5
    assert poly_eval(COEFFS, 1.0) == pytest.approx(2.5)


def test_poly_eval_negative_x_can_be_negative():
    # at x=-1: 1 + 1 + 0.5 - 2 + 0 = 0.5
    assert poly_eval(COEFFS, -1.0) == pytest.approx(0.5)
    # find a coefficient set where this goes negative
    neg_coeffs = (0.0, 0.0, 0.0, -1.0, 0.0)
    assert poly_eval(neg_coeffs, 0.5) == pytest.approx(-0.5)


def test_mean_abs_rel_l1_value():
    cur = mx.array([2.0, 2.0, 2.0, 2.0])
    prev = mx.array([1.0, 1.0, 1.0, 1.0])
    # mean(|cur-prev|)=1.0, mean(|prev|)=1.0 -> exactly 1.0
    assert mean_abs_rel_l1(cur, prev) == pytest.approx(1.0, rel=1e-9)
    # division-guard branch: prev all-zero -> denom clamps to 1e-12 -> 1e12
    big = mean_abs_rel_l1(mx.array([1.0, 1.0]), mx.array([0.0, 0.0]))
    assert big == pytest.approx(1e12, rel=1e-6)


def test_mean_abs_rel_l1_identical_inputs_zero():
    a = mx.array([[1.0, 2.0], [3.0, 4.0]])
    val = mean_abs_rel_l1(a, a)
    assert val == pytest.approx(0.0)


def _fresh_state(num_steps: int = 25) -> TeaCacheState:
    s = TeaCacheState()
    s.reset_for_new_generation(num_steps=num_steps)
    return s


def test_threshold_zero_short_circuit_returns_computed_no_poly_eval():
    state = _fresh_state()
    mod_in = mx.ones((1, 16, 64))
    dec = gate_step(
        state,
        rel_l1_thresh=0.0,
        coefficients=COEFFS,
        skip_first=0,
        skip_last=0,
        num_steps=25,
        step_idx=5,
        mod_in=mod_in,
    )
    assert dec.kind == "computed"
    assert dec.should_compute is True
    # At threshold<=0 the cache can never be consumed (no future step can
    # skip), so the gate must NOT request a cache update — building the
    # residual would keep body/tail intermediates alive past the tail and
    # perturb Metal in-place buffer donation.
    assert dec.should_update_cache is False
    assert dec.rel_l1 is None
    assert dec.predicted_distance is None


def test_threshold_zero_with_pathological_negative_coeffs_still_no_skip():
    state = _fresh_state()
    state.previous_mod_input = mx.ones((1, 16, 64)) * 0.5
    mod_in = mx.ones((1, 16, 64))
    neg_coeffs = (0.0, 0.0, 0.0, -100.0, 0.0)
    dec = gate_step(
        state,
        rel_l1_thresh=0.0,
        coefficients=neg_coeffs,
        skip_first=0,
        skip_last=0,
        num_steps=25,
        step_idx=5,
        mod_in=mod_in,
    )
    assert dec.kind == "computed"


def test_forced_first_window_no_cache_update():
    state = _fresh_state()
    mod_in = mx.ones((1, 16, 64))
    dec = gate_step(
        state,
        rel_l1_thresh=0.25,
        coefficients=COEFFS,
        skip_first=2,
        skip_last=1,
        num_steps=25,
        step_idx=0,
        mod_in=mod_in,
    )
    assert dec.kind == "forced"
    assert dec.should_compute is True
    assert dec.should_update_cache is False


def test_forced_last_window():
    state = _fresh_state()
    mod_in = mx.ones((1, 16, 64))
    dec = gate_step(
        state,
        rel_l1_thresh=0.25,
        coefficients=COEFFS,
        skip_first=1,
        skip_last=1,
        num_steps=25,
        step_idx=24,
        mod_in=mod_in,
    )
    assert dec.kind == "forced"


def test_numerical_miss_on_nan_in_mod_in():
    state = _fresh_state()
    state.previous_mod_input = mx.ones((1, 16, 64))
    mod_in = mx.full((1, 16, 64), float("nan"))
    dec = gate_step(
        state,
        rel_l1_thresh=0.25,
        coefficients=COEFFS,
        skip_first=1,
        skip_last=1,
        num_steps=25,
        step_idx=5,
        mod_in=mod_in,
    )
    assert dec.kind == "numerical-miss"
    assert dec.should_compute is True
    assert dec.should_update_cache is False


def test_first_eligible_step_no_previous_is_computed_with_cache_update():
    state = _fresh_state()
    mod_in = mx.ones((1, 16, 64))
    dec = gate_step(
        state,
        rel_l1_thresh=0.25,
        coefficients=COEFFS,
        skip_first=1,
        skip_last=1,
        num_steps=25,
        step_idx=1,
        mod_in=mod_in,
    )
    assert dec.kind == "computed"
    assert dec.should_update_cache is True


def test_skip_decision_below_threshold():
    state = _fresh_state()
    state.previous_mod_input = mx.ones((1, 16, 64))
    # Small change ⇒ small predicted distance ⇒ acc stays below thresh
    mod_in = state.previous_mod_input + 0.001
    dec = gate_step(
        state,
        rel_l1_thresh=10.0,
        coefficients=COEFFS,
        skip_first=1,
        skip_last=1,
        num_steps=25,
        step_idx=5,
        mod_in=mod_in,
    )
    assert dec.kind == "skipped"
    assert dec.should_compute is False
    assert dec.should_update_cache is False


def test_compute_decision_resets_accumulator():
    state = _fresh_state()
    state.previous_mod_input = mx.ones((1, 16, 64))
    state.accumulated_distance = 0.1
    # Large change ⇒ predicted distance pushes acc over thresh
    mod_in = state.previous_mod_input * 10.0
    dec = gate_step(
        state,
        rel_l1_thresh=0.5,
        coefficients=COEFFS,
        skip_first=1,
        skip_last=1,
        num_steps=25,
        step_idx=5,
        mod_in=mod_in,
    )
    assert dec.kind == "computed"
    assert state.accumulated_distance == 0.0  # reset on compute


def test_predicted_distance_clamped_at_zero():
    state = _fresh_state()
    state.previous_mod_input = mx.ones((1, 16, 64))
    mod_in = state.previous_mod_input + 0.01
    neg_coeffs = (0.0, 0.0, 0.0, -100.0, 0.0)
    dec = gate_step(
        state,
        rel_l1_thresh=0.5,
        coefficients=neg_coeffs,
        skip_first=1,
        skip_last=1,
        num_steps=25,
        step_idx=5,
        mod_in=mod_in,
    )
    # predicted would be ≤ 0; clamped to 0; acc unchanged; below thresh → skip
    assert dec.kind == "skipped"
    assert dec.predicted_distance is not None
    assert dec.predicted_distance == 0.0
