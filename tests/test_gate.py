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
    # v0.10.0: consecutive-delta anchoring (Option A) — a skip requires a
    # cached residual to reuse; seed it so the threshold path (not the
    # seed/re-seed guard) is exercised.
    state.cached_residual = mx.ones((1, 16, 64))
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
    # v0.10.0: consecutive-delta anchoring (Option A) — seed a cached residual
    # so this reaches the threshold path (not the seed/re-seed guard, which
    # doesn't touch accumulated_distance).
    state.cached_residual = mx.ones((1, 16, 64))
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
    # v0.10.0: consecutive-delta anchoring (Option A) — seed a cached residual
    # so a skip decision can actually be issued (seed/re-seed guard requires one).
    state.cached_residual = mx.ones((1, 16, 64))
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


def _seeded_state() -> TeaCacheState:
    """State as it looks after the first computed+cached step."""
    state = TeaCacheState()
    state.previous_mod_input = mx.ones((4,))
    state.cached_residual = mx.ones((4,))
    return state


_FLAT_COEFFS = (0.0, 0.0, 0.0, 0.0, 0.0)  # poly ≡ 0 → accumulator never grows
_GATE_KWARGS = dict(
    rel_l1_thresh=0.5,
    skip_first=0,
    skip_last=0,
    num_steps=100,
    mod_in=mx.ones((4,)) * 1.01,
)


def test_anchor_advances_on_a_skipped_step():
    """Upstream-faithful (Option A): previous_mod_input advances every gated
    step, so rel_l1 is always the consecutive delta the polynomial was
    calibrated on — not cumulative drift since the last compute."""
    state = _seeded_state()
    mod_in = mx.ones((4,)) * 1.01
    decision = gate_step(state, coefficients=_FLAT_COEFFS, step_idx=1, **{**_GATE_KWARGS, "mod_in": mod_in})
    assert decision.kind == "skipped"
    assert state.previous_mod_input is mod_in


def test_rel_l1_is_consecutive_delta_across_a_skip_run():
    """Three steps with equal successive deltas must report equal rel_l1 on
    each gated step (consecutive anchoring). Cumulative anchoring would
    report a growing sequence."""
    state = _seeded_state()
    state.previous_mod_input = mx.full((4,), 1.00)
    rel_l1s = []
    for i, scale in enumerate((1.01, 1.02, 1.03), start=1):
        decision = gate_step(
            state,
            coefficients=_FLAT_COEFFS,
            step_idx=i,
            **{**_GATE_KWARGS, "mod_in": mx.full((4,), scale)},
        )
        rel_l1s.append(decision.rel_l1)
    assert all(r is not None for r in rel_l1s)
    # consecutive deltas: |1.01-1.00|/1.00, |1.02-1.01|/1.01, |1.03-1.02|/1.02
    expected = [0.01 / 1.00, 0.01 / 1.01, 0.01 / 1.02]
    for got, want in zip(rel_l1s, expected, strict=True):
        assert abs(got - want) < 1e-6, (got, want)


def test_skip_requires_a_cached_residual():
    """Anchor and residual are decoupled now; a skip without a residual to
    reuse must never be issued — the gate computes and seeds instead."""
    state = TeaCacheState()
    state.previous_mod_input = mx.ones((4,))  # anchor set (e.g. by a forced step)
    state.cached_residual = None
    decision = gate_step(state, coefficients=_FLAT_COEFFS, step_idx=1, **_GATE_KWARGS)
    assert decision.should_compute is True
    assert decision.should_update_cache is True


def test_consecutive_skip_streak_forces_compute_at_cap():
    """Runaway guard (audit H1): a polynomial that never grows the
    accumulator must not skip unboundedly on a stale residual."""
    from mlx_teacache._kernel.gate import MAX_CONSECUTIVE_SKIPS

    state = _seeded_state()
    kinds = []
    for i in range(1, MAX_CONSECUTIVE_SKIPS + 2):
        decision = gate_step(state, coefficients=_FLAT_COEFFS, step_idx=i, **_GATE_KWARGS)
        kinds.append(decision.kind)
    assert kinds[:MAX_CONSECUTIVE_SKIPS] == ["skipped"] * MAX_CONSECUTIVE_SKIPS
    assert kinds[MAX_CONSECUTIVE_SKIPS] == "computed"
    assert state.consecutive_skips == 0


def test_forced_step_advances_anchor_but_not_streak():
    """Forced-window steps carry a usable signal: the anchor advances so the
    next gated rel_l1 is a true consecutive delta, but cache/accumulator/
    streak are untouched (forced outputs never update the cache)."""
    state = _seeded_state()
    state.consecutive_skips = 3
    mod_in = mx.ones((4,)) * 2.0
    decision = gate_step(
        state,
        coefficients=_FLAT_COEFFS,
        rel_l1_thresh=0.5,
        skip_first=5,
        skip_last=0,
        num_steps=100,
        step_idx=2,  # inside the skip_first window
        mod_in=mod_in,
    )
    assert decision.kind == "forced"
    assert state.previous_mod_input is mod_in
    assert state.consecutive_skips == 3


def test_numerical_miss_drops_the_cache_so_the_next_step_reseeds():
    """A non-finite mod_in computes-without-caching; it must ALSO invalidate the
    residual cached before the miss and zero the accumulator/streak, so the next
    finite step re-seeds (compute + cache) instead of skipping on a residual that
    is now >= 2 diffusion steps stale, judged against a pre-miss anchor."""
    state = _fresh_state()
    state.previous_mod_input = mx.ones((1, 16, 64))
    state.cached_residual = mx.ones((1, 16, 64))
    state.cached_residual_neg = mx.ones((1, 16, 64))
    state.accumulated_distance = 0.2
    state.consecutive_skips = 3
    kw = dict(rel_l1_thresh=10.0, coefficients=COEFFS, skip_first=1, skip_last=1, num_steps=25)

    miss = gate_step(state, step_idx=5, mod_in=mx.full((1, 16, 64), float("nan")), **kw)
    assert miss.kind == "numerical-miss"
    assert state.cached_residual is None and state.cached_residual_neg is None
    assert state.accumulated_distance == 0.0 and state.consecutive_skips == 0

    # Huge threshold: without the cache drop this step would be a "skipped" on the stale residual.
    nxt = gate_step(state, step_idx=6, mod_in=mx.ones((1, 16, 64)) * 1.01, **kw)
    assert nxt.kind == "computed"
    assert nxt.should_compute is True and nxt.should_update_cache is True


def test_trailing_forced_window_leaves_the_anchor_alone():
    """Once step_idx >= num_steps - skip_last every remaining step is forced, so
    an anchor written there could never be read (the next generation resets it).
    Skipping the write also skips its host sync on the trailing steps."""
    state = _seeded_state()
    old_anchor = state.previous_mod_input
    decision = gate_step(
        state,
        coefficients=_FLAT_COEFFS,
        rel_l1_thresh=0.5,
        skip_first=1,
        skip_last=2,
        num_steps=25,
        step_idx=23,  # inside the skip_last window
        mod_in=mx.ones((4,)) * 2.0,
    )
    assert decision.kind == "forced"
    assert state.previous_mod_input is old_anchor


# ---------------------------------------------------------------------------
# v0.10.1 — reduction dtype and non-finite prediction guard
# ---------------------------------------------------------------------------


def test_mean_abs_rel_l1_accumulates_in_float32_on_bf16_inputs():
    """bug caught: mx.mean on a bf16 array returns a bf16 scalar, ~1e-3 relative
    error that grows with element count and is not absorbed by the calibrated
    polynomials (worst case 3 % on the predicted distance at z-image's threshold)."""
    import numpy as np

    mx.random.seed(7)
    n = 1 << 22
    prev = (mx.random.normal((n,)) * 3.0 + 0.5).astype(mx.bfloat16)
    cur = (prev.astype(mx.float32) * 1.001 + mx.random.normal((n,)) * 0.01).astype(mx.bfloat16)
    mx.eval(prev, cur)
    p64 = np.asarray(prev.astype(mx.float32), dtype=np.float64)
    c64 = np.asarray(cur.astype(mx.float32), dtype=np.float64)
    ref = np.abs(c64 - p64).mean() / np.abs(p64).mean()
    got = mean_abs_rel_l1(cur, prev)
    assert abs(got - ref) / ref < 1e-4, f"rel error {abs(got - ref) / ref:.2e}"


def test_non_finite_rel_l1_from_finite_inputs_is_a_numerical_miss_not_a_skip():
    """bug caught: max(0.0, nan) == 0.0 → predicted 0 → accumulator never moves →
    perpetual skip. Reachable with finite inputs: a reduction that overflows to
    inf gives inf/inf = nan."""
    state = _fresh_state(num_steps=25)
    huge = mx.full((1024,), 3.0e38, dtype=mx.float32)
    state.previous_mod_input = huge
    state.cached_residual = mx.zeros((1,))
    d = gate_step(
        state,
        rel_l1_thresh=0.2,
        coefficients=COEFFS,
        skip_first=1,
        skip_last=1,
        num_steps=25,
        step_idx=5,
        mod_in=-huge,
    )
    assert d.kind == "numerical-miss"
    assert d.should_compute and not d.should_update_cache
    assert state.cached_residual is None
    assert state.accumulated_distance == 0.0


def test_numerical_miss_from_overflow_reports_no_rel_l1():
    """bug caught: passing the overflowed inf/nan through as `rel_l1`, where every
    other no-valid-signal path reports None (and json.dumps would emit NaN)."""
    state = _fresh_state(num_steps=25)
    huge = mx.full((1024,), 3.0e38, dtype=mx.float32)
    state.previous_mod_input = huge
    state.cached_residual = mx.zeros((1,))
    d = gate_step(
        state,
        rel_l1_thresh=0.2,
        coefficients=COEFFS,
        skip_first=1,
        skip_last=1,
        num_steps=25,
        step_idx=5,
        mod_in=-huge,
    )
    assert d.kind == "numerical-miss"
    assert d.rel_l1 is None
