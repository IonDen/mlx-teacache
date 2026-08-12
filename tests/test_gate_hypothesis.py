# tests/test_gate_hypothesis.py
import mlx.core as mx
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlx_teacache.cache import TeaCacheState
from mlx_teacache.gate import gate_step

_FINITE_FLOATS = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
_COEFFS = st.tuples(_FINITE_FLOATS, _FINITE_FLOATS, _FINITE_FLOATS, _FINITE_FLOATS, _FINITE_FLOATS)


def _fresh(num_steps: int = 25) -> TeaCacheState:
    s = TeaCacheState()
    s.reset_for_new_generation(num_steps=num_steps)
    return s


@given(coeffs=_COEFFS)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_threshold_zero_never_skips(coeffs):
    state = _fresh()
    state.previous_mod_input = mx.ones((1, 8, 16))
    mod_in = mx.ones((1, 8, 16)) * 1.5
    dec = gate_step(
        state,
        rel_l1_thresh=0.0,
        coefficients=coeffs,
        skip_first=0,
        skip_last=0,
        num_steps=25,
        step_idx=5,
        mod_in=mod_in,
    )
    assert dec.kind != "skipped"


@given(
    step_idx=st.integers(min_value=0, max_value=99),
    skip_first=st.integers(min_value=0, max_value=5),
    skip_last=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=100, deadline=None)
def test_forced_windows_honored(step_idx, skip_first, skip_last):
    num_steps = 100
    state = _fresh(num_steps=num_steps)
    # Seed the anchor so out-of-window steps reach the threshold-compare path
    # (poly(x)=1.0 >= thresh 0.5 -> "computed"), not just the seed branch.
    state.previous_mod_input = mx.ones((1, 4, 8)) * 2.0
    # v0.10.0: consecutive-delta anchoring (Option A) — the seed/re-seed guard
    # now also requires a cached residual; seed one so out-of-window steps
    # reach the threshold-compare path instead of the seed branch.
    state.cached_residual = mx.ones((1, 4, 8))
    mod_in = mx.ones((1, 4, 8))
    dec = gate_step(
        state,
        rel_l1_thresh=0.5,
        coefficients=(0.0,) * 4 + (1.0,),
        skip_first=skip_first,
        skip_last=skip_last,
        num_steps=num_steps,
        step_idx=step_idx,
        mod_in=mod_in,
    )
    in_first_window = step_idx < skip_first
    in_last_window = step_idx >= num_steps - skip_last
    if in_first_window or in_last_window:
        assert dec.kind == "forced"
    else:
        assert dec.kind != "forced"
        # The threshold path was actually taken (not the seed branch).
        assert dec.rel_l1 is not None
