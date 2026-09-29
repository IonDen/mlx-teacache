# tests/test_cache.py
import mlx.core as mx

from mlx_teacache.cache import TeaCacheState
from mlx_teacache.gate import gate_step

_COEFFS = (0.0, 0.0, 0.0, 1.0, 0.0)  # p(x) = x


def test_fresh_state_fields():
    s = TeaCacheState()
    assert s.step_counter == 0
    assert s.previous_mod_input is None
    assert s.cached_residual is None
    assert s.cached_residual_neg is None
    assert s.accumulated_distance == 0.0
    assert s.skip_window_validated is False
    assert s.consecutive_skips == 0


def test_reset_for_new_generation_clears_all():
    s = TeaCacheState()
    s.step_counter = 5
    s.previous_mod_input = mx.ones((1, 4, 8))
    s.cached_residual = mx.ones((1, 8, 8))
    s.cached_residual_neg = mx.ones((1, 8, 8))
    s.accumulated_distance = 0.123
    s.skip_window_validated = True
    s.consecutive_skips = 3
    s.reset_for_new_generation(num_steps=10)
    assert s.step_counter == 0
    assert s.previous_mod_input is None
    assert s.cached_residual is None
    assert s.cached_residual_neg is None
    assert s.accumulated_distance == 0.0
    assert s.skip_window_validated is False
    assert s.consecutive_skips == 0


def test_reset_for_new_generation_clears_cached_residual_neg():
    """cached_residual_neg must be cleared alongside cached_residual when a
    generation starts. Prevents cross-generation pollution under CFG."""
    state = TeaCacheState()
    state.cached_residual = mx.zeros((1, 4))
    state.cached_residual_neg = mx.zeros((1, 4))
    state.reset_for_new_generation(num_steps=10)
    assert state.cached_residual is None
    assert state.cached_residual_neg is None


def test_seed_step_resets_the_skip_streak():
    """Bug caught: the seed branch of gate_step leaves a stale consecutive_skips
    from before the re-seed, so the runaway cap fires early after a re-seed."""
    state = TeaCacheState()
    state.reset_for_new_generation(num_steps=25)
    state.consecutive_skips = 5  # stale streak; no anchor and no residual yet
    decision = gate_step(
        state,
        rel_l1_thresh=1.0,
        coefficients=_COEFFS,
        skip_first=0,
        skip_last=0,
        num_steps=25,
        step_idx=3,
        mod_in=mx.ones((1, 4)),
    )
    assert decision.kind == "computed"
    assert decision.should_update_cache is True
    assert state.consecutive_skips == 0
