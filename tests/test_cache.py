# tests/test_cache.py
import mlx.core as mx

from mlx_teacache.cache import TeaCacheState


def test_fresh_state_fields():
    s = TeaCacheState()
    assert s.step_counter == 0
    assert s.previous_mod_input is None
    assert s.cached_residual is None
    assert s.accumulated_distance == 0.0
    assert s.last_timestep is None
    assert s.skip_window_validated is False
    assert s.num_steps is None


def test_reset_for_new_generation_clears_all():
    s = TeaCacheState()
    s.step_counter = 5
    s.previous_mod_input = mx.ones((1, 4, 8))
    s.cached_residual = mx.ones((1, 8, 8))
    s.accumulated_distance = 0.123
    s.last_timestep = 0.5
    s.skip_window_validated = True
    s.reset_for_new_generation(num_steps=10)
    assert s.step_counter == 0
    assert s.previous_mod_input is None
    assert s.cached_residual is None
    assert s.accumulated_distance == 0.0
    assert s.last_timestep is None
    assert s.skip_window_validated is False
    assert s.num_steps == 10


def test_reset_for_new_generation_clears_cached_residual_neg():
    """cached_residual_neg must be cleared alongside cached_residual when a
    generation starts. Prevents cross-generation pollution under CFG."""
    state = TeaCacheState()
    state.cached_residual = mx.zeros((1, 4))
    state.cached_residual_neg = mx.zeros((1, 4))
    state.reset_for_new_generation(num_steps=10)
    assert state.cached_residual is None
    assert state.cached_residual_neg is None
