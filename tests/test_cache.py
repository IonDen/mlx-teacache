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


def test_reset_is_bit_equal_to_constructor_for_array_fields():
    s = TeaCacheState()
    s.previous_mod_input = mx.ones((1, 4))
    s.reset_for_new_generation(num_steps=25)
    fresh = TeaCacheState()
    fresh.reset_for_new_generation(num_steps=25)
    assert s.previous_mod_input is fresh.previous_mod_input  # both None
    assert s.cached_residual is fresh.cached_residual  # both None
