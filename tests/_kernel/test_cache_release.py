"""The residual must be materialised at store time and released at the two
lifecycle exits. The first test measures the real MLX allocator with the cache
pool disabled, so a lazily-held subtraction (which pins both operands) reads
as roughly twice the residual's own size."""

import gc

import mlx.core as mx

from mlx_teacache._kernel.cache import TeaCacheState

_N = 4 * 1024 * 1024  # 16 MiB of float32


def test_store_residuals_materialises_the_subtraction_so_operands_can_be_freed() -> None:
    """bug caught: plain assignment leaves `b - a` pending, pinning both operands (2x)."""
    prev_limit = mx.set_cache_limit(0)
    try:
        gc.collect()
        before = mx.get_active_memory()
        a = mx.random.normal((_N,))
        b = a * 2.0
        mx.eval(a, b)
        state = TeaCacheState()
        state.store_residuals(pos=b - a)
        del a, b
        gc.collect()
        held = mx.get_active_memory() - before
        assert held < 1.5 * _N * 4, f"{held / 2**20:.1f} MiB held; a materialised residual is ~16 MiB"
        assert state.cached_residual is not None
    finally:
        mx.set_cache_limit(prev_limit)


def test_store_residuals_neg_only_leaves_pos_alone() -> None:
    state = TeaCacheState()
    state.cached_residual = mx.zeros((2,))
    state.store_residuals(neg=mx.ones((2,)))
    assert state.cached_residual is not None
    assert state.cached_residual.shape == (2,)
    assert state.cached_residual_neg is not None


def test_release_arrays_drops_the_three_arrays_and_nothing_else() -> None:
    # bug caught: forgetting previous_mod_input, or zeroing the counters
    state = TeaCacheState()
    state.reset_for_new_generation(num_steps=10)
    state.previous_mod_input = mx.zeros((2,))
    state.store_residuals(pos=mx.zeros((2,)), neg=mx.zeros((2,)))
    state.step_counter = 7
    state.release_arrays()
    assert state.previous_mod_input is None
    assert state.cached_residual is None
    assert state.cached_residual_neg is None
    assert state.step_counter == 7
    assert state.num_steps == 10
