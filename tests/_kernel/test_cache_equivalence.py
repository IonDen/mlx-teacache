"""Phase A discipline: _kernel.cache.TeaCacheState is the same dataclass
as v0.5.x src/mlx_teacache/cache.TeaCacheState (post-shim)."""

from __future__ import annotations

import dataclasses

import mlx.core as mx


def test_field_set_matches_v05():
    from mlx_teacache._kernel.cache import TeaCacheState

    expected_field_names = {
        "step_counter",
        "previous_mod_input",
        "cached_residual",
        "cached_residual_neg",
        "accumulated_distance",
        "last_timestep",
        "skip_window_validated",
        "num_steps",
        # v0.10.0: consecutive-delta anchoring (Option A) — runaway-skip streak counter.
        "consecutive_skips",
    }
    actual = {f.name for f in dataclasses.fields(TeaCacheState)}
    assert actual == expected_field_names


def test_reset_signature_takes_num_steps():
    from mlx_teacache._kernel.cache import TeaCacheState

    s = TeaCacheState()
    s.step_counter = 5
    s.accumulated_distance = 1.0
    s.cached_residual = mx.array([1.0])
    s.cached_residual_neg = mx.array([2.0])
    s.skip_window_validated = True
    s.num_steps = 8
    s.last_timestep = 0.5
    s.previous_mod_input = mx.array([0.1])

    s.reset_for_new_generation(num_steps=12)

    assert s.step_counter == 0
    assert s.previous_mod_input is None
    assert s.cached_residual is None
    assert s.cached_residual_neg is None
    assert s.accumulated_distance == 0.0
    assert s.last_timestep is None
    assert s.skip_window_validated is False
    assert s.num_steps == 12


def test_shim_re_exports_state_identity():
    """Old import path still works."""
    from mlx_teacache._kernel.cache import TeaCacheState as KernelState
    from mlx_teacache.cache import TeaCacheState as LegacyState

    assert LegacyState is KernelState
