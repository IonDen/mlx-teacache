"""The one runner-neutral session type every integration shares. Pure-core: no mflux, no weights."""

import mlx.core as mx

from mlx_teacache.integrations import _session
from mlx_teacache.testing import fake_handle

_COEFFS = (1.0, 0.0, 0.0, 0.0, 0.0)


def _session_handle() -> _session.InternalHandle:
    return _session.InternalHandle(
        rel_l1_thresh=0.2,
        coefficients=_COEFFS,
        skip_first_n_steps=1,
        skip_last_n_steps=1,
    )


def test_session_defaults_match_the_three_old_copies():
    """bug caught: a default drifts while merging the copies (Qwen's pairer missing on FLUX handles, the token
    starting at 1, a pending finalize left set), or two sessions share one mutable cache."""
    session = _session_handle()
    assert session._gen_ctx.token == 0
    assert session._gen_ctx.active_num_steps is None
    assert session._gen_ctx.consumed_at_token is None
    assert session._state.no_benefit_warned is False
    assert session._pending_finalize is None
    assert session._callback_instance is None
    assert session._generate_image_was_instance_attr is False
    assert session._original_generate_image is None
    assert session._pairer.branch_idx == 0
    assert session._pairer.last_seen_token is None
    assert session._pairer.shared_decision is None
    assert session.calibrated_range is None
    assert session._state.cache.step_counter == 0
    assert session._state.stats.generations == 0

    other = _session_handle()
    assert other._state.cache is not session._state.cache
    assert other._state.stats is not session._state.stats
    assert other._gen_ctx is not session._gen_ctx
    assert other._pairer is not session._pairer


def test_release_arrays_drops_the_three_arrays_and_keeps_counters():
    """bug caught: release misses one of the three body-sized arrays (previous_mod_input, say), or resets the
    counters a stats reader still needs."""
    session = _session_handle()
    cache = session._state.cache
    cache.previous_mod_input = mx.zeros((4,))
    cache.cached_residual = mx.zeros((4,))
    cache.cached_residual_neg = mx.zeros((4,))
    cache.step_counter = 3
    cache.accumulated_distance = 0.5
    cache.consecutive_skips = 2
    cache.num_steps = 10

    session.release_arrays()

    assert cache.previous_mod_input is None
    assert cache.cached_residual is None
    assert cache.cached_residual_neg is None
    assert cache.step_counter == 3
    assert cache.accumulated_distance == 0.5
    assert cache.consecutive_skips == 2
    assert cache.num_steps == 10


def test_fake_handle_has_no_session_and_restores():
    """bug caught: the new private attribute breaks the public fake (missing, or restore() trips over it)."""
    handle = fake_handle()
    assert handle._session is None
    assert handle._callback_instance is None
    handle.restore()
    handle.restore()
    assert handle.stats._frozen is True
