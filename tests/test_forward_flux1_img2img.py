# tests/test_forward_flux1_img2img.py
"""FLUX.1 forward — img2img relative step indexing.

img2img generations start at t = init_time_step > 0 in mflux. TeaCache must
use a 0-based per-generation counter (state.step_counter) as its step index
inside gate_step, NOT the absolute scheduler timestep. These tests assert
that change.
"""

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from mlx_teacache.cache import TeaCacheState
from mlx_teacache.errors import InvalidStepWindowError
from mlx_teacache.integrations.mflux.lifecycle import GenerationContext
from mlx_teacache.stats import TeaCacheStats
from mlx_teacache.variants.flux1_dev.integration import flux1_forward_with_gate

# --- Stubs ---


@dataclass
class _FakeHandleState:
    cache: TeaCacheState = field(default_factory=TeaCacheState)
    stats: TeaCacheStats = field(default_factory=TeaCacheStats)
    no_benefit_warned: bool = False


@dataclass
class _FakeHandle:
    variant_id: str = "flux1-dev"
    skip_first_n_steps: int = 1
    skip_last_n_steps: int = 1
    rel_l1_thresh: float = 0.0  # fast-path; no need to seed cached_residual
    coefficients: tuple = (1.0, 0.0, 0.0, 0.0, 0.0)
    _gen_ctx: GenerationContext = field(default_factory=GenerationContext)
    _state: _FakeHandleState = field(default_factory=_FakeHandleState)


def _make_fake_inner():
    """Bare-minimum mflux Flux1 transformer stub. Each method is a passthrough
    or identity so flux1_forward_with_gate can run without a real model."""
    inner = MagicMock()
    inner.x_embedder = lambda x: x
    inner.context_embedder = lambda x: x
    inner.compute_text_embeddings = lambda t, pp, tte, cfg: mx.zeros((1, 1, 4))
    inner.compute_rotary_embeddings = lambda pe, posE, cfg, ki: mx.zeros((1, 1, 4))
    inner.transformer_blocks = []  # empty body — fast path doesn't traverse
    inner.single_transformer_blocks = []
    inner.norm_out = lambda x, t: x
    inner.proj_out = lambda x: x
    return inner


def _make_fake_config(num_inference_steps: int = 25):
    return MagicMock(num_inference_steps=num_inference_steps)


# --- Tests ---


def test_img2img_first_call_uses_step_counter_zero_even_when_t_positive():
    """First img2img call has absolute t = init_time_step > 0, but gate_step
    must be invoked with step_idx=0 (relative). The stats record reflects this."""
    handle = _FakeHandle()
    # Simulate lifecycle setting active_num_steps for an img2img call:
    handle._gen_ctx.active_num_steps = 8  # 25 - 17
    inner = _make_fake_inner()
    config = _make_fake_config(25)

    # First call: absolute t=17 (img2img start)
    hidden_states = mx.zeros((1, 16, 4))
    prompt_embeds = mx.zeros((1, 8, 4))
    pooled = mx.zeros((1, 4))

    flux1_forward_with_gate(
        inner,
        handle,
        t=17,
        config=config,
        hidden_states=hidden_states,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled,
    )

    # Step decision has step_idx=0 (relative), timestep=17.0 (absolute).
    assert len(handle._state.stats._staging.decisions) == 1
    decision = handle._state.stats._staging.decisions[0]
    assert decision.step_idx == 0, f"expected relative 0-based index, got {decision.step_idx}"
    assert decision.timestep == 17.0, "expected absolute scheduler timestep"


def test_forward_does_not_reset_cache_on_t_zero():
    """Cache reset is now lifecycle-owned. forward.py must NOT call
    reset_for_new_generation, even when t == 0."""
    handle = _FakeHandle()
    handle._gen_ctx.active_num_steps = 25  # txt2img
    # Seed the cache with a known step_counter > 0; if forward improperly reset,
    # this would go back to 0.
    handle._state.cache.step_counter = 5
    handle._state.cache.skip_window_validated = True  # bypass validation

    inner = _make_fake_inner()
    config = _make_fake_config(25)
    hidden_states = mx.zeros((1, 16, 4))
    prompt_embeds = mx.zeros((1, 8, 4))
    pooled = mx.zeros((1, 4))

    flux1_forward_with_gate(
        inner,
        handle,
        t=0,
        config=config,
        hidden_states=hidden_states,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled,
    )

    # step_counter advances by 1 (from 5 to 6); was NOT reset to 0.
    assert handle._state.cache.step_counter == 6, (
        f"forward.py reset the cache when it should not have "
        f"(step_counter became {handle._state.cache.step_counter})"
    )


def test_skip_window_validated_against_active_num_steps_not_nominal():
    """For img2img with active=4 and skip_first=2 + skip_last=2, the validation
    must raise (sum >= active). Under the old code (validated against nominal=25),
    the validation would silently pass even though we'd never have any eligible
    step. The new code uses the active window from _gen_ctx."""
    handle = _FakeHandle(skip_first_n_steps=2, skip_last_n_steps=2)
    handle._gen_ctx.active_num_steps = 4  # img2img with strength=0.84 + 25 steps
    handle._state.cache.step_counter = 0
    handle._state.cache.skip_window_validated = False

    inner = _make_fake_inner()
    config = _make_fake_config(25)
    hidden_states = mx.zeros((1, 16, 4))
    prompt_embeds = mx.zeros((1, 8, 4))
    pooled = mx.zeros((1, 4))

    with pytest.raises(InvalidStepWindowError) as exc:
        flux1_forward_with_gate(
            inner,
            handle,
            t=21,
            config=config,  # absolute t = 21 (img2img start)
            hidden_states=hidden_states,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled,
        )
    # The error should mention the active count, not the nominal 25.
    assert "active_num_steps=4" in str(exc.value) or "active" in str(exc.value).lower()


def test_forward_does_not_call_reset_directly():
    """Sanity duplicate of the `test_forward_does_not_reset_cache_on_t_zero`
    test above, asserting the broader property: forward.py is no longer the
    reset owner. The actual safety property — that lifecycle DOES reset stale
    state on a new generation — is asserted in
    `tests/test_lifecycle_img2img.py::test_before_loop_resets_stale_cache_for_img2img`
    (added in Task 8 alongside the lifecycle plumbing tests).
    """
    # The assertion is the same as the preceding step_counter test: forward
    # advances state but never zeroes it. Keeping it small here so the file
    # documents intent at the forward layer.
    handle = _FakeHandle()
    handle._gen_ctx.active_num_steps = 25
    handle._state.cache.step_counter = 3
    handle._state.cache.skip_window_validated = True

    inner = _make_fake_inner()
    config = _make_fake_config(25)
    hidden_states = mx.zeros((1, 16, 4))
    prompt_embeds = mx.zeros((1, 8, 4))
    pooled = mx.zeros((1, 4))

    flux1_forward_with_gate(
        inner,
        handle,
        t=3,
        config=config,
        hidden_states=hidden_states,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled,
    )
    assert handle._state.cache.step_counter == 4
