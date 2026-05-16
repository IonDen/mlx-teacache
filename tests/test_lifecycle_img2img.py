# tests/test_lifecycle_img2img.py
"""Tests for img2img-aware lifecycle: _active_step_count helper, active step
plumbing through before/after-loop, image_strength=1.0 no-op, and the deletion
regression guard for img2img rejection.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from mlx_teacache.cache import TeaCacheState
from mlx_teacache.integrations.mflux.lifecycle import (
    GenerationContext,
    GenerationContextCallback,
    PendingFinalize,
    _active_step_count,
)
from mlx_teacache.stats import StepDecision, TeaCacheStats


def _make_config(num_inference_steps: int, *, image_strength: float | None = None) -> SimpleNamespace:
    """Faithfully stub mflux.Config: emulate the init_time_step property exactly.

    Real mflux logic (mflux/models/common/config/config.py):
        if is_img2img:
            strength = max(0.0, min(1.0, image_strength))
            return max(1, int(num_inference_steps * strength))
        return 0
    """
    if image_strength is None or image_strength <= 0.0:
        init_time_step = 0
    else:
        strength = max(0.0, min(1.0, image_strength))
        init_time_step = max(1, int(num_inference_steps * strength))
    return SimpleNamespace(
        num_inference_steps=num_inference_steps,
        init_time_step=init_time_step,
    )


@pytest.mark.parametrize(
    "image_strength,expected_init,expected_active",
    [
        (None, 0, 25),   # txt2img
        (0.0, 0, 25),    # txt2img (explicit zero strength)
        (0.04, 1, 24),   # img2img — tiny strength
        (0.5, 12, 13),   # img2img — half
        (0.7, 17, 8),    # img2img — 0.7 yields 8 calls per mflux semantics
        (1.0, 25, 0),    # img2img — full preservation, zero denoising calls
    ],
)
def test_active_step_count_matches_mflux_semantics(image_strength, expected_init, expected_active):
    cfg = _make_config(25, image_strength=image_strength)
    assert cfg.init_time_step == expected_init  # stub sanity
    assert _active_step_count(cfg) == expected_active


def test_active_step_count_missing_init_time_step_attribute_treats_as_txt2img():
    """Custom mflux subclasses might not expose init_time_step. Default to 0."""
    cfg = SimpleNamespace(num_inference_steps=25)  # no init_time_step
    assert _active_step_count(cfg) == 25


def test_active_step_count_init_time_step_is_None():
    """getattr fallback covers `init_time_step=None` too."""
    cfg = SimpleNamespace(num_inference_steps=25, init_time_step=None)
    assert _active_step_count(cfg) == 25


def test_active_step_count_floor_at_zero():
    """Defensive: if init_time_step somehow exceeds nominal, we floor at 0."""
    cfg = SimpleNamespace(num_inference_steps=10, init_time_step=15)
    assert _active_step_count(cfg) == 0


# ---- Lifecycle plumbing ----


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
    rel_l1_thresh: float = 0.20
    _gen_ctx: GenerationContext = field(default_factory=GenerationContext)
    _state: _FakeHandleState = field(default_factory=_FakeHandleState)
    _original_generate_image: Any = None
    _generate_image_was_instance_attr: bool = False
    _pending_finalize: Any = None


def test_before_loop_sets_active_num_steps_for_txt2img():
    handle = _FakeHandle()
    cb = GenerationContextCallback(handle)
    cb.call_before_loop(seed=1, prompt="hi", latents=None, config=_make_config(25))
    assert handle._gen_ctx.active_num_steps == 25


def test_before_loop_sets_active_num_steps_for_img2img_07():
    handle = _FakeHandle()
    cb = GenerationContextCallback(handle)
    cb.call_before_loop(seed=1, prompt="hi", latents=None,
                        config=_make_config(25, image_strength=0.7))
    assert handle._gen_ctx.active_num_steps == 8  # 25 - 17


def test_before_loop_zero_active_steps_for_strength_1():
    """image_strength=1.0 yields 0 active steps; lifecycle reset still runs
    cleanly with num_steps=0 (cache simply never engages)."""
    handle = _FakeHandle()
    cb = GenerationContextCallback(handle)
    cb.call_before_loop(seed=1, prompt="hi", latents=None,
                        config=_make_config(25, image_strength=1.0))
    assert handle._gen_ctx.active_num_steps == 0


def test_before_loop_does_not_raise_img2img_supported_error():
    """Regression guard: the v0.1 Img2ImgNotSupportedError no longer fires."""
    handle = _FakeHandle()
    cb = GenerationContextCallback(handle)
    # Must not raise:
    cb.call_before_loop(seed=1, prompt="hi", latents=None,
                        config=_make_config(25, image_strength=0.5))


def test_after_loop_uses_active_num_steps_for_pending_finalize():
    """call_after_loop reads active count from _gen_ctx (NOT config.num_inference_steps),
    so finalize_last_generation's len(decisions) == num_inference_steps invariant
    holds under img2img."""
    handle = _FakeHandle()
    cb = GenerationContextCallback(handle)
    cfg = _make_config(25, image_strength=0.7)  # active = 8
    cb.call_before_loop(seed=1, prompt="hi", latents=None, config=cfg)
    # Stage 8 step decisions to match the active count.
    for i in range(8):
        handle._state.stats.record(StepDecision(i, float(i), None, 0.0, "computed"))
    cb.call_after_loop(seed=1, prompt="hi", latents=None, config=cfg)
    assert handle._pending_finalize == PendingFinalize(
        num_inference_steps=8,
        cfg_was_active=False,
    )


def test_after_loop_defensive_recompute_when_gen_ctx_active_is_none():
    """If _gen_ctx.active_num_steps was never set (shouldn't happen, but
    defensive), recompute from config rather than silently skipping finalization."""
    handle = _FakeHandle()
    cb = GenerationContextCallback(handle)
    # Skip the before_loop call; jump straight to after_loop.
    cfg = _make_config(25, image_strength=0.5)  # active = 13
    cb.call_after_loop(seed=1, prompt="hi", latents=None, config=cfg)
    assert handle._pending_finalize == PendingFinalize(
        num_inference_steps=13,
        cfg_was_active=False,
    )


def test_before_loop_resets_stale_cache_for_img2img():
    """The actual safety property surfaced by the audit: lifecycle reset must
    clear stale cache state from a prior generation, including step_counter,
    cached_residual, previous_mod_input, accumulated_distance, last_timestep,
    and (re)set num_steps. The test seeds stale fields and asserts they are
    cleared by call_before_loop."""
    import mlx.core as mx

    handle = _FakeHandle()
    # Seed stale state simulating a previous generation that finished:
    cache = handle._state.cache
    cache.step_counter = 8
    cache.cached_residual = mx.zeros((1, 24, 4))
    cache.previous_mod_input = mx.zeros((1, 16, 4))
    cache.accumulated_distance = 0.42
    cache.last_timestep = 21.0
    cache.skip_window_validated = True
    cache.num_steps = 25

    cb = GenerationContextCallback(handle)
    # New img2img generation with active=8.
    cfg = _make_config(25, image_strength=0.7)
    cb.call_before_loop(seed=1, prompt="hi", latents=None, config=cfg)

    # Cache fields must be reset to a clean per-generation state with
    # num_steps == active_num_steps (8), not the stale 25 from the prior run.
    assert cache.step_counter == 0
    assert cache.cached_residual is None
    assert cache.previous_mod_input is None
    assert cache.accumulated_distance == 0.0
    assert cache.last_timestep is None
    assert cache.skip_window_validated is False
    assert cache.num_steps == 8  # lifecycle reset uses active count
