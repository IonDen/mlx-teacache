# tests/test_lifecycle.py
"""Lifecycle helpers — callback + generate_image wrapper. Tests use fake
flux objects to avoid pulling in mflux at unit-test time."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from mlx_teacache.cache import TeaCacheState
from mlx_teacache.integrations.mflux.lifecycle import (
    GenerationContext,
    GenerationContextCallback,
    PendingFinalize,
    wrap_generate_image,
)
from mlx_teacache.stats import StepDecision, TeaCacheStats


@dataclass
class _FakeHandle:
    variant_id: str = "flux1-dev"
    skip_first_n_steps: int = 1
    skip_last_n_steps: int = 1
    _gen_ctx: GenerationContext = field(default_factory=GenerationContext)
    _state: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(
            stats=TeaCacheStats(),
            cache=TeaCacheState(),
            no_benefit_warned=False,
        )
    )
    _original_generate_image: object = None
    _generate_image_was_instance_attr: bool = False
    _pending_finalize: PendingFinalize | None = None


# -- _GenerationContextCallback --


def _txt2img_config():
    return SimpleNamespace(image_path=None, image_strength=None, num_inference_steps=25)


def _img2img_config():
    # 0.5 * 25 = 12 init_time_step; matches mflux Config semantics.
    return SimpleNamespace(
        image_path="x.png",
        image_strength=0.5,
        num_inference_steps=25,
        init_time_step=12,
    )


def test_before_loop_captures_num_steps():
    handle = _FakeHandle()
    cb = GenerationContextCallback(handle)
    cb.call_before_loop(seed=42, prompt="hi", latents=None, config=_txt2img_config())
    assert handle._gen_ctx.active_num_steps == 25
    assert handle._gen_ctx.token == 1
    assert handle._gen_ctx.consumed_at_token is None


def test_before_loop_accepts_img2img_no_rejection():
    """img2img is supported as of v0.2 — no exception expected."""
    handle = _FakeHandle(variant_id="flux2-klein-4b")
    cb = GenerationContextCallback(handle)
    # Should not raise:
    cb.call_before_loop(seed=42, prompt="hi", latents=None, config=_img2img_config())
    # active_num_steps is set from _active_step_count: 25 - 12 = 13 for this config.
    assert handle._gen_ctx.active_num_steps == 13


def test_after_loop_marks_pending_finalize_does_not_commit():
    """call_after_loop sets _pending_finalize but does NOT commit stats —
    the generate_image wrapper commits after original() returns cleanly.
    Per audit High #2: defers so user after-loop callbacks raising won't
    leak partial committed stats."""
    handle = _FakeHandle()
    cb = GenerationContextCallback(handle)
    cb.call_before_loop(seed=42, prompt="hi", latents=None, config=_txt2img_config())
    handle._state.stats.record(StepDecision(0, 1.0, None, 0.0, "computed"))
    cb.call_after_loop(seed=42, prompt="hi", latents=None, config=_txt2img_config())
    assert handle._gen_ctx.active_num_steps is None
    assert handle._gen_ctx.consumed_at_token is None
    # Public counters NOT moved yet
    assert handle._state.stats.generations == 0
    assert handle._state.stats.computed_count == 0
    # Pending finalize metadata recorded
    assert handle._pending_finalize == PendingFinalize(
        num_inference_steps=25,
        cfg_was_active=False,
    )


def test_interrupt_does_not_finalize_stats():
    handle = _FakeHandle()
    cb = GenerationContextCallback(handle)
    cb.call_before_loop(seed=42, prompt="hi", latents=None, config=_txt2img_config())
    handle._state.stats.record(StepDecision(0, 1.0, None, 0.0, "computed"))
    cb.call_interrupt(t=5, seed=42, prompt="hi", latents=None, config=_txt2img_config(), time_steps=None)
    # call_interrupt is a no-op for stats — the generate_image wrapper handles cleanup
    assert handle._state.stats.generations == 0
    assert handle._state.stats.computed_count == 0  # staging not committed
    # _gen_ctx is NOT cleared by call_interrupt either; the wrapper clears it
    assert handle._gen_ctx.active_num_steps == 25


def test_before_loop_accepts_extra_kwargs():
    handle = _FakeHandle()
    cb = GenerationContextCallback(handle)
    # Real mflux passes canny_image= and depth_image= via kwargs
    cb.call_before_loop(
        seed=42, prompt="hi", latents=None, config=_txt2img_config(), canny_image=None, depth_image=None
    )
    assert handle._gen_ctx.active_num_steps == 25


# -- wrap_generate_image --


def test_wrap_finalizes_stats_only_when_call_after_loop_ran():
    """Wrapper finalizes only when call_after_loop set _pending_finalize AND
    original() returned naturally."""
    handle = _FakeHandle()
    handle._pending_finalize = None

    flux = SimpleNamespace()

    def original():
        # Simulate the after-loop callback running cleanly inside the loop
        handle._state.stats.record(StepDecision(0, 1.0, None, 0.0, "computed"))
        handle._pending_finalize = PendingFinalize(num_inference_steps=1, cfg_was_active=False)
        return "ok"

    flux.generate_image = original
    wrap_generate_image(flux, handle)
    result = flux.generate_image()
    assert result == "ok"
    # Finalize ran, public counters moved
    assert handle._state.stats.generations == 1
    assert handle._state.stats.computed_count == 1
    assert handle._generate_image_was_instance_attr is True


def test_wrap_after_loop_user_callback_raises_no_partial_commit():
    """Per audit High #2: if call_after_loop set _pending_finalize but then a
    LATER after-loop callback raises (so original() raises), the wrapper
    must discard staging rather than commit."""
    handle = _FakeHandle()
    handle._pending_finalize = None

    flux = SimpleNamespace()

    def original():
        # mlx-teacache call_after_loop ran and set _pending_finalize.
        handle._state.stats.record(StepDecision(0, 1.0, None, 0.0, "computed"))
        handle._pending_finalize = PendingFinalize(num_inference_steps=1, cfg_was_active=False)
        # Now a later user after-loop callback raises.
        raise RuntimeError("user after-loop bug")

    flux.generate_image = original
    wrap_generate_image(flux, handle)
    with pytest.raises(RuntimeError, match="user after-loop bug"):
        flux.generate_image()
    # Public counters MUST NOT have moved.
    assert handle._state.stats.generations == 0
    assert handle._state.stats.computed_count == 0


def test_wrap_clears_context_on_exception_and_discards_staging():
    handle = _FakeHandle()
    handle._gen_ctx.active_num_steps = 25
    handle._state.stats.record(StepDecision(0, 1.0, None, 0.0, "computed"))

    flux = SimpleNamespace()

    def boom():
        raise RuntimeError("kaboom")

    flux.generate_image = boom
    wrap_generate_image(flux, handle)

    with pytest.raises(RuntimeError, match="kaboom"):
        flux.generate_image()
    # context cleared
    assert handle._gen_ctx.active_num_steps is None
    # staging discarded => public counters didn't move
    assert handle._state.stats.computed_count == 0
    assert handle._state.stats.generations == 0


def test_wrap_records_no_prior_instance_attr_when_class_level():
    """If generate_image came from the class, _generate_image_was_instance_attr is False."""

    class FluxClass:
        def generate_image(self):
            return "class-method"

    handle = _FakeHandle()
    flux = FluxClass()
    wrap_generate_image(flux, handle)
    assert handle._generate_image_was_instance_attr is False


def test_wrap_callback_replacement_raises_missing_context():
    """Per audit medium #4: if flux.callbacks is replaced after apply_teacache,
    the wrapped generate_image must raise MissingGenerationContextError before
    running, rather than silently completing with no lifecycle hook."""
    from mlx_teacache.errors import MissingGenerationContextError

    handle = _FakeHandle()
    # Pretend a callback was registered on the original registry.
    cb_sentinel = object()
    handle._callback_instance = cb_sentinel  # type: ignore[attr-defined]

    class _RegistryWithCb:
        before_loop_callbacks = [cb_sentinel]
        in_loop_callbacks: list = []
        after_loop_callbacks: list = []
        interrupt_callbacks: list = []

    class _ReplacedRegistry:
        before_loop_callbacks: list = []
        in_loop_callbacks: list = []
        after_loop_callbacks: list = []
        interrupt_callbacks: list = []

    flux = SimpleNamespace(callbacks=_RegistryWithCb())
    flux.generate_image = lambda: "ok"
    wrap_generate_image(flux, handle)

    # First call: callback still present -> succeeds
    assert flux.generate_image() == "ok"

    # Now replace callbacks (user does this)
    flux.callbacks = _ReplacedRegistry()
    with pytest.raises(MissingGenerationContextError):
        flux.generate_image()


def test_after_loop_releases_cached_arrays():
    """bug caught: call_after_loop leaving body-sized residuals resident through
    VAE decode (where peak memory lands) and for the process lifetime."""
    import mlx.core as mx

    handle = _FakeHandle()
    cb = GenerationContextCallback(handle)
    cb.call_before_loop(seed=42, prompt="hi", latents=None, config=_txt2img_config())
    handle._state.cache.previous_mod_input = mx.zeros((2,))
    handle._state.cache.cached_residual = mx.zeros((2,))
    handle._state.cache.cached_residual_neg = mx.zeros((2,))
    cb.call_after_loop(seed=42, prompt="hi", latents=None, config=_txt2img_config())
    c = handle._state.cache
    assert c.previous_mod_input is None
    assert c.cached_residual is None
    assert c.cached_residual_neg is None


def test_wrapper_releases_cached_arrays_when_generation_raises():
    """bug caught: an interrupted or failed generation keeping its body-sized
    arrays resident until the next generation's before_loop."""
    import mlx.core as mx

    handle = _FakeHandle()
    cache = handle._state.cache

    def _boom(**kw):
        cache.cached_residual = mx.zeros((2,))
        cache.previous_mod_input = mx.zeros((2,))
        raise KeyboardInterrupt

    flux = SimpleNamespace(callbacks=None, generate_image=_boom)
    wrap_generate_image(flux, handle)
    with pytest.raises(KeyboardInterrupt):
        flux.generate_image(prompt="x")
    assert cache.cached_residual is None
    assert cache.previous_mod_input is None
