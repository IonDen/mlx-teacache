"""mflux 0.21 deletes the per-generation predict closure before the after-loop callbacks so that --low-ram can free
the transformer before decoding (Z-Image and FLUX.2 Klein). With TeaCache applied, nothing else may keep the
transformer reachable once that closure is gone. The proxy variants (FLUX.1, Qwen-Image) have the same duty when
mflux's MemorySaver sets `model.transformer = None`. No weights: duck-typed models and the tiny fake transformers of the
forward tests. The FLUX.2 cases call a forward that imports mflux inside the function, so the file is in the
mflux lane.

The same fakes also pin how mflux calls the installed `_predict` factory: 0.24 as `self._predict(self.transformer)`,
main (6aebc2d, Z-Image) with `compile_predict=` on every generation."""

import gc
import importlib
import inspect
import pickle
import warnings
import weakref
from collections.abc import Callable
from importlib.metadata import version
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest

from mlx_teacache import apply_teacache
from mlx_teacache.errors import IncompatibleModelError
from mlx_teacache.integrations._session import GenerationContext
from tests._mflux_surface import installed_predict_factory, unbound_predict_keywords
from tests._variant_fakes import fake_for_variant
from tests.conftest import expect_distilled_warning
from tests.test_forward_flux2 import _FakeFlux2Inner
from tests.test_forward_z_image_fake import _CAP_SEQ, _DIM, _X_SEQ, _FakeZImageTransformer

# One active generation of 4 steps: the default window (1 + 1) is valid and no no-benefit warning fires.
_CONFIG = SimpleNamespace(num_inference_steps=4, init_time_step=0)


class _Transformer:
    """Stand-in transformer for the lifetime tests; only its being weak-referenceable and collectable matters."""


def _z_image_step(predict: Any) -> Any:
    # ZImage._predict closure signature: (latents, timestep, sigmas, text_encodings, negative_encodings, guidance)
    return predict(
        mx.zeros((_X_SEQ, _DIM)),
        mx.array([0.5]),
        mx.array([1.0, 0.5, 0.0]),
        mx.zeros((_CAP_SEQ, _DIM)),
        None,
        0.0,
    )


def _flux2_step(predict: Any) -> Any:
    # Flux2Klein._predict closure signature: (latents, latent_ids, prompt_embeds, text_ids,
    # negative_prompt_embeds, negative_text_ids, guidance, timestep)
    return predict(
        mx.zeros((1, 4, 4)), mx.zeros((4, 3)), mx.zeros((1, 2, 4)), mx.zeros((2, 3)), None, None, 1.0,
        mx.array([1000.0]),
    )  # fmt: skip


_CASES = {
    "z-image-base": (_FakeZImageTransformer, _z_image_step),
    "flux2-klein-base-4b": (_FakeFlux2Inner, _flux2_step),
    "flux2-klein-4b": (_FakeFlux2Inner, _flux2_step),
}


@pytest.mark.parametrize("variant_id", sorted(_CASES))
def test_with_teacache_applied_dropping_the_closure_frees_the_transformer(variant_id: str) -> None:
    """Bug: something the patch creates (the internal handle, the lifecycle callback, the rollback, the predict
    factory) keeps a reference to the transformer, so mflux 0.21's `del predict` frees nothing and --low-ram decodes
    with the transformer still resident while TeaCache is applied."""
    transformer_class, step = _CASES[variant_id]
    fake = fake_for_variant(variant_id)
    fake.transformer = transformer_class()
    ref = weakref.ref(fake.transformer)
    with expect_distilled_warning(variant_id):
        handle = apply_teacache(fake)
    for callback in fake.callbacks.before_loop:
        callback.call_before_loop(seed=0, prompt="", latents=None, config=_CONFIG)
    predict = fake._predict(fake.transformer)  # what mflux's generate_image does at the top of the loop
    mx.eval(step(predict))  # one real forced step through the gated forward
    del predict  # mflux 0.21: `del predict` right before ctx.after_loop(latents)
    fake.transformer = None  # MemorySaver with --low-ram: the pipeline drops its transformer
    gc.collect()
    assert ref() is None
    assert handle.variant_id == variant_id  # the handle is still alive and patched


_PROXY_VARIANTS = ["flux1-dev", "flux1-krea-dev", "flux1-schnell", "qwen-image"]


def _apply_proxy_variant(variant_id: str) -> tuple[Any, Any, Any, Any]:
    """(fake, handle, weakref to the original transformer, original generate_image) with TeaCache applied."""
    fake = fake_for_variant(variant_id)
    fake.transformer = _Transformer()
    ref = weakref.ref(fake.transformer)
    original_generate = fake.generate_image
    with warnings.catch_warnings():
        # Qwen-Image's alias warns about an uncalibrated checkpoint on mflux >= 0.19; irrelevant here.
        warnings.simplefilter("ignore")
        handle = apply_teacache(fake)
    return fake, handle, ref, original_generate


@pytest.mark.parametrize("variant_id", _PROXY_VARIANTS)
def test_proxy_variants_free_the_transformer_once_the_pipeline_drops_it(variant_id: str) -> None:
    """Bug: the proxy variants' rollback holds the original transformer so restore() can put it back, so mflux's
    MemorySaver setting `model.transformer = None` (--low-ram, --pid-decode) frees nothing while the handle lives and
    the weights stay resident through the VAE decode."""
    fake, handle, ref, _ = _apply_proxy_variant(variant_id)
    fake.transformer = None
    gc.collect()
    assert handle.variant_id == variant_id
    assert ref() is None


@pytest.mark.parametrize("variant_id", _PROXY_VARIANTS)
def test_restore_after_the_pipeline_dropped_the_transformer_undoes_the_rest(variant_id: str) -> None:
    """Bug: restore() after mflux dropped the transformer either raises (the command's teardown crashes after a
    finished image) or puts back a transformer that mflux meant to free. It must leave the slot empty and still undo
    the generate_image wrapper, the callback and the double-apply sentinel."""
    fake, handle, _, original_generate = _apply_proxy_variant(variant_id)
    # The patch is really installed, so the assertions after restore() are not vacuous.
    assert fake.generate_image is not original_generate
    assert fake._teacache_handle is handle
    fake.transformer = None
    gc.collect()

    handle.restore()

    assert fake.transformer is None
    assert fake.generate_image is original_generate
    assert fake.callbacks.before_loop == [] and fake.callbacks.after_loop == []
    assert not hasattr(fake, "_teacache_handle")


@pytest.mark.parametrize("variant_id", _PROXY_VARIANTS)
def test_restore_leaves_the_slot_empty_while_something_else_still_holds_the_proxy(variant_id: str) -> None:
    """Bug: restore() decides on whether the proxy is still alive instead of whether the pipeline still holds it, so a
    stray reference (a notebook variable, a REPL's `_`) makes restore() put back the transformer mflux dropped."""
    fake, handle, _, _ = _apply_proxy_variant(variant_id)
    held = fake.transformer
    fake.transformer = None

    handle.restore()

    assert fake.transformer is None
    del held


@pytest.mark.parametrize("variant_id", _PROXY_VARIANTS)
def test_restore_keeps_a_transformer_the_caller_installed_after_apply(variant_id: str) -> None:
    """Bug: restore() overwrites a transformer the caller swapped in after apply_teacache whenever the proxy is still
    referenced somewhere, so the outcome depends on reference counts instead of on the model's state."""
    fake, handle, _, _ = _apply_proxy_variant(variant_id)
    held = fake.transformer
    replacement = _Transformer()
    fake.transformer = replacement

    handle.restore()

    assert fake.transformer is replacement
    del held


def test_flux1_mfluxs_memory_saver_frees_the_transformer_during_a_teacache_generation() -> None:
    """Bug: TeaCache keeps the FLUX.1 transformer reachable through a real generation, so mflux's own MemorySaver
    (keep_transformer=False, what --low-ram registers for one seed) frees nothing before the decode. Runs the fake's
    denoising loop, so forward steps go through the proxy before the eviction."""
    from mflux.callbacks.instances.memory_saver import MemorySaver

    if "num_seeds" not in inspect.signature(MemorySaver.__init__).parameters:
        # mflux 0.17.5: no num_seeds, and the constructor always sets the process-wide MLX cache limit.
        pytest.skip("this mflux's MemorySaver predates num_seeds and an optional cache limit")
    fake = fake_for_variant("flux1-dev")
    ref = weakref.ref(fake.transformer)
    handle = apply_teacache(fake)
    fake.callbacks.register(
        MemorySaver(model=fake, keep_transformer=False, cache_limit_bytes=None, num_seeds=1)
    )

    assert fake.generate_image(seed=0, prompt="", num_inference_steps=4) == "image"

    assert fake.transformer is None
    assert ref() is None
    assert (
        len(handle.stats.last_generation.decisions) == 4
    )  # the generation completed and committed its stats
    handle.restore()
    assert fake.transformer is None


def test_qwen_image_frees_the_transformer_after_gated_forward_steps() -> None:
    """Bug: Qwen-Image's gated forward or its CFG pairer stores the real transformer on the handle or on itself while
    stepping, so it stays reachable after mflux drops the proxy. Two positive/negative step pairs through a tiny real
    mflux QwenTransformer, then the eviction."""
    from tests.test_qwen_forward_parity import _config, _inputs, _tiny_transformer

    fake = fake_for_variant("qwen-image")
    fake.transformer = _tiny_transformer("bf16")
    ref = weakref.ref(fake.transformer)
    handle = apply_teacache(fake)
    config = _config()
    config.init_time_step = 0
    for callback in fake.callbacks.before_loop:
        callback.call_before_loop(seed=0, prompt="", latents=None, config=config)
    latents, positive, negative, mask = _inputs()
    for t in range(2):
        for prompt in (positive, negative):
            mx.eval(
                fake.transformer(
                    t=t, config=config, hidden_states=latents, encoder_hidden_states=prompt,
                    encoder_hidden_states_mask=mask,
                )
            )  # fmt: skip
    for callback in list(fake.callbacks.after_loop):
        callback.call_after_loop(seed=0, prompt="", latents=latents, config=config)

    fake.transformer = None
    gc.collect()

    assert ref() is None
    assert handle.variant_id == "qwen-image"


@pytest.mark.parametrize(
    "module",
    [
        "mlx_teacache.variants.z_image_base.integration",
        "mlx_teacache.integrations.mflux.forward_flux2",
    ],
)
def test_the_predict_factory_keeps_no_reference_to_the_transformer(module: str) -> None:
    """Bug: the predict factory itself stores the transformer (or the closure) on the handle or on itself; this unit
    test names the factory when the apply-level test above goes red."""
    # The factory records on the session's generation context that mflux asked for this generation's closure.
    factory = importlib.import_module(module).make_teacache_predict_factory(
        SimpleNamespace(_gen_ctx=GenerationContext())
    )
    transformer = _Transformer()
    ref = weakref.ref(transformer)
    predict = factory(transformer)
    assert callable(predict)
    del transformer, predict
    gc.collect()
    assert ref() is None


# --- how mflux calls the installed _predict factory: mflux 0.24 `self._predict(self.transformer)`, mflux main
# 6aebc2d (Z-Image) adds `compile_predict=not self.ssd_stream`. The fake pipelines run the real apply_teacache path and
# the real gated forwards; the contract helper binds a call in generate_image source against the installed factory.

_STEPS = 4
_Z_IMAGE = "z-image-base"
_KLEIN = "flux2-klein-base-4b"


def _z_image_keyword_step(predict: Any) -> Any:
    # How ZImage.generate_image calls the closure (mflux 0.24 and main 6aebc2d).
    return predict(
        latents=mx.zeros((_X_SEQ, _DIM)),
        timestep=mx.array([0.5]),
        sigmas=mx.array([1.0, 0.5, 0.0]),
        text_encodings=mx.zeros((_CAP_SEQ, _DIM)),
        negative_encodings=None,
        guidance=0.0,
    )


def _klein_step(predict: Any) -> Any:
    # How Flux2Klein.generate_image calls the closure (mflux 0.24).
    return predict(
        latents=mx.zeros((1, 4, 4)),
        latent_ids=mx.zeros((4, 3)),
        prompt_embeds=mx.zeros((1, 2, 4)),
        text_ids=mx.zeros((2, 3)),
        negative_prompt_embeds=None,
        negative_text_ids=None,
        guidance=1.0,
        timestep=mx.array([1000.0]),
    )


_TRANSFORMERS: dict[str, tuple[Callable[[], Any], Callable[[Any], Any]]] = {
    _Z_IMAGE: (_FakeZImageTransformer, _z_image_keyword_step),
    _KLEIN: (_FakeFlux2Inner, _klein_step),
}

# The ways a pipeline asks the factory for its closure: mflux 0.24, mflux main with and without --ssd-stream.
_CALLS: dict[str, dict[str, Any]] = {
    "0.24": {},
    "main": {"compile_predict": True},
    "main-ssd-stream": {"compile_predict": False},
}


class _NoRealModel:
    """Stands in for the class-level `ZImage._predict` / `Flux2Klein._predict`: reaching it means the patch was
    not installed and the pipeline would build mflux's own closure."""

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the unpatched _predict ran; the fake must never build a real model")


def _fake_pipeline(variant_id: str, predict_kwargs: dict[str, Any]) -> Any:
    """A duck-typed pipeline whose generate_image runs the mflux loop order: before-loop callbacks, then
    `predict = self._predict(self.transformer, **predict_kwargs)`, per step one predict call followed by the
    in-loop callbacks and then the step's evaluation, `del predict`, then the after-loop callbacks.

    Two switches, read on every call: `bypass_predict` builds the step function without `self._predict` (a mflux
    release that stopped calling it), counting its calls in `unpatched_steps`; `value_error_at_step` raises
    ValueError from inside the loop at that step, before the step's in-loop callbacks."""
    make_transformer, step = _TRANSFORMERS[variant_id]
    fake = fake_for_variant(variant_id)
    fake.transformer = make_transformer()
    fake._predict = _NoRealModel()
    fake.bypass_predict = False
    fake.value_error_at_step = None
    fake.unpatched_steps = 0

    def unpatched_predict(**_kw: Any) -> Any:
        fake.unpatched_steps += 1  # stands in for one call of mflux's own transformer
        return mx.zeros((1,))

    def generate_image(seed: int = 0, prompt: str = "", num_inference_steps: int = _STEPS, **_kw: Any) -> str:
        config = SimpleNamespace(num_inference_steps=num_inference_steps, init_time_step=0)
        latents = mx.zeros((1, 4, 4))
        for cb in fake.callbacks.before_loop:
            cb.call_before_loop(seed=seed, prompt=prompt, latents=latents, config=config)
        predict = (
            unpatched_predict if fake.bypass_predict else fake._predict(fake.transformer, **predict_kwargs)
        )
        for t in range(num_inference_steps):
            if t == fake.value_error_at_step:
                raise ValueError(f"the model failed at step {t}")
            noise = step(predict)
            for cb in fake.callbacks.in_loop:
                cb.call_in_loop(
                    t=t, seed=seed, prompt=prompt, latents=latents, config=config, time_steps=None
                )
            mx.eval(noise)  # mflux evaluates the step after its in-loop callbacks
        del predict
        for cb in fake.callbacks.after_loop:
            cb.call_after_loop(seed=seed, prompt=prompt, latents=latents, config=config)
        return "image"

    fake.generate_image = generate_image
    return fake


# --- the bug as a user meets it: mflux main's call through the real patch --------------------------------


@pytest.mark.parametrize("call", sorted(_CALLS))
@pytest.mark.parametrize("variant_id", sorted(_TRANSFORMERS))
def test_a_patched_pipeline_generates_with_the_gate_however_mflux_calls_predict(
    variant_id: str, call: str
) -> None:
    """Bug: the installed factory takes only `transformer`, so mflux main's
    `self._predict(self.transformer, compile_predict=...)` raises TypeError and no image is generated."""
    fake = _fake_pipeline(variant_id, _CALLS[call])
    handle = apply_teacache(fake)
    try:
        assert fake.generate_image(num_inference_steps=_STEPS) == "image"
        assert handle.stats.generations == 1
        last = handle.stats.last_generation
        assert last is not None and len(last.decisions) == _STEPS  # the gate decided every step
    finally:
        handle.restore()


@pytest.mark.parametrize("variant_id", sorted(_TRANSFORMERS))
def test_an_unknown_predict_keyword_raises_a_package_error_naming_it_and_the_mflux_version(
    variant_id: str,
) -> None:
    """Bug: a keyword the patch does not know is swallowed (TeaCache runs on a call it was never reviewed
    against) or surfaces as a bare TypeError that names neither the keyword nor the mflux version."""
    fake = _fake_pipeline(variant_id, {"compile_predict": True, "stream_layers": True})
    handle = apply_teacache(fake)
    try:
        with pytest.raises(IncompatibleModelError) as excinfo:
            fake.generate_image(num_inference_steps=_STEPS)
        message = str(excinfo.value)
        assert "stream_layers" in message
        assert "compile_predict" not in message.split("accepts")[0]  # the known keyword is not reported
        assert f"mflux {version('mflux')}" in message
    finally:
        handle.restore()


@pytest.mark.parametrize("variant_id", sorted(_TRANSFORMERS))
def test_after_a_refused_predict_call_restore_works_and_no_generation_is_in_flight(variant_id: str) -> None:
    """Bug: the refused call leaves the step count set, so restore() refuses as generation-in-flight and the
    pipeline stays patched."""
    fake = _fake_pipeline(variant_id, {"stream_layers": True})
    original_generate = fake.generate_image
    handle = apply_teacache(fake)
    with pytest.raises(IncompatibleModelError):
        fake.generate_image(num_inference_steps=_STEPS)
    assert handle._session._gen_ctx.active_num_steps is None
    handle.restore()
    assert isinstance(vars(fake)["_predict"], _NoRealModel)
    assert fake.generate_image is original_generate
    assert not hasattr(fake, "_teacache_handle")


@pytest.mark.parametrize("variant_id", sorted(_TRANSFORMERS))
def test_the_unknown_keyword_error_survives_a_pickle_round_trip(variant_id: str) -> None:
    """Bug: the private subclass inherits IncompatibleModelError.__reduce__, which drops its own fields, so
    a pickled error (multiprocessing, a worker pool) comes back as a different message or fails to load."""
    fake = _fake_pipeline(variant_id, {"stream_layers": True, "another": 1})
    handle = apply_teacache(fake)
    try:
        with pytest.raises(IncompatibleModelError) as excinfo:
            fake.generate_image(num_inference_steps=_STEPS)
    finally:
        handle.restore()
    error = excinfo.value
    error.add_note("seen in a worker")
    loaded = pickle.loads(pickle.dumps(error))
    assert type(loaded) is type(error)
    assert str(loaded) == str(error)
    assert "another" in str(loaded) and "stream_layers" in str(loaded)
    assert loaded.__notes__ == ["seen in a worker"]
    assert loaded.unknown == ["another", "stream_layers"]
    assert loaded.mflux_version == version("mflux")


# --- the guard: a generation that never asked the patched _predict for its closure ------------------------
# mflux runs the before-loop callbacks before `self._predict(...)` and the in-loop callbacks after each step's
# predict call, so an unpatched closure is first seen at the in-loop callback of step 1: it ran exactly once.


@pytest.mark.parametrize("variant_id", sorted(_TRANSFORMERS))
def test_a_generation_that_bypasses_the_patched_predict_fails_at_the_first_step(variant_id: str) -> None:
    """Bug: mflux stops calling `self._predict` and TeaCache silently finishes an image it never gated."""
    fake = _fake_pipeline(variant_id, {})
    handle = apply_teacache(fake)
    fake.bypass_predict = True
    try:
        with pytest.raises(IncompatibleModelError) as excinfo:
            fake.generate_image(num_inference_steps=_STEPS)
        assert fake.unpatched_steps == 1  # step 1 ran; step 2 never did
        message = str(excinfo.value)
        assert f"{type(fake).__name__}.generate_image" in message
        assert "self._predict" in message
        assert handle.stats.generations == 0
    finally:
        fake.bypass_predict = False
        handle.restore()


@pytest.mark.parametrize("variant_id", sorted(_TRANSFORMERS))
def test_two_generations_through_the_patched_predict_both_finish(variant_id: str) -> None:
    """Bug: the record that the factory ran is consumed by the first generation and never set again, so the
    second generation on the same handle is refused."""
    fake = _fake_pipeline(variant_id, {"compile_predict": True})
    handle = apply_teacache(fake)
    try:
        assert fake.generate_image(num_inference_steps=_STEPS) == "image"
        assert fake.generate_image(num_inference_steps=_STEPS) == "image"
        assert handle.stats.generations == 2
    finally:
        handle.restore()


@pytest.mark.parametrize("variant_id", sorted(_TRANSFORMERS))
def test_a_bypassing_generation_after_a_patched_one_still_fails(variant_id: str) -> None:
    """Bug: the record that the factory ran is never reset, so after one patched generation every later one
    passes the guard even when it never builds the patched closure."""
    fake = _fake_pipeline(variant_id, {})
    handle = apply_teacache(fake)
    try:
        assert fake.generate_image(num_inference_steps=_STEPS) == "image"
        fake.bypass_predict = True
        with pytest.raises(IncompatibleModelError):
            fake.generate_image(num_inference_steps=_STEPS)
        assert fake.unpatched_steps == 1
    finally:
        fake.bypass_predict = False
        handle.restore()


class _ApplyAtFirstStep:
    """An in-loop callback that calls apply_teacache on the running pipeline at step 0, the way an app toggle or a
    user callback can while a generation is already in its loop."""

    def __init__(self, fake: Any) -> None:
        self._fake = fake
        self.handle: Any = None

    def call_in_loop(self, t: int, **_kw: Any) -> None:
        if self.handle is None:
            self.handle = apply_teacache(self._fake)


@pytest.mark.parametrize("variant_id", sorted(_TRANSFORMERS))
def test_applying_teacache_during_a_running_generation_lets_it_finish_plain(variant_id: str) -> None:
    """Bug: the guard judges a generation whose closure was built before apply_teacache ran (its before-loop
    callbacks never saw TeaCache), so applying TeaCache mid-run stops that run instead of letting it finish plain
    and taking effect on the next one."""
    fake = _fake_pipeline(variant_id, {})
    fake.bypass_predict = True  # the closure mflux built before the patch existed
    applier = _ApplyAtFirstStep(fake)
    fake.callbacks.register(applier)
    running = fake.generate_image  # the unwrapped call already in progress

    assert running(num_inference_steps=_STEPS) == "image"

    assert fake.unpatched_steps == _STEPS  # every step of the running call ran unpatched
    handle = applier.handle
    assert handle is not None
    try:
        fake.bypass_predict = False
        # The next call goes through TeaCache.
        assert fake.generate_image(num_inference_steps=_STEPS) == "image"
        assert handle.stats.generations == 1
    finally:
        handle.restore()


@pytest.mark.parametrize(
    "module",
    [
        "mlx_teacache.variants.z_image_base.integration",
        "mlx_teacache.integrations.mflux.forward_flux2",
    ],
)
def test_the_predict_factory_takes_compile_predict_by_position(module: str) -> None:
    """Bug: compile_predict is keyword-only while mflux declares `_predict(transformer, compile_predict=True)`, so a
    positional `self._predict(self.transformer, False)` raises a bare TypeError instead of building the closure."""
    session = SimpleNamespace(_gen_ctx=GenerationContext())
    factory = importlib.import_module(module).make_teacache_predict_factory(session)
    predict = factory(_Transformer(), False)
    assert callable(predict)
    assert session._gen_ctx.predict_built is True


@pytest.mark.parametrize("variant_id", sorted(_TRANSFORMERS))
def test_both_predict_errors_name_the_mflux_and_the_mlx_teacache_versions(variant_id: str) -> None:
    """Bug: the messages ask the user to report both version numbers but show only mflux's, which on a git install
    of mflux main reads as the release it is not; the guard's message says the unpatched step "ran" when mflux has
    usually only built it."""
    both = (f"mflux {version('mflux')}", f"mlx-teacache {version('mlx-teacache')}")
    fake = _fake_pipeline(variant_id, {"stream_layers": True})
    handle = apply_teacache(fake)
    try:
        with pytest.raises(IncompatibleModelError) as unknown_keyword:
            fake.generate_image(num_inference_steps=_STEPS)
        fake.bypass_predict = True
        with pytest.raises(IncompatibleModelError) as not_called:
            fake.generate_image(num_inference_steps=_STEPS)
    finally:
        fake.bypass_predict = False
        handle.restore()
    for error in (unknown_keyword.value, not_called.value):
        assert all(part in str(error) for part in both), str(error)
    assert "built its first denoising step without calling self._predict" in str(not_called.value)


def test_the_predict_errors_say_when_a_version_is_unknown() -> None:
    """Bug: a missing mlx-teacache version prints as "mlx-teacache None" instead of saying it is unknown."""
    from mlx_teacache.errors import _PredictNotCalledError, _UnknownPredictKeywordError

    common = {"actual_type": "ZImage", "actual_model_name": None, "supported": [], "mflux_version": "0.24.0"}
    errors = [
        _UnknownPredictKeywordError(**common, unknown=["stream"], teacache_version=None),
        _PredictNotCalledError(**common, teacache_version=None),
    ]
    for error in errors:
        assert "mlx-teacache (version unknown)" in str(error)
        assert "None" not in str(error)


@pytest.mark.parametrize("bypass", [False, True], ids=["patched", "bypassed"])
@pytest.mark.parametrize("variant_id", sorted(_TRANSFORMERS))
def test_an_error_inside_the_loop_surfaces_as_itself(variant_id: str, bypass: bool) -> None:
    """Bug: the guard checks on the way out of generate_image (or in its finally), so an unrelated failure in the
    loop is replaced by the guard's error. Only the `bypassed` cases can go red for that bug; the `patched` cases are
    the control: the error comes at step 0, before any in-loop callback, with the closure already built."""
    fake = _fake_pipeline(variant_id, {})
    handle = apply_teacache(fake)
    fake.bypass_predict = bypass
    fake.value_error_at_step = 0
    try:
        with pytest.raises(ValueError, match="the model failed at step 0"):
            fake.generate_image(num_inference_steps=_STEPS)
    finally:
        fake.bypass_predict = False
        fake.value_error_at_step = None
        handle.restore()


@pytest.mark.parametrize("variant_id", sorted(_TRANSFORMERS))
def test_after_the_guard_fires_restore_works_and_no_generation_is_in_flight(variant_id: str) -> None:
    """Bug: the guard's error leaves the step count set, so restore() refuses as generation-in-flight and the
    pipeline stays patched."""
    fake = _fake_pipeline(variant_id, {})
    original_generate = fake.generate_image
    handle = apply_teacache(fake)
    fake.bypass_predict = True
    with pytest.raises(IncompatibleModelError):
        fake.generate_image(num_inference_steps=_STEPS)
    assert handle._session._gen_ctx.active_num_steps is None
    handle.restore()
    assert isinstance(vars(fake)["_predict"], _NoRealModel)
    assert fake.generate_image is original_generate
    assert all(
        getattr(fake.callbacks, name) == [] for name in ("before_loop", "in_loop", "after_loop", "interrupt")
    )
    assert not hasattr(fake, "_teacache_handle")


# --- the contract helper the mflux contract smoke uses ----------------------------------------------------

# The call each mflux revision makes at the top of the denoising loop, as source text.
_RELEASE_0_24 = """
def generate_image(self, seed, prompt):
    ctx.before_loop(latents)
    predict = self._predict(self.transformer)
"""
_MAIN_6AEBC2D = """
class ZImage:
    def generate_image(self, seed, prompt):
        ctx.before_loop(latents)
        # A compiled trace would not see the weights the streamer installs and removes around each block.
        predict = self._predict(self.transformer, compile_predict=not self.ssd_stream)

    @staticmethod
    def _predict(transformer, compile_predict: bool = True):
        pass
"""


@pytest.mark.parametrize("source", [_RELEASE_0_24, _MAIN_6AEBC2D], ids=["0.24", "main-6aebc2d"])
@pytest.mark.parametrize("variant_id", sorted(_TRANSFORMERS))
def test_the_installed_factory_binds_every_mflux_predict_call(variant_id: str, source: str) -> None:
    """Bug: the installed factory's signature does not accept a keyword mflux passes (mflux main's
    compile_predict), which the contract smoke would otherwise only see as a substring miss."""
    assert unbound_predict_keywords(source, installed_predict_factory(variant_id)) == []


def _takes_transformer_only(transformer: Any) -> None:
    pass


def _takes_compile_predict_and_a_catch_all(
    transformer: Any, *, compile_predict: bool = True, **rest: Any
) -> None:
    pass


def test_the_helper_reports_a_keyword_the_factory_does_not_take() -> None:
    """Bug: the helper looks only at positional arguments, or not at the call at all, and passes main's call
    against a factory that takes only `transformer`."""
    assert unbound_predict_keywords(_MAIN_6AEBC2D, _takes_transformer_only) == ["compile_predict"]


def test_the_helper_reports_a_keyword_that_only_a_catch_all_would_take() -> None:
    """Bug: the helper counts a keyword that lands in `**kwargs` as accepted, so a factory whose catch-all
    raises at run time passes the contract check for any keyword mflux adds."""
    source = (
        "def generate_image(self):\n    p = self._predict(self.transformer, compile_predict=True, stream=1)\n"
    )
    assert unbound_predict_keywords(source, _takes_compile_predict_and_a_catch_all) == ["stream"]


def test_the_helper_reports_too_many_positional_arguments() -> None:
    """Bug: the helper binds a fixed single positional argument instead of the call's own, so mflux passing a
    second positional argument is not caught."""
    source = "def generate_image(self):\n    p = self._predict(self.transformer, self.vae)\n"
    problems = unbound_predict_keywords(source, _takes_transformer_only)
    assert len(problems) == 1 and "positional" in problems[0]


def test_the_helper_reports_a_first_argument_that_is_not_the_transformer() -> None:
    """Bug: the helper checks only how many positional arguments the call passes, so mflux handing the factory
    another module (`self._predict(self.vae)`) binds cleanly and the contract smoke stays green."""
    source = "def generate_image(self):\n    p = self._predict(self.vae)\n"
    assert unbound_predict_keywords(source, _takes_transformer_only) == [
        "the first argument is not self.transformer"
    ]


def test_the_helper_reports_a_generate_image_that_never_calls_predict() -> None:
    """Bug: a generate_image that stopped calling `self._predict` (the patch would never attach) passes
    because there is no call to bind."""
    source = "def generate_image(self):\n    p = self._build_predict(self.transformer)\n"
    assert unbound_predict_keywords(source, _takes_transformer_only) == [
        "generate_image never calls self._predict"
    ]
