"""mflux 0.21 deletes the per-generation predict closure before the after-loop callbacks so that --low-ram can free
the transformer before decoding (Z-Image and FLUX.2 Klein). With TeaCache applied, nothing else may keep the
transformer reachable once that closure is gone. The proxy variants (FLUX.1, Qwen-Image) have the same duty when
mflux's MemorySaver sets `model.transformer = None`. No weights: duck-typed models and the tiny fake transformers of the
forward tests. The FLUX.2 cases call a forward that imports mflux inside the function, so the file is in the
mflux lane."""

import gc
import importlib
import inspect
import warnings
import weakref
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest

from mlx_teacache import apply_teacache
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
        "mlx_teacache.variants.flux2_klein_base_4b.integration",
    ],
)
def test_the_predict_factory_keeps_no_reference_to_the_transformer(module: str) -> None:
    """Bug: the predict factory itself stores the transformer (or the closure) on the handle or on itself; this unit
    test names the factory when the apply-level test above goes red."""
    factory = importlib.import_module(module).make_teacache_predict_factory(SimpleNamespace())
    transformer = _Transformer()
    ref = weakref.ref(transformer)
    predict = factory(transformer)
    assert callable(predict)
    del transformer, predict
    gc.collect()
    assert ref() is None
