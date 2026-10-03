"""mflux 0.21 deletes the per-generation predict closure before the after-loop callbacks so that --low-ram can free
the transformer before decoding (Z-Image and FLUX.2 Klein). With TeaCache applied, nothing else may keep the
transformer reachable once that closure is gone. No weights, no mflux: duck-typed models and the tiny fake
transformers of the forward tests."""

import gc
import importlib
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


@pytest.mark.xfail(strict=True, reason="the FLUX.1 rollback holds the original transformer until restore()")
@pytest.mark.parametrize("variant_id", ["flux1-dev", "flux1-krea-dev", "qwen-image"])
def test_proxy_variants_free_the_transformer_once_the_pipeline_drops_it(variant_id: str) -> None:
    """Bug (known gap): the proxy variants' rollback holds the original transformer so restore() can put it back,
    so dropping flux.transformer frees nothing while the handle lives and --low-ram cannot reclaim it. Strict xfail:
    when the transformer does get freed this test turns red and asks for the xfail mark to be removed."""
    fake = fake_for_variant(variant_id)
    fake.transformer = _Transformer()
    ref = weakref.ref(fake.transformer)
    with warnings.catch_warnings():
        # Qwen-Image's alias warns about an uncalibrated checkpoint on mflux >= 0.19; irrelevant here.
        warnings.simplefilter("ignore")
        handle = apply_teacache(fake)
    fake.transformer = None
    gc.collect()
    assert handle.variant_id == variant_id
    assert ref() is None


class _Transformer:
    """Stand-in transformer for the factory-level test; only its lifetime matters."""


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
