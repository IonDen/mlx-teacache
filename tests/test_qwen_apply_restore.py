"""Qwen-Image apply()/restore() — proxy transformer attr symmetry. Pure-core
(duck-typed fake flux; never imports mflux weights)."""

from types import SimpleNamespace

from mlx_teacache.variants.qwen_image.integration import ProxyQwenTransformer, apply
from tests._fakes import FaithfulCallbackRegistry


def _fake_transformer():
    return SimpleNamespace(name="real-qwen-transformer")


def _fake_flux():
    return SimpleNamespace(
        transformer=_fake_transformer(),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
    )


def test_apply_swaps_in_proxy_and_restore_recovers_original() -> None:
    flux = _fake_flux()
    original = flux.transformer
    original_gi = flux.generate_image
    handle = apply(flux, rel_l1_thresh=0.25)
    assert isinstance(flux.transformer, ProxyQwenTransformer)
    assert flux.transformer is not original
    assert flux.generate_image is not original_gi  # wrapped by apply()
    handle.restore()
    assert flux.transformer is original
    assert flux.generate_image is original_gi  # generate_image unwrapped too


def test_restore_unsubscribes_lifecycle_callback() -> None:
    flux = _fake_flux()
    handle = apply(flux, rel_l1_thresh=0.25)
    cb = handle._callback_instance
    assert cb in flux.callbacks.before_loop  # registered after apply()
    handle.restore()
    assert cb not in flux.callbacks.before_loop
    assert cb not in flux.callbacks.after_loop
    assert cb not in flux.callbacks.interrupt


def test_proxy_delegates_parameters_to_inner() -> None:
    inner = SimpleNamespace(parameters=lambda: {"w": 1}, trainable_parameters=lambda: {"w": 1})
    proxy = ProxyQwenTransformer(inner=inner, handle=object())
    assert proxy.parameters() == {"w": 1}
    assert proxy.trainable_parameters() == {"w": 1}


def test_proxy_getattr_falls_back_to_inner() -> None:
    inner = SimpleNamespace(img_in="IMG_IN_MODULE")
    proxy = ProxyQwenTransformer(inner=inner, handle=object())
    assert proxy.img_in == "IMG_IN_MODULE"
