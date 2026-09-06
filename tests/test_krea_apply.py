"""flux1-krea-dev apply()/restore() on a duck-typed fake: reuses the FLUX.1 proxy
and restores it; registry exposes the variant. Pure-core (no mflux import)."""

from types import SimpleNamespace

import pytest

from mlx_teacache.variants import _REGISTRY
from mlx_teacache.variants.flux1_dev.integration import ProxyFlux1Transformer
from mlx_teacache.variants.flux1_krea_dev import config
from mlx_teacache.variants.flux1_krea_dev.integration import apply
from tests._fakes import FaithfulCallbackRegistry


def _fake_flux():
    return SimpleNamespace(
        transformer=SimpleNamespace(name="krea-transformer"),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
    )


def test_registry_lists_the_variant_with_its_meta() -> None:
    entry = _REGISTRY["flux1-krea-dev"]
    assert entry["META"]["hf_model_id"] == "black-forest-labs/FLUX.1-Krea-dev"
    assert entry["default_thresh"] == config.DEFAULT_THRESH
    assert entry["META"]["non_distilled"] is True


def test_apply_swaps_in_the_flux1_proxy_and_restore_recovers_it() -> None:
    flux = _fake_flux()
    original = flux.transformer
    handle = apply(flux, rel_l1_thresh=0.25)
    assert isinstance(flux.transformer, ProxyFlux1Transformer)
    assert handle.rel_l1_thresh == 0.25
    assert handle.provenance.source == "builtin"
    handle.restore()
    assert flux.transformer is original
    assert handle._callback_instance not in flux.callbacks.before_loop


def test_default_threshold_comes_from_config() -> None:
    flux = _fake_flux()
    handle = apply(flux)
    try:
        assert handle.rel_l1_thresh == config.DEFAULT_THRESH
        assert handle.coefficients == config.COEFFICIENTS
    finally:
        handle.restore()


def test_register_failure_leaves_nothing_behind(monkeypatch) -> None:
    flux = _fake_flux()
    original = flux.transformer

    def _partial_then_boom(cb):
        flux.callbacks.before_loop.append(cb)
        raise RuntimeError("register boom")

    monkeypatch.setattr(flux.callbacks, "register", _partial_then_boom)
    with pytest.raises(RuntimeError, match="register boom"):
        apply(flux, rel_l1_thresh=0.25)
    assert flux.transformer is original
    assert flux.callbacks.before_loop == []
