"""Each FLUX.2 / Z-Image variant replaces `flux._predict` with an instance
attribute. Teardown must restore a PRE-EXISTING instance `_predict` rather than
deleting it — mirroring the `_generate_image_was_instance_attr` bookkeeping
(backlog 0031 #1).

On a real Flux2Klein, `_predict` is a class method (not an instance attr), so the
common case (no instance attr at apply time) must still delete the override and
re-expose the class method. These tests pin both directions, parametrized over
every `_predict`-replacement variant. Pure-core: calling a variant `apply()`
directly is duck-typed and never imports mflux.
"""

import importlib
from types import SimpleNamespace

import pytest

from tests._fakes import FaithfulCallbackRegistry

_PREDICT_VARIANTS = [
    "flux2_klein_4b",
    "flux2_klein_9b",
    "flux2_klein_base_4b",
    "flux2_klein_base_9b",
    "z_image_base",
]


def _variant_apply(variant):
    return importlib.import_module(f"mlx_teacache.variants.{variant}.integration").apply


def _fake_flux(**extra):
    return SimpleNamespace(
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
        **extra,
    )


@pytest.mark.parametrize("variant", _PREDICT_VARIANTS)
def test_restore_recovers_preexisting_instance_predict(variant: str) -> None:
    apply = _variant_apply(variant)
    sentinel = object()
    flux = _fake_flux(_predict=sentinel)
    handle = apply(flux, rel_l1_thresh=0.25)
    assert flux._predict is not sentinel  # patched over by the factory
    handle.restore()
    assert getattr(flux, "_predict", None) is sentinel  # restored, not deleted


@pytest.mark.parametrize("variant", _PREDICT_VARIANTS)
def test_restore_deletes_predict_when_none_preexisting(variant: str) -> None:
    apply = _variant_apply(variant)
    flux = _fake_flux()
    handle = apply(flux, rel_l1_thresh=0.25)
    assert "_predict" in vars(flux)  # patched
    handle.restore()
    assert "_predict" not in vars(flux)  # default case unchanged (class method re-exposed)
