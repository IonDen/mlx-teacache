"""Mid-apply failure must leave the model pristine, for every _predict-replacing
variant, without weights. Two failure points: callbacks.register raising after a
partial append, and wrap_generate_image raising after registration succeeded.
FLUX.1 and Qwen have the same pins in tests/test_api.py and
tests/test_qwen_apply_restore.py; this file closes the gap for the other five."""

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
_LISTS = ("before_loop", "in_loop", "after_loop", "interrupt")


def _fake_flux():
    return SimpleNamespace(callbacks=FaithfulCallbackRegistry(), generate_image=lambda **kw: "image")


def _lists(flux):
    return {n: list(getattr(flux.callbacks, n)) for n in _LISTS}


@pytest.mark.parametrize("variant", _PREDICT_VARIANTS)
def test_register_failure_leaves_nothing_behind(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """bug caught: flux.callbacks.register(callback) outside the try — a partial
    registration survives the exception with no rollback."""
    apply = importlib.import_module(f"mlx_teacache.variants.{variant}.integration").apply
    flux = _fake_flux()
    original_gi = flux.generate_image
    before = _lists(flux)

    def _partial_then_boom(cb):
        flux.callbacks.before_loop.append(cb)
        raise RuntimeError("register boom")

    monkeypatch.setattr(flux.callbacks, "register", _partial_then_boom)
    with pytest.raises(RuntimeError, match="register boom"):
        apply(flux, rel_l1_thresh=0.25)
    assert _lists(flux) == before, "callback left registered"
    assert "_predict" not in vars(flux), "_predict left patched"
    assert flux.generate_image is original_gi, "generate_image left wrapped"


@pytest.mark.parametrize("variant", _PREDICT_VARIANTS)
def test_wrap_failure_rolls_back_the_registration(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """bug caught: deleting the rollback loop in the except clause (mutation-verified
    2026-09-05 to leave every other fast FLUX.2 test green)."""
    mod = importlib.import_module(f"mlx_teacache.variants.{variant}.integration")
    flux = _fake_flux()
    original_gi = flux.generate_image
    before = _lists(flux)
    monkeypatch.setattr(
        mod, "wrap_generate_image", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("wrap boom"))
    )
    with pytest.raises(RuntimeError, match="wrap boom"):
        mod.apply(flux, rel_l1_thresh=0.25)
    assert _lists(flux) == before, "callback left registered"
    assert "_predict" not in vars(flux), "_predict left patched"
    assert flux.generate_image is original_gi, "generate_image left wrapped"
