# tests/test_api.py
"""End-to-end tests for apply_teacache using a synthetic flux model that
mimics enough of the Flux1 surface to exercise the patching/restore cycle.
Real-model parity is in tests/test_parity_*.py."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_teacache import (
    AlreadyPatchedError,
    IncompatibleModelError,
    apply_teacache,
)

pytestmark = pytest.mark.parity


class _FakeCallbackRegistry:
    def __init__(self):
        self.before_loop_callbacks = []
        self.in_loop_callbacks = []
        self.after_loop_callbacks = []
        self.interrupt_callbacks = []

    def register(self, cb):
        self.before_loop_callbacks.append(cb)
        self.after_loop_callbacks.append(cb)
        self.interrupt_callbacks.append(cb)


class _FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embedder = nn.Linear(4, 8, bias=False)

    def __call__(self, **kwargs):
        return mx.zeros((1, 8))


def _make_fake_flux1(model_name="dev"):
    """Build a fake that detect.identify_variant will accept as Flux1."""
    from mflux.models.flux.variants.txt2img.flux import Flux1

    flux = Flux1.__new__(Flux1)
    flux.model_config = SimpleNamespace(model_name=model_name)
    flux.transformer = _FakeTransformer()
    flux.callbacks = _FakeCallbackRegistry()
    flux.generate_image = lambda **kw: "image"
    return flux


def test_apply_and_restore_roundtrip():
    flux = _make_fake_flux1()
    original_transformer = flux.transformer
    original_generate = flux.generate_image
    handle = apply_teacache(flux, rel_l1_thresh=0.25)
    assert handle.variant_id == "flux1-dev"
    assert handle.rel_l1_thresh == 0.25
    assert flux.transformer is not original_transformer
    assert flux.generate_image is not original_generate
    assert flux._teacache_handle is handle
    assert handle._callback_instance in flux.callbacks.before_loop_callbacks
    handle.restore()
    assert flux.transformer is original_transformer
    # generate_image: was an instance attr ⇒ should be restored to original
    assert flux.generate_image is original_generate
    assert getattr(flux, "_teacache_handle", None) is None
    assert handle._callback_instance not in flux.callbacks.before_loop_callbacks


def test_double_apply_raises():
    flux = _make_fake_flux1()
    apply_teacache(flux, rel_l1_thresh=0.25)
    with pytest.raises(AlreadyPatchedError):
        apply_teacache(flux, rel_l1_thresh=0.4)


def test_re_apply_after_restore_succeeds():
    flux = _make_fake_flux1()
    h1 = apply_teacache(flux, rel_l1_thresh=0.25)
    h1.restore()
    h2 = apply_teacache(flux, rel_l1_thresh=0.4)
    assert h2.rel_l1_thresh == 0.4
    h2.restore()


def test_context_manager_restores():
    flux = _make_fake_flux1()
    original_transformer = flux.transformer
    with apply_teacache(flux) as h:
        assert flux.transformer is not original_transformer
        assert h.variant_id == "flux1-dev"
    assert flux.transformer is original_transformer


def test_invalid_threshold_raises():
    flux = _make_fake_flux1()
    with pytest.raises(ValueError, match="rel_l1_thresh"):
        apply_teacache(flux, rel_l1_thresh=1.5)


def test_invalid_skip_negative_raises():
    flux = _make_fake_flux1()
    with pytest.raises(ValueError, match="skip_first"):
        apply_teacache(flux, skip_first_n_steps=-1)


def test_invalid_coefficients_length_raises():
    flux = _make_fake_flux1()
    with pytest.raises(ValueError, match="length 5"):
        apply_teacache(flux, coefficients=[1.0, 2.0])


def test_unsupported_model_raises_incompatible():
    class Other: ...

    other = Other()
    with pytest.raises(IncompatibleModelError):
        apply_teacache(other)


def test_stats_initially_empty():
    flux = _make_fake_flux1()
    h = apply_teacache(flux)
    assert h.stats.total_steps_seen == 0
    assert h.stats.generations == 0
    assert h.stats.speedup_estimate == 1.0
    h.restore()


def test_transactional_apply_rollback_on_failure(monkeypatch):
    """Per audit medium #3: if a mutation after callback registration raises,
    apply_teacache must roll back fully — no leftover callback, no wrapped
    generate_image, no proxy, no sentinel."""
    flux = _make_fake_flux1()
    original_transformer = flux.transformer
    original_generate = flux.generate_image
    original_callback_count = len(flux.callbacks.before_loop_callbacks)

    # Make wrap_generate_image raise.
    from mlx_teacache.integrations.mflux import lifecycle

    def boom(flux, handle):
        raise RuntimeError("simulated wrap failure")

    monkeypatch.setattr(lifecycle, "wrap_generate_image", boom)

    with pytest.raises(RuntimeError, match="simulated wrap failure"):
        apply_teacache(flux)

    # Full rollback: no leftover state.
    assert flux.transformer is original_transformer
    assert flux.generate_image is original_generate
    assert len(flux.callbacks.before_loop_callbacks) == original_callback_count
    assert getattr(flux, "_teacache_handle", None) is None


@pytest.mark.parity
def test_apply_teacache_accepts_flux2_klein_9b():
    """Smoke: apply_teacache returns a handle with the right variant_id on Klein 9B.
    Catches api.py regressions in the variant_id Literal or the FLUX.2 _predict guard."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    from mlx_teacache import apply_teacache

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_9b())
    flux.freeze()
    handle = apply_teacache(flux)
    try:
        assert handle.variant_id == "flux2-klein-9b"
        assert len(handle.coefficients) == 5
        assert handle.provenance.source == "builtin"
    finally:
        handle.restore()


@pytest.mark.parity
def test_apply_teacache_accepts_flux2_klein_base_4b():
    """Smoke: apply_teacache returns a handle with the right variant_id on Klein base-4B.
    Catches api.py regressions in the variant_id Literal or the FLUX.2 _predict guard."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    from mlx_teacache import apply_teacache

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
    flux.freeze()
    handle = apply_teacache(flux)
    try:
        assert handle.variant_id == "flux2-klein-base-4b"
        assert len(handle.coefficients) == 5
        assert handle.provenance.source == "builtin"
    finally:
        handle.restore()
