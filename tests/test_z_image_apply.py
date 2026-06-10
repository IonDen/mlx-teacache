"""Pure-core tests for the z-image-base apply()/restore() + predict factory.

These exercise the mflux-FREE half of the variant's integration.py — threshold/
coefficient/provenance resolution, the VariantPatch apply/rollback cycle, and
the predict closure's context-consumption + skip-window validation — using a
SimpleNamespace fake flux. No real ZImage weights and no mflux import (the whole
apply_teacache -> z_image_base.apply path is duck-typed), so this file stays in
the pure-core lane (NOT in conftest._MFLUX_FILES, NOT parity-marked) and counts
toward CI coverage. The model-dependent forward re-walk is covered by the
mflux-marked tests/test_parity_z_image.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_teacache import AlreadyPatchedError, apply_teacache
from mlx_teacache._kernel.gate import GateDecision
from mlx_teacache.errors import InvalidStepWindowError, MissingGenerationContextError
from mlx_teacache.variants.z_image_base.config import COEFFICIENTS, DEFAULT_THRESH
from mlx_teacache.variants.z_image_base.integration import (
    _InternalHandle,
    _step_decision_from_gate,
    make_teacache_predict_factory,
)
from tests._fakes import FaithfulCallbackRegistry


def _fake_zimage() -> SimpleNamespace:
    """Duck-typed Z-Image: detect.matches keys on aliases; apply() only touches
    callbacks / generate_image / _predict. No mflux, no weights."""
    return SimpleNamespace(
        model_config=SimpleNamespace(aliases=["z-image", "zimage"]),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
    )


# --- apply() / restore() roundtrip via the public API ---------------------


def test_apply_and_restore_roundtrip():
    flux = _fake_zimage()
    original_generate = flux.generate_image
    handle = apply_teacache(flux, rel_l1_thresh=0.25)

    assert handle.variant_id == "z-image-base"
    assert handle.rel_l1_thresh == 0.25
    assert "_predict" in vars(flux)  # patched
    assert flux.generate_image is not original_generate
    assert handle._callback_instance in flux.callbacks.before_loop

    handle.restore()
    assert "_predict" not in vars(flux)  # class staticmethod re-exposed
    assert flux.generate_image is original_generate
    assert handle._callback_instance not in flux.callbacks.before_loop


def test_default_threshold_is_variant_default():
    flux = _fake_zimage()
    handle = apply_teacache(flux)  # no explicit threshold, builtin coefficients
    try:
        assert handle.rel_l1_thresh == DEFAULT_THRESH == 0.12
        assert handle.coefficients == COEFFICIENTS
        assert handle.provenance.source == "builtin"
    finally:
        handle.restore()


def test_custom_coefficients_use_fallback_threshold_and_user_provenance():
    flux = _fake_zimage()
    custom = (1.0, 0.0, 0.0, 0.0, 0.0)
    handle = apply_teacache(flux, coefficients=custom)
    try:
        # custom coefficients skip the per-variant default -> package fallback 0.20
        assert handle.rel_l1_thresh == 0.20
        assert handle.coefficients == custom
        assert handle.provenance.source == "user"
    finally:
        handle.restore()


def test_context_manager_restores():
    flux = _fake_zimage()
    with apply_teacache(flux) as h:
        assert "_predict" in vars(flux)
        assert h.rel_l1_thresh == 0.12
    assert "_predict" not in vars(flux)


def test_double_apply_raises():
    flux = _fake_zimage()
    h = apply_teacache(flux)
    try:
        with pytest.raises(AlreadyPatchedError):
            apply_teacache(flux)
    finally:
        h.restore()


# --- predict factory: context + skip-window validation (no forward) -------


def _ready_handle(*, skip_first: int = 1, skip_last: int = 1) -> _InternalHandle:
    return _InternalHandle(
        rel_l1_thresh=0.12,
        coefficients=COEFFICIENTS,
        skip_first_n_steps=skip_first,
        skip_last_n_steps=skip_last,
    )


def _call_predict(predict, *, negative=None):  # noqa: ANN001
    return predict(
        latents=mx.ones((1, 4, 8)),
        timestep=mx.array([1.0]),
        sigmas=mx.ones((4,)),
        text_encodings=mx.ones((1, 8, 16)),
        negative_encodings=negative,
        guidance=1.0,
    )


def test_factory_returns_fresh_closure_per_generation():
    handle = _ready_handle()
    factory = make_teacache_predict_factory(handle)
    transformer = SimpleNamespace()
    assert factory(transformer) is not factory(transformer)


def test_missing_context_raises():
    handle = _ready_handle()  # _gen_ctx.active_num_steps is None
    predict = make_teacache_predict_factory(handle)(SimpleNamespace())
    with pytest.raises(MissingGenerationContextError):
        _call_predict(predict)


def test_invalid_step_window_raises():
    handle = _ready_handle(skip_first=1, skip_last=1)
    # 1 + 1 >= 2 active steps -> invalid window. Mark context ready first.
    handle._gen_ctx.token += 1
    handle._gen_ctx.active_num_steps = 2
    handle._gen_ctx.consumed_at_token = None
    predict = make_teacache_predict_factory(handle)(SimpleNamespace())
    with pytest.raises(InvalidStepWindowError):
        _call_predict(predict)


# --- _step_decision_from_gate pure mapping --------------------------------


def test_step_decision_from_gate_maps_fields():
    decision = GateDecision(
        kind="skipped",
        should_compute=False,
        should_update_cache=False,
        rel_l1=0.05,
        predicted_distance=0.1,
        accumulated_distance=0.3,
    )
    sd = _step_decision_from_gate(decision, step_idx=7, timestep=123.0)
    assert sd.step_idx == 7
    assert sd.timestep == 123.0
    assert sd.rel_l1 == 0.05
    assert sd.accumulated_distance == 0.3
    assert sd.decision == "skipped"
