"""Qwen-Image's coefficients were calibrated on Qwen/Qwen-Image. mflux 0.19
maps the `qwen-image` alias to Qwen/Qwen-Image-2512, so a user who upgrades
mflux silently loads a checkpoint the polynomial has never been measured on.
apply() must say so, unless the caller brought their own coefficients.
Pure-core: duck-typed fake flux, no mflux import."""

from types import SimpleNamespace

import pytest

from mlx_teacache.errors import TeaCacheUncalibratedCheckpointWarning
from mlx_teacache.variants.qwen_image.config import META
from mlx_teacache.variants.qwen_image.integration import apply
from tests._fakes import FaithfulCallbackRegistry


def _fake_flux(model_name: str | None, base_model: str | None = None):
    flux = SimpleNamespace(
        transformer=SimpleNamespace(name="real-qwen-transformer"),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
    )
    if model_name is not None:
        flux.model_config = SimpleNamespace(
            model_name=model_name, base_model=base_model, aliases=["qwen-image", "qwen"]
        )
    return flux


def test_meta_names_the_calibrated_checkpoint():
    """bug caught: META['hf_model_id'] drifting away from the checkpoint the
    committed calibration JSON was captured on."""
    assert META["hf_model_id"] == "Qwen/Qwen-Image"


def test_apply_warns_when_the_loaded_checkpoint_is_not_the_calibrated_one():
    """bug caught: no check at all, or reading aliases instead of model_name
    (the 2512 checkpoint keeps the `qwen-image` alias)."""
    flux = _fake_flux("Qwen/Qwen-Image-2512")
    with pytest.warns(TeaCacheUncalibratedCheckpointWarning, match="Qwen/Qwen-Image-2512"):
        handle = apply(flux, rel_l1_thresh=0.25)
    handle.restore()


def test_apply_is_silent_on_the_calibrated_checkpoint():
    # filterwarnings = error: any warning raised here fails the test.
    apply(_fake_flux("Qwen/Qwen-Image"), rel_l1_thresh=0.25).restore()


def test_user_coefficients_silence_the_warning():
    """bug caught: warning on every mismatch, even when the caller calibrated
    the checkpoint themselves and passed the result."""
    apply(
        _fake_flux("Qwen/Qwen-Image-2512"), rel_l1_thresh=0.25, coefficients=(0.0, 0.0, 0.0, 1.0, 0.0)
    ).restore()


def test_no_model_name_means_no_warning():
    """bug caught: AttributeError on duck-typed models without model_config."""
    apply(_fake_flux(None), rel_l1_thresh=0.25).restore()


def test_a_local_mirror_declared_through_base_model_is_silent():
    """bug caught: warning on every custom model_name. mflux resolves a local path
    or a pre-quantized mirror by copying the base config and setting base_model to
    the base's model_name, so `--base-model qwen-image` on a mirror of the
    calibrated checkpoint must not warn."""
    apply(_fake_flux("/models/qwen-image-q4", base_model="Qwen/Qwen-Image"), rel_l1_thresh=0.25).restore()


def test_the_message_tells_a_mirror_owner_what_to_do():
    """bug caught: a message that reads as "wrong model" to someone who loaded a
    mirror of the right one without declaring the base."""
    with pytest.warns(TeaCacheUncalibratedCheckpointWarning, match="mirror"):
        apply(_fake_flux("/models/some-checkpoint"), rel_l1_thresh=0.25).restore()
