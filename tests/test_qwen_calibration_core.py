"""The Qwen apply core reads every per-checkpoint value from the calibration it is given, so a second
checkpoint can reuse the forward with its own fit. Mflux lane: the fake comes from the variant fakes."""

import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_teacache import apply_teacache
from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache.errors import TeaCacheUncalibratedCheckpointWarning
from mlx_teacache.variants.qwen_image.integration import QwenCalibration, apply_qwen
from tests._fakes import FaithfulCallbackRegistry
from tests._variant_fakes import fake_for_variant

_COEFFS = (1.0, 2.0, 3.0, 4.0, 0.0)


def _calibration(hf_model_id: str = "Qwen/Qwen-Image") -> QwenCalibration:
    return QwenCalibration(
        variant_id="qwen-test",
        coefficients=_COEFFS,
        calibrated_range=(0.1, 0.2),
        default_thresh=0.42,
        provenance=Provenance(
            source="builtin",
            revision="test-revision",
            calibration_dataset="test",
            fit_metric="test",
            fit_metric_value=0.5,
            reference_url="https://example.invalid/test",
            default_thresh=0.42,
        ),
        hf_model_id=hf_model_id,
    )


def _gate_handle(flux):
    return next(cb for cb in flux.callbacks.before_loop if hasattr(cb, "_handle"))._handle


def test_apply_qwen_uses_the_calibration_it_is_given():
    """Bug: apply_qwen still reads the original variant's module-level coefficients, range or default
    threshold instead of the calibration argument."""
    flux = fake_for_variant("qwen-image")
    handle = apply_qwen(flux, _calibration())
    assert handle.coefficients == _COEFFS
    assert handle.rel_l1_thresh == 0.42
    assert handle.provenance.revision == "test-revision"
    assert _gate_handle(flux).calibrated_range == (0.1, 0.2)
    handle.restore()


def test_caller_coefficients_override_the_calibration_and_drop_its_range():
    """Bug: the calibration's range clamps a caller's own polynomial."""
    flux = fake_for_variant("qwen-image")
    handle = apply_qwen(flux, _calibration(), coefficients=(0.0, 0.0, 0.0, 1.0, 0.0))
    assert handle.coefficients == (0.0, 0.0, 0.0, 1.0, 0.0)
    assert handle.rel_l1_thresh == 0.20
    assert _gate_handle(flux).calibrated_range is None
    handle.restore()


def _loaded(model_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        transformer=SimpleNamespace(name="real-qwen-transformer"),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
        model_config=SimpleNamespace(model_name=model_name, base_model=None, aliases=["qwen-image"]),
    )


def test_the_uncalibrated_warning_names_the_calibrations_variant_and_checkpoint():
    """Bug: the warning text keeps the hard-coded 'qwen-image' / Qwen-Image names for another variant."""
    with pytest.warns(TeaCacheUncalibratedCheckpointWarning) as record:
        apply_qwen(
            _loaded("Qwen/Qwen-Image"), _calibration("Qwen/Qwen-Image-2512"), rel_l1_thresh=0.25
        ).restore()
    message = str(record[0].message)
    assert "variant 'qwen-test'" in message
    assert "'Qwen/Qwen-Image-2512'" in message


def test_a_matching_checkpoint_is_silent_for_the_calibrations_own_id():
    """Bug: the check still compares against the original variant's checkpoint, so a second variant warns
    on its own checkpoint (filterwarnings = error turns any warning into a failure)."""
    apply_qwen(
        _loaded("Qwen/Qwen-Image-2512"), _calibration("Qwen/Qwen-Image-2512"), rel_l1_thresh=0.25
    ).restore()


def test_the_warning_points_at_the_caller_of_apply_teacache():
    """Bug: the extra apply_qwen frame shifts the warning onto mlx_teacache internals (api.py or the
    variant's apply) instead of the user's call."""
    flux = _loaded("Qwen/Qwen-Image-2512")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        handle = apply_teacache(flux, rel_l1_thresh=0.25)
    handle.restore()
    ours = [w for w in record if issubclass(w.category, TeaCacheUncalibratedCheckpointWarning)]
    assert len(ours) == 1
    assert Path(ours[0].filename).resolve() == Path(__file__).resolve()
