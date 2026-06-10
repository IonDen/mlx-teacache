# tests/test_errors.py
"""Each exception names the parameter and the actual bad value in its message,
and inherits from TeaCacheError so users can catch the whole family with one
except clause. Tests are pure Python — no mflux, no MLX."""

import pytest

from mlx_teacache.errors import (
    AlreadyPatchedError,
    CalibrationError,
    IncompatibleModelError,
    InternalStateError,
    InvalidStepWindowError,
    MissingGenerationContextError,
    TeaCacheError,
    TransformerShapeError,
)
from mlx_teacache.stats import StatsFrozenError


def test_all_subclass_teacache_error():
    for cls in [
        IncompatibleModelError,
        AlreadyPatchedError,
        CalibrationError,
        TransformerShapeError,
        InvalidStepWindowError,
        MissingGenerationContextError,
        InternalStateError,
        StatsFrozenError,
    ]:
        assert issubclass(cls, TeaCacheError)


def test_internal_state_error_raised_on_length_invariant_violation():
    """InternalStateError fires when staged decisions != num_inference_steps."""
    from mlx_teacache._kernel.stats import TeaCacheStats
    from mlx_teacache.errors import InternalStateError

    stats = TeaCacheStats()
    with pytest.raises(InternalStateError, match="length invariant"):
        stats.finalize_last_generation(num_inference_steps=3, cfg_was_active=False)


def test_incompatible_model_lists_supported_variants():
    err = IncompatibleModelError(
        actual_type="SomeOtherModel",
        actual_model_name="foo",
        supported=["flux1-dev", "flux1-schnell", "flux2-klein-4b"],
    )
    msg = str(err)
    assert "SomeOtherModel" in msg
    assert "flux1-dev" in msg
    assert "flux2-klein-4b" in msg


def test_already_patched_includes_existing_handle_info():
    err = AlreadyPatchedError(variant_id="flux2-klein-4b", rel_l1_thresh=0.25)
    msg = str(err)
    assert "flux2-klein-4b" in msg
    assert "0.25" in msg
    assert "restore()" in msg.lower()


def test_invalid_step_window_names_values():
    err = InvalidStepWindowError(skip_first=1, skip_last=1, num_steps=2)
    msg = str(err)
    assert "skip_first=1" in msg
    assert "skip_last=1" in msg
    assert "active_num_steps=2" in msg
    # The legacy num_steps kwarg still works for construction; the old wording
    # "num_steps=2" is no longer in the message.


def test_invalid_step_window_message_reports_both_counts_when_provided():
    """New img2img call sites pass both nominal and active counts."""
    err = InvalidStepWindowError(
        skip_first=2,
        skip_last=2,
        num_steps=4,
        nominal_num_inference_steps=25,
        active_num_steps=4,
    )
    msg = str(err)
    assert "active_num_steps=4" in msg
    assert "nominal_num_inference_steps=25" in msg


def test_transformer_shape_error_names_step_and_shapes():
    err = TransformerShapeError(step_idx=7, expected=(1, 4096, 3072), actual=(1, 4097, 3072))
    msg = str(err)
    assert "step_idx=7" in msg
    assert "(1, 4096, 3072)" in msg
    assert "(1, 4097, 3072)" in msg


def test_calibration_error_names_variant():
    err = CalibrationError(variant_id="flux2-klein-4b", reason="missing")
    msg = str(err)
    assert "flux2-klein-4b" in msg
    assert "missing" in msg


def test_missing_generation_context_has_remediation():
    err = MissingGenerationContextError()
    msg = str(err)
    assert "handle.restore()" in msg
    assert "apply_teacache" in msg


def test_img2img_not_supported_error_removed_from_errors_module():
    """v0.3.0 removed the class deprecated in v0.2.0."""
    with pytest.raises(ImportError):
        from mlx_teacache.errors import Img2ImgNotSupportedError  # noqa: F401


def test_img2img_not_supported_error_removed_from_top_level():
    with pytest.raises(ImportError):
        from mlx_teacache import Img2ImgNotSupportedError  # noqa: F401


def test_value_error_guard_is_catchable_as_teacache_error():
    from mlx_teacache.errors import TeaCacheValueError

    assert issubclass(TeaCacheValueError, TeaCacheError)
    assert issubclass(TeaCacheValueError, ValueError)  # IS-A ValueError preserved


def test_import_is_cycle_free():
    import importlib

    importlib.import_module("mlx_teacache")
    importlib.import_module("mlx_teacache._kernel.stats")


def test_apply_teacache_misuse_raises_teacache_value_error():
    from mlx_teacache import apply_teacache
    from mlx_teacache.errors import TeaCacheValueError

    with pytest.raises(TeaCacheValueError):
        apply_teacache(object(), skip_first_n_steps=-1)


def test_stats_frozen_error_caught_by_teacache_error():
    from mlx_teacache._kernel.stats import StepDecision, TeaCacheStats

    stats = TeaCacheStats()
    stats._freeze()
    frozen_decision = StepDecision(
        step_idx=0,
        timestep=1.0,
        rel_l1=None,
        accumulated_distance=0.0,
        decision="computed",
    )
    with pytest.raises(TeaCacheError):
        stats.record(frozen_decision)


def test_builtin_coefficient_self_check_rejects_bad_tuple(monkeypatch):
    """_build_registry validates COEFFICIENTS; corrupt data raises CalibrationError."""
    import importlib
    import types

    from mlx_teacache.errors import CalibrationError
    from mlx_teacache.variants import _build_registry

    # Build a minimal fake config module with an invalid COEFFICIENTS tuple
    bad_config = types.ModuleType("fake_config")
    bad_config.META = {"variant_id": "fake-variant"}  # type: ignore[attr-defined]
    bad_config.COEFFICIENTS = (1.0, float("nan"), 0.0, 0.0, 0.0)  # type: ignore[attr-defined]

    bad_detect = types.ModuleType("fake_detect")
    bad_detect.matches = lambda flux: False  # type: ignore[attr-defined]

    # Patch importlib.import_module inside the variants package so one iter
    # returns our bad config while all other real modules pass through.
    real_import = importlib.import_module
    call_count: list[int] = [0]

    def patched_import(name: str, *args: object, **kwargs: object) -> types.ModuleType:
        if name == "mlx_teacache.variants.fake_pkg.config":
            return bad_config
        if name == "mlx_teacache.variants.fake_pkg.detect":
            return bad_detect
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    # Also patch pkgutil.iter_modules to inject our fake package
    import pkgutil

    def patched_iter(path: object, *args: object, **kwargs: object) -> object:
        yield from [pkgutil.ModuleInfo(module_finder=None, name="fake_pkg", ispkg=True)]  # type: ignore[call-arg]
        call_count[0] += 1

    monkeypatch.setattr(pkgutil, "iter_modules", patched_iter)
    monkeypatch.setattr(importlib, "import_module", patched_import)

    with pytest.raises(CalibrationError) as exc_info:
        _build_registry()

    assert "fake-variant" in str(exc_info.value)
