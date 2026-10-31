# tests/test_errors.py
"""Each exception names the parameter and the actual bad value in its message,
and inherits from TeaCacheError so users can catch the whole family with one
except clause. Tests are pure Python — no mflux, no MLX."""

import warnings

import pytest

import mlx_teacache.errors as errors_module
from mlx_teacache._kernel.stats import StatsFrozenError
from mlx_teacache.errors import (
    AlreadyPatchedError,
    CalibrationError,
    IncompatibleModelError,
    InternalStateError,
    InvalidStepWindowError,
    MissingGenerationContextError,
    TeaCacheDisabledWarning,
    TeaCacheError,
    TeaCacheNoBenefitWarning,
    TeaCacheUncalibratedCheckpointWarning,
    TeaCacheUntestedMfluxWarning,
    TransformerShapeError,
)


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
    with pytest.warns(DeprecationWarning, match="active_num_steps"):
        err = InvalidStepWindowError(skip_first=1, skip_last=1, num_steps=2)
    msg = str(err)
    assert "skip_first=1" in msg
    assert "skip_last=1" in msg
    assert "active_num_steps=2" in msg
    # The deprecated num_steps kwarg still works for construction; the old wording
    # "num_steps=2" is no longer in the message.


def test_invalid_step_window_message_reports_both_counts_when_provided():
    """New img2img call sites pass both nominal and active counts."""
    with pytest.warns(DeprecationWarning, match="active_num_steps"):
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


def test_num_steps_alias_warns_at_the_caller():
    with pytest.warns(DeprecationWarning, match="active_num_steps") as rec:
        err = InvalidStepWindowError(skip_first=1, skip_last=2, num_steps=3)
    assert len(rec) == 1
    assert rec[0].filename == __file__
    assert "num_steps= is deprecated" in str(rec[0].message)
    assert "removed in 2.0" in str(rec[0].message)
    assert str(err) == (
        "skip_first_n_steps + skip_last_n_steps must be < active denoising steps. "
        "Got skip_first=1, skip_last=2, active_num_steps=3 (sum 3 >= 3)."
    )
    assert (err.skip_first, err.skip_last, err.num_steps, err.active_num_steps) == (1, 2, 3, 3)
    assert err.nominal_num_inference_steps is None


def test_num_steps_alias_loses_to_active_num_steps():
    with pytest.warns(DeprecationWarning, match="active_num_steps"):
        err = InvalidStepWindowError(skip_first=2, skip_last=2, num_steps=3, active_num_steps=4)
    assert (err.num_steps, err.active_num_steps) == (3, 4)
    assert "active_num_steps=4 (sum 4 >= 4)" in str(err)


def test_active_num_steps_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        err = InvalidStepWindowError(skip_first=1, skip_last=1, active_num_steps=2)
    assert (err.num_steps, err.active_num_steps) == (2, 2)
    assert "active_num_steps=2 (sum 2 >= 2)" in str(err)


def test_missing_step_count_raises_type_error():
    with pytest.raises(TypeError, match=r"^InvalidStepWindowError needs active_num_steps=$"):
        InvalidStepWindowError(skip_first=1, skip_last=1)


def test_check_step_window_refusal_does_not_warn():
    from mlx_teacache import check_step_window

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(InvalidStepWindowError) as info:
            check_step_window(2)
    assert (info.value.num_steps, info.value.active_num_steps) == (2, 2)


_RUNTIME_WARNINGS = [
    TeaCacheNoBenefitWarning,
    TeaCacheUncalibratedCheckpointWarning,
    TeaCacheDisabledWarning,
    TeaCacheUntestedMfluxWarning,
]


@pytest.mark.parametrize("cls", _RUNTIME_WARNINGS, ids=lambda c: c.__name__)
def test_every_runtime_warning_derives_from_teacache_warning(cls: type[Warning]) -> None:
    assert issubclass(cls, errors_module.TeaCacheWarning)
    assert issubclass(errors_module.TeaCacheWarning, UserWarning)
    assert not issubclass(errors_module.TeaCacheWarning, DeprecationWarning)


def test_filtering_teacache_warning_silences_all_four() -> None:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=errors_module.TeaCacheWarning)
        for cls in _RUNTIME_WARNINGS:
            warnings.warn(cls("x"), stacklevel=1)
    assert rec == []


def test_untested_mflux_warning_docstring_states_the_open_upper_bound() -> None:
    """Bug: help() tells users the extra caps mflux two minors past the verified release,
    which stopped being true when the cap was dropped."""
    doc = TeaCacheUntestedMfluxWarning.__doc__ or ""
    assert "no upper bound" in doc
    assert "two minor" not in doc


_RUNTIME_WARNING_NAMES = [
    "TeaCacheNoBenefitWarning",
    "TeaCacheUncalibratedCheckpointWarning",
    "TeaCacheDisabledWarning",
    "TeaCacheUntestedMfluxWarning",
]


@pytest.mark.parametrize("name", _RUNTIME_WARNING_NAMES)
def test_a_filter_naming_one_warning_still_matches_only_that_one(name: str) -> None:
    """Bug: two warnings collapse into one class, so a filter meant for one silences the other."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=getattr(errors_module, name))
        for other in _RUNTIME_WARNING_NAMES:
            warnings.warn(getattr(errors_module, other)("x"), stacklevel=1)
    assert sorted(type(w.message).__name__ for w in rec) == sorted(
        other for other in _RUNTIME_WARNING_NAMES if other != name
    )


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
