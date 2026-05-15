# tests/test_errors.py
"""Each exception names the parameter and the actual bad value in its message,
and inherits from TeaCacheError so users can catch the whole family with one
except clause. Tests are pure Python — no mflux, no MLX."""

from mlx_teacache.errors import (
    AlreadyPatchedError,
    CalibrationError,
    Img2ImgNotSupportedError,
    IncompatibleModelError,
    InternalStateError,
    InvalidStepWindowError,
    MissingGenerationContextError,
    TeaCacheError,
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
        Img2ImgNotSupportedError,
        InternalStateError,
    ]:
        assert issubclass(cls, TeaCacheError)


def test_internal_state_error_carries_message():
    err = InternalStateError("cached_residual was None on a skipped step")
    msg = str(err)
    assert "cached_residual" in msg


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
    assert "num_steps=2" in msg


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


def test_img2img_not_supported_names_variant():
    err = Img2ImgNotSupportedError(variant="flux1-dev")
    msg = str(err)
    assert "flux1-dev" in msg
    assert "image_path" in msg or "image_strength" in msg
