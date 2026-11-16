"""Every variant warns at apply time when the model's config names a checkpoint that is neither the one its built-in
coefficients were fitted on nor declared as its base, and the caller passed no coefficients of their own. match_variant
reports the same answer before anything loads. Pure-core: duck-typed fakes, no mflux import."""

import types
import warnings
from types import SimpleNamespace

import pytest

from mlx_teacache import TeaCacheUncalibratedCheckpointWarning, apply_teacache, match_variant
from mlx_teacache.errors import CalibrationError
from mlx_teacache.variants import _REGISTRY, _detect_contract
from tests._fakes import FaithfulCallbackRegistry

# The checkpoint each variant's built-in coefficients were fitted on (the calibration records in each config.py).
_CALIBRATED_ON = {
    "flux1-dev": "black-forest-labs/FLUX.1-dev",
    "flux1-krea-dev": "black-forest-labs/FLUX.1-Krea-dev",
    "flux1-schnell": "black-forest-labs/FLUX.1-schnell",
    "flux2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
    "flux2-klein-9b": "black-forest-labs/FLUX.2-klein-9B",
    "flux2-klein-base-4b": "black-forest-labs/FLUX.2-klein-base-4B",
    "flux2-klein-base-9b": "black-forest-labs/FLUX.2-klein-base-9B",
    "qwen-image": "Qwen/Qwen-Image",
    "qwen-image-2512": "Qwen/Qwen-Image-2512",
    "z-image-base": "Tongyi-MAI/Z-Image",
}
_VARIANTS = sorted(_CALIBRATED_ON)
_COEFFS = (0.0, 0.0, 0.0, 1.0, 0.0)


class _Pipeline:
    """A pipeline class defined outside mflux: detection falls back to the config's aliases."""


def _flux(variant_id: str, model_name: str | None, base_model: str | None = None) -> _Pipeline:
    flux = _Pipeline()
    flux.model_config = SimpleNamespace(  # type: ignore[attr-defined]
        aliases=[_REGISTRY[variant_id]["model_names"][0]], model_name=model_name, base_model=base_model
    )
    flux.transformer = SimpleNamespace(name="real-transformer")  # type: ignore[attr-defined]
    flux.callbacks = FaithfulCallbackRegistry()  # type: ignore[attr-defined]
    flux.generate_image = lambda **kw: "image"  # type: ignore[attr-defined]
    return flux


def _apply_warns_uncalibrated(flux: object, **kwargs: object) -> bool:
    """Apply with every other warning ignored and the uncalibrated warning raised; True when it was raised (the model
    is then left unpatched), False when apply went through (the handle is restored)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error", TeaCacheUncalibratedCheckpointWarning)
        try:
            handle = apply_teacache(flux, **kwargs)  # type: ignore[arg-type]
        except TeaCacheUncalibratedCheckpointWarning:
            return True
        handle.restore()
        return False


def test_the_fake_reaches_every_registered_variant() -> None:
    """Bug: a variant registers without a calibrated checkpoint listed here, so the tests below silently skip it."""
    assert sorted(_REGISTRY) == _VARIANTS


@pytest.mark.parametrize("variant_id", _VARIANTS)
def test_every_variant_warns_on_an_unknown_checkpoint(variant_id: str) -> None:
    """Bug: the check stays Qwen-only, so a FLUX, Klein or Z-Image finetune runs the built-in fit silently."""
    flux = _flux(variant_id, "someone/finetune")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error", TeaCacheUncalibratedCheckpointWarning)
        with pytest.raises(TeaCacheUncalibratedCheckpointWarning) as caught:
            apply_teacache(flux)
    message = str(caught.value)
    assert f"variant {variant_id!r}" in message
    assert "'someone/finetune'" in message
    assert repr(_CALIBRATED_ON[variant_id]) in message
    assert "mirror" in message
    assert getattr(flux, "_teacache_handle", None) is None


@pytest.mark.parametrize("variant_id", _VARIANTS)
@pytest.mark.parametrize("casing", [str.upper, str.lower, str])
def test_a_declared_base_model_counts_as_calibrated(variant_id: str, casing: object) -> None:
    """Bug: a mirror or finetune that declares the calibrated checkpoint as its base warns (or does so only when the
    casing matches exactly)."""
    base = casing(_CALIBRATED_ON[variant_id])  # type: ignore[operator]
    assert _apply_warns_uncalibrated(_flux(variant_id, "/models/some-mirror-q4", base)) is False


@pytest.mark.parametrize("variant_id", _VARIANTS)
def test_caller_coefficients_silence_the_check(variant_id: str) -> None:
    """Bug: the warning fires even when the caller calibrated the checkpoint and passed the result."""
    flux = _flux(variant_id, "someone/finetune")
    # filterwarnings = error: any warning here (the threshold is passed, so not the coefficients FutureWarning) fails.
    apply_teacache(flux, rel_l1_thresh=0.2, coefficients=_COEFFS).restore()


@pytest.mark.parametrize("variant_id", _VARIANTS)
def test_warning_points_at_the_caller_line(variant_id: str) -> None:
    """Bug: the stacklevel is wrong after the move, so the warning points into mlx_teacache instead of the call."""
    flux = _flux(variant_id, "someone/finetune")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        handle = apply_teacache(flux)
    handle.restore()
    ours = [w for w in record if issubclass(w.category, TeaCacheUncalibratedCheckpointWarning)]
    assert len(ours) == 1
    assert ours[0].filename == __file__


@pytest.mark.parametrize("variant_id", _VARIANTS)
@pytest.mark.parametrize(
    ("model_name", "base_model"),
    [("someone/finetune", None), ("UPPER", None), ("/models/mirror", "BASE")],
    ids=["unknown", "calibrated-upper-case", "mirror-with-base"],
)
def test_match_variant_calibrated_agrees_with_the_warning(
    variant_id: str, model_name: str, base_model: str | None
) -> None:
    """Bug: the plugin's pre-load answer (VariantInfo.calibrated) and the apply-time warning disagree."""
    calibrated_on = _CALIBRATED_ON[variant_id]
    model_name = calibrated_on.upper() if model_name == "UPPER" else model_name
    base_model = calibrated_on if base_model == "BASE" else base_model
    flux = _flux(variant_id, model_name, base_model)
    info = match_variant(flux.model_config, _Pipeline)  # type: ignore[attr-defined]
    assert info is not None and info.variant_id == variant_id
    assert info.calibrated is (not _apply_warns_uncalibrated(flux))
    assert info.calibrated is (model_name != "someone/finetune")


@pytest.mark.parametrize("variant_id", _VARIANTS)
def test_a_config_without_a_model_name_is_not_checked(variant_id: str) -> None:
    """Bug: a duck-typed model with aliases but no model_name is reported as an uncalibrated checkpoint."""
    flux = _flux(variant_id, None)
    assert _apply_warns_uncalibrated(flux) is False
    info = match_variant(flux.model_config, _Pipeline)  # type: ignore[attr-defined]
    assert info is not None and info.calibrated is True


def test_a_variant_without_hf_model_id_fails_the_import_naming_it() -> None:
    """Bug: hf_model_id stays optional, so a new variant without it registers and the shared check compares the
    config against nothing (or crashes with a KeyError at apply time)."""
    detect = types.SimpleNamespace(
        matches=lambda f: False, matches_config=lambda c, k: False, MODEL_NAMES=("p",)
    )
    meta = {"variant_id": "phantom", "display_name": "P", "license": "L", "non_distilled": True}
    with pytest.raises(CalibrationError, match="hf_model_id") as caught:
        _detect_contract(detect, meta, variant_id="phantom")
    assert caught.value.variant_id == "phantom"
