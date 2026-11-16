"""Coefficients without a threshold: one plain FutureWarning per apply, naming the threshold the handle uses.

A threshold is tuned for one polynomial, so a caller who brings their own coefficients and no rel_l1_thresh gets
the variant's resolved value (0.30 on flux1-krea-dev, 0.20 on every other variant) and a FutureWarning that 1.0
will require the threshold. The warning is raised after a variant matched and before anything is patched, so a
refusal under an error filter leaves the model as it was.

Pure-core: weight-free models from tests._variant_fakes; no weights, milliseconds. Expected thresholds are the
documented values (0.30 on flux1-krea-dev, 0.20 on every other variant) written out as literals, not read from the
registry or the integrations.
"""

import inspect
import warnings
from typing import Any

import pytest

from mlx_teacache import TeaCacheDisabledWarning, TeaCacheHandle, TeaCacheWarning, apply_teacache
from mlx_teacache.variants import _REGISTRY
from tests._variant_fakes import fake_for_variant
from tests.conftest import expect_distilled_warning

# Any five-term polynomial that is no variant's built-in one.
_CALLER_COEFFICIENTS = (1.0, 0.5, 0.25, 0.0, 0.0)

_VARIANT_IDS = sorted(_REGISTRY)

# The threshold a caller-coefficient apply resolves to, from the 0.17 rule (unchanged in 0.18).
_EXPECTED_THRESH = {
    "flux1-dev": 0.20,
    "flux1-schnell": 0.20,
    "flux1-krea-dev": 0.30,
    "flux2-klein-4b": 0.20,
    "flux2-klein-9b": 0.20,
    "flux2-klein-base-4b": 0.20,
    "flux2-klein-base-9b": 0.20,
    "z-image-base": 0.20,
    "qwen-image": 0.20,
    "qwen-image-2512": 0.20,
}


def _apply_recording(variant_id: str, **kwargs: Any) -> tuple[TeaCacheHandle, list[warnings.WarningMessage]]:
    """Apply on a fresh weight-free model with every warning recorded (no dedup), then restore."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        handle = apply_teacache(fake_for_variant(variant_id), **kwargs)
    handle.restore()
    return handle, [w for w in caught if issubclass(w.category, FutureWarning)]


def test_the_expected_table_covers_the_registry() -> None:
    assert sorted(_EXPECTED_THRESH) == _VARIANT_IDS


@pytest.mark.parametrize("variant_id", _VARIANT_IDS)
def test_caller_coefficients_without_a_threshold_warn_once(variant_id: str) -> None:
    """Bug caught: the warning is missing, or fires twice per apply (once in api.py and once in an integration)."""
    _, future = _apply_recording(variant_id, coefficients=_CALLER_COEFFICIENTS)
    assert [w.category for w in future] == [FutureWarning]


@pytest.mark.parametrize("variant_id", _VARIANT_IDS)
def test_warning_names_the_threshold_the_handle_uses(variant_id: str) -> None:
    """Bug caught: the registry's policy and an integration literal drift, so the warning names a value the gate
    does not use."""
    handle, future = _apply_recording(variant_id, coefficients=_CALLER_COEFFICIENTS)
    assert handle.rel_l1_thresh == pytest.approx(_EXPECTED_THRESH[variant_id], abs=0.0)
    assert len(future) == 1
    message = str(future[0].message)
    assert f"rel_l1_thresh={_EXPECTED_THRESH[variant_id]:.2f}" in message
    assert f"variant {variant_id!r}" in message
    assert "mlx-teacache 1.0 will raise an error here" in message


@pytest.mark.parametrize("variant_id", ["flux1-krea-dev", "flux1-dev"])
@pytest.mark.parametrize("rel_l1_thresh", [0.20, 0.30])
def test_an_explicit_threshold_does_not_warn(variant_id: str, rel_l1_thresh: float) -> None:
    """Bug caught: the warning keys on `coefficients` alone."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        handle = apply_teacache(
            fake_for_variant(variant_id), coefficients=_CALLER_COEFFICIENTS, rel_l1_thresh=rel_l1_thresh
        )
    handle.restore()
    assert handle.rel_l1_thresh == rel_l1_thresh


@pytest.mark.parametrize("variant_id", ["flux1-krea-dev", "flux1-dev"])
def test_an_explicit_zero_threshold_warns_only_that_caching_is_off(variant_id: str) -> None:
    """Bug caught: the warning keys on `coefficients` alone (a 0.0 threshold is a threshold)."""
    with pytest.warns(TeaCacheDisabledWarning), warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        handle = apply_teacache(
            fake_for_variant(variant_id), coefficients=_CALLER_COEFFICIENTS, rel_l1_thresh=0.0
        )
    handle.restore()
    assert handle.rel_l1_thresh == 0.0


@pytest.mark.parametrize("variant_id", _VARIANT_IDS)
def test_builtin_coefficients_do_not_warn(variant_id: str) -> None:
    """Bug caught: the warning keys on `rel_l1_thresh is None` alone."""
    with expect_distilled_warning(variant_id), warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        handle = apply_teacache(fake_for_variant(variant_id))
    handle.restore()
    assert handle.variant_id == variant_id


def test_a_refused_apply_leaves_the_model_untouched() -> None:
    """Bug caught: the warning is raised after the integration patched the model."""
    flux = fake_for_variant("flux1-dev")
    transformer = flux.transformer
    # The fake keeps generate_image as an instance attribute, so "not wrapped" means the same object as before.
    before = dict(vars(flux))
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        with pytest.raises(FutureWarning, match=r"rel_l1_thresh=0\.20"):
            apply_teacache(flux, coefficients=_CALLER_COEFFICIENTS)
    assert "_teacache_handle" not in vars(flux)
    assert flux.transformer is transformer
    assert vars(flux).keys() == before.keys()
    assert all(vars(flux)[name] is value for name, value in before.items())

    handle = apply_teacache(flux, coefficients=_CALLER_COEFFICIENTS, rel_l1_thresh=0.2)
    assert handle.rel_l1_thresh == 0.2
    handle.restore()


def test_warning_points_at_the_published_sweep_scripts() -> None:
    """Bug caught: the message sends an installed user to a scripts/ folder that the wheel does not contain."""
    _, future = _apply_recording("flux1-dev", coefficients=_CALLER_COEFFICIENTS)
    assert len(future) == 1
    message = str(future[0].message)
    assert "https://github.com/IonDen/mlx-teacache/tree/main/scripts" in message
    assert "scripts/ folder" not in message


def test_warning_points_at_the_caller_line() -> None:
    """Bug caught: wrong stacklevel. One frame too low names a line inside mlx_teacache; one frame too high names
    pytest's own frame that called this test, so the filename and the line of the apply_teacache call below both
    have to match."""
    flux = fake_for_variant("flux1-dev")
    frame = inspect.currentframe()
    assert frame is not None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call_line = frame.f_lineno + 1
        handle = apply_teacache(flux, coefficients=_CALLER_COEFFICIENTS)
    handle.restore()
    future = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert len(future) == 1
    assert future[0].filename == __file__
    assert future[0].lineno == call_line


def test_warning_is_a_plain_future_warning() -> None:
    """Bug caught: the class changes so the TeaCacheWarning family filter silences it, or Python hides it by
    default as a DeprecationWarning."""
    _, future = _apply_recording("qwen-image", coefficients=_CALLER_COEFFICIENTS)
    assert len(future) == 1
    category = future[0].category
    assert category is FutureWarning
    assert not issubclass(category, TeaCacheWarning)
    assert not issubclass(category, DeprecationWarning)
