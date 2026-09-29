"""Apply, restore and provenance on every registered variant (fake pipelines, no weights)."""

import contextlib
from typing import Any

import pytest

from mlx_teacache import apply_teacache
from mlx_teacache.errors import TeaCacheNoBenefitWarning
from mlx_teacache.variants import _REGISTRY
from tests._variant_fakes import fake_for_variant

pytestmark = pytest.mark.mflux

_DISTILLED = {"flux2-klein-4b", "flux2-klein-9b"}
_PREDICT_FAMILIES = {
    "flux2-klein-4b",
    "flux2-klein-9b",
    "flux2-klein-base-4b",
    "flux2-klein-base-9b",
    "z-image-base",
}
_CUSTOM = (0.5, 1.5, 2.5, 3.5, 4.5)
_IDS = sorted(_REGISTRY)


def _apply(variant_id: str, flux: Any, **kw: Any) -> Any:
    """apply_teacache, expecting the distilled no-benefit warning on a builtin-coefficient apply."""
    if variant_id in _DISTILLED and "coefficients" not in kw:
        ctx: Any = pytest.warns(TeaCacheNoBenefitWarning)
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        return apply_teacache(flux, **kw)


def test_matrix_covers_all_nine_registered_variants() -> None:
    """Bug: a variant added to the registry is silently left out of the lifecycle matrix."""
    assert _IDS == [
        "flux1-dev",
        "flux1-krea-dev",
        "flux1-schnell",
        "flux2-klein-4b",
        "flux2-klein-9b",
        "flux2-klein-base-4b",
        "flux2-klein-base-9b",
        "qwen-image",
        "z-image-base",
    ]


@pytest.mark.parametrize("variant_id", _IDS)
def test_builtin_apply_records_builtin_provenance(variant_id: str) -> None:
    """Bug: a variant reports the wrong provenance source when the caller passes no coefficients."""
    flux = fake_for_variant(variant_id)
    handle = _apply(variant_id, flux)
    try:
        assert handle.variant_id == variant_id
        assert handle.provenance.source == "builtin"
    finally:
        handle.restore()


@pytest.mark.parametrize("variant_id", _IDS)
def test_custom_coefficients_record_user_provenance(variant_id: str) -> None:
    """Bug: caller coefficients are labelled builtin (or replaced) in the handle's provenance."""
    flux = fake_for_variant(variant_id)
    handle = _apply(variant_id, flux, coefficients=_CUSTOM)
    try:
        assert handle.provenance.source == "user"
        assert handle.coefficients == (0.5, 1.5, 2.5, 3.5, 4.5)
    finally:
        handle.restore()


@pytest.mark.parametrize("variant_id", _IDS)
def test_restore_undoes_every_patch(variant_id: str) -> None:
    """Bug: restore() leaves the proxy, _predict, wrapped generate_image, callback or sentinel behind."""
    flux = fake_for_variant(variant_id)
    original_transformer = flux.transformer
    original_generate = flux.generate_image
    handle = _apply(variant_id, flux)

    # The patch is really installed, so the restore assertions below are not vacuous.
    assert flux.generate_image is not original_generate
    assert getattr(flux, "_teacache_handle", None) is handle
    if variant_id in _PREDICT_FAMILIES:
        assert "_predict" in vars(flux)
    else:
        assert flux.transformer is not original_transformer
    assert any(len(lst) == 1 for lst in (flux.callbacks.before_loop, flux.callbacks.after_loop))

    handle.restore()

    assert flux.transformer is original_transformer
    assert "_predict" not in vars(flux)
    assert flux.generate_image is original_generate
    for hook_list in (
        flux.callbacks.before_loop,
        flux.callbacks.in_loop,
        flux.callbacks.after_loop,
        flux.callbacks.interrupt,
    ):
        assert hook_list == []
    assert not hasattr(flux, "_teacache_handle")
