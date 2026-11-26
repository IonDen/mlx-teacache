"""A model built with mflux's `ssd_stream=True` (`--ssd-stream`) is refused at apply time, for every variant.

mflux's SSD streamer swaps each transformer block for a stand-in whose weights are loaded around each call; TeaCache
has not been verified with it. No weights: the duck-typed fakes of `tests._variant_fakes`, which import no mflux."""

import pickle
import warnings
from typing import Any

import pytest

from mlx_teacache import apply_teacache
from mlx_teacache.errors import IncompatibleModelError
from tests._variant_fakes import fake_for_variant

_LISTS = ("before_loop", "in_loop", "after_loop", "interrupt")

# One predict-strategy variant per family plus one proxy-strategy variant: the check must not live in one apply().
_VARIANTS = ["z-image-base", "flux2-klein-base-4b", "qwen-image"]
# Plus the distilled flux2-klein-4b, whose apply warns TeaCacheNoBenefitWarning: without a variant that warns, a
# refusal moved below the variant-level warnings would still pass under the error filter.
_REFUSED_VARIANTS = [*_VARIANTS, "flux2-klein-4b"]


def _snapshot(flux: Any) -> dict[str, Any]:
    return {
        "attributes": dict(vars(flux)),
        "callbacks": {name: list(getattr(flux.callbacks, name)) for name in _LISTS},
    }


@pytest.mark.parametrize("variant_id", _REFUSED_VARIANTS)
def test_a_model_built_with_ssd_stream_is_refused_and_left_untouched(variant_id: str) -> None:
    """Bug: the refusal is missing (TeaCache patches a streamed model it was never verified on), lives in only one
    variant's apply(), or runs after the patch has started mutating the model. For flux2-klein-4b, bug: the refusal
    sits below the variant-level warnings, so a caller is warned about a model TeaCache then refuses (and under an
    error filter gets TeaCacheNoBenefitWarning instead of the refusal)."""
    flux = fake_for_variant(variant_id)
    flux.ssd_stream = True
    before = _snapshot(flux)
    original_generate = flux.generate_image

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the refusal comes before any variant-level warning
        with pytest.raises(IncompatibleModelError) as excinfo:
            apply_teacache(flux)

    message = str(excinfo.value)
    assert "does not support SSD streaming yet" in message
    assert "ssd_stream=True" in message
    assert "--ssd-stream" in message
    assert "without" in message
    assert _snapshot(flux) == before
    assert "_predict" not in vars(flux)
    assert flux.generate_image is original_generate
    assert not hasattr(flux, "_teacache_handle")


@pytest.mark.parametrize("variant_id", _VARIANTS)
def test_a_model_built_without_ssd_stream_is_patched(variant_id: str) -> None:
    """Bug: the check reads the attribute's presence instead of its value, so mflux main's default
    ssd_stream=False is refused too."""
    flux = fake_for_variant(variant_id)
    flux.ssd_stream = False
    handle = apply_teacache(flux)
    try:
        assert flux._teacache_handle is handle
    finally:
        handle.restore()


def test_a_model_from_an_mflux_without_the_attribute_is_patched() -> None:
    """Bug: the check reads `flux.ssd_stream` directly, so on mflux 0.24 (no such attribute) every apply raises
    AttributeError."""
    flux = fake_for_variant("z-image-base")
    assert not hasattr(flux, "ssd_stream")
    handle = apply_teacache(flux)
    try:
        assert "_predict" in vars(flux)
    finally:
        handle.restore()


def test_the_refusal_survives_a_pickle_round_trip() -> None:
    """Bug: the constructor and `__reduce__` stop agreeing: a keyword the constructor requires is left out of the
    rebuild, so the refusal raised in a worker process fails to load in the parent with a TypeError."""
    flux = fake_for_variant("z-image-base")
    flux.ssd_stream = True
    with pytest.raises(IncompatibleModelError) as excinfo:
        apply_teacache(flux)
    error = excinfo.value
    error.add_note("seen in a worker")
    loaded = pickle.loads(pickle.dumps(error))
    assert type(loaded) is type(error)
    assert str(loaded) == str(error)
    assert loaded.supported == ["z-image-base"]
    assert loaded.__notes__ == ["seen in a worker"]
