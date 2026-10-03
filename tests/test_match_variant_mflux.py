"""match_variant against the installed mflux: real ModelConfigs from the names users type, the real pipeline classes,
and apply_teacache on fakes of every variant. No weights. In the mflux lane through tests/_lanes.py."""

import importlib
from importlib.metadata import version

import pytest

from mlx_teacache import apply_teacache, match_variant
from mlx_teacache._mflux_versions import release_tuple
from mlx_teacache.variants import _REGISTRY
from tests._variant_fakes import fake_for_variant
from tests.conftest import expect_distilled_warning

_PIPELINE = {
    "flux1": ("mflux.models.flux.variants.txt2img.flux", "Flux1"),
    "flux2": ("mflux.models.flux2.variants.txt2img.flux2_klein", "Flux2Klein"),
    "qwen": ("mflux.models.qwen.variants.txt2img.qwen_image", "QwenImage"),
    "z_image": ("mflux.models.z_image.variants.z_image", "ZImage"),
}
_FAMILY = {
    "flux1-dev": "flux1",
    "flux1-krea-dev": "flux1",
    "flux1-schnell": "flux1",
    "flux2-klein-4b": "flux2",
    "flux2-klein-9b": "flux2",
    "flux2-klein-base-4b": "flux2",
    "flux2-klein-base-9b": "flux2",
    "qwen-image": "qwen",
    "z-image-base": "z_image",
}


def _pipeline(family: str) -> type:
    module, name = _PIPELINE[family]
    return getattr(importlib.import_module(module), name)


def _config(name: str) -> object:
    from mflux.models.common.config.model_config import ModelConfig

    return ModelConfig.from_name(name)


@pytest.mark.parametrize("variant_id", sorted(_REGISTRY))
def test_same_answer_as_apply_teacache_on_every_variant(variant_id: str) -> None:
    """Bug: the pre-load answer and the variant apply_teacache actually patches differ for some variant."""
    fake = fake_for_variant(variant_id)
    with expect_distilled_warning(variant_id):
        handle = apply_teacache(fake)
    try:
        info = match_variant(fake.model_config, type(fake))
        assert info is not None and info.variant_id == handle.variant_id == variant_id
    finally:
        handle.restore()


@pytest.mark.parametrize(
    ("variant_id", "name"),
    [(vid, name) for vid in sorted(_REGISTRY) for name in _REGISTRY[vid]["model_names"]],
)
def test_every_listed_model_name_resolves_to_its_variant(variant_id: str, name: str) -> None:
    """Bug: MODEL_NAMES lists a name mflux does not accept, or one that resolves to a config the detector refuses, so
    a launcher tells users to type a name that fails."""
    info = match_variant(_config(name), _pipeline(_FAMILY[variant_id]))
    assert info is not None and info.variant_id == variant_id


@pytest.mark.parametrize(
    ("name", "family", "variant_id"),
    [
        ("klein-base-4b", "flux2", "flux2-klein-base-4b"),
        ("flux2-base-9b", "flux2", "flux2-klein-base-9b"),
        ("klein-4b", "flux2", "flux2-klein-4b"),
        ("dev-krea", "flux1", "flux1-krea-dev"),
        ("z-image-turbo", "z_image", None),
        ("dev-kontext", "flux1", None),
        ("qwen-image-edit", "qwen", None),
    ],
)
def test_other_mflux_aliases_resolve_like_the_listed_ones(
    name: str, family: str, variant_id: str | None
) -> None:
    """Bug: a user who types another accepted alias (klein-base-4b, dev-krea) is told the model is unsupported, or a
    turbo / kontext / edit model is reported as supported."""
    info = match_variant(_config(name), _pipeline(family))
    assert (None if info is None else info.variant_id) == variant_id


def test_qwen_calibrated_follows_the_checkpoint_mflux_resolves() -> None:
    """Bug: calibrated ignores which checkpoint the installed mflux resolves, so on mflux 0.19+ (where the qwen-image
    alias is Qwen/Qwen-Image-2512) a launcher is told the alias is calibrated and loads 2512 without a warning, or the
    original Qwen/Qwen-Image is reported as uncalibrated."""
    alias_is_2512 = release_tuple(version("mflux")) >= (0, 19, 0)
    by_alias = match_variant(_config("qwen-image"), _pipeline("qwen"))
    by_repo = match_variant(_config("Qwen/Qwen-Image"), _pipeline("qwen"))
    assert by_alias is not None and by_repo is not None
    assert (by_alias.variant_id, by_repo.variant_id) == ("qwen-image", "qwen-image")
    assert by_alias.calibrated is (not alias_is_2512)
    assert by_repo.calibrated is True
