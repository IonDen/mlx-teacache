"""Detection refuses mflux pipeline classes whose call contract TeaCache does not implement."""

import types

import pytest

from mlx_teacache import IncompatibleModelError, apply_teacache
from mlx_teacache.variants import _REGISTRY


def _mflux_class(name: str) -> type:
    # A class that *claims* to live in mflux, exactly as the real pipelines do.
    return type(name, (), {"__module__": f"mflux.models.fake.{name.lower()}"})


def _instance(cls: type, alias: str) -> object:
    obj = cls()
    obj.model_config = types.SimpleNamespace(aliases=[alias], model_name=None)
    return obj


@pytest.mark.parametrize(
    ("cls_name", "alias"),
    [
        (
            "Flux2KleinEdit",
            "flux2-klein-base-4b",
        ),  # bug: Edit calls predict(image_latents=...) -> TypeError step 0
        ("Flux1Concept", "schnell"),  # bug: unpacks (noise, attention) from a one-array proxy
        ("Flux1ConceptFromImage", "schnell"),
        ("Flux1InContextDev", "dev"),  # bug: silently runs at an unverified operating point
        ("Flux1InContextFill", "dev"),
        ("Flux1Redux", "dev"),
        ("Flux1Controlnet", "dev"),
        ("QwenImageEdit", "qwen-image"),
    ],
)
def test_unsupported_mflux_pipeline_is_refused_before_patching(cls_name: str, alias: str) -> None:
    """Bug caught: detect keyed on aliases only lets these classes through."""
    flux = _instance(_mflux_class(cls_name), alias)
    with pytest.raises(IncompatibleModelError, match=cls_name):
        apply_teacache(flux)
    assert not hasattr(flux, "_teacache_handle")


@pytest.mark.parametrize(
    ("cls_name", "alias", "variant_id"),
    [
        ("Flux1", "dev", "flux1-dev"),
        ("Flux1", "krea-dev", "flux1-krea-dev"),
        ("Flux1", "schnell", "flux1-schnell"),
        ("Flux2Klein", "flux2-klein-base-4b", "flux2-klein-base-4b"),
        ("Flux2Klein", "flux2-klein-9b", "flux2-klein-9b"),
        ("ZImage", "z-image", "z-image-base"),
        ("QwenImage", "qwen-image", "qwen-image"),
    ],
)
def test_supported_mflux_pipeline_still_matches(cls_name: str, alias: str, variant_id: str) -> None:
    """Bug caught: an allowlist typo refuses a supported model."""
    flux = _instance(_mflux_class(cls_name), alias)
    assert _REGISTRY[variant_id]["matches"](flux)


def test_non_mflux_subclass_is_duck_typed_by_alias() -> None:
    """Bug caught: the guard refuses user-defined wrappers that don't live in mflux."""
    user_cls = type("MyFluxWrapper", (), {"__module__": "my_project.models"})
    flux = _instance(user_cls, "dev")
    assert _REGISTRY["flux1-dev"]["matches"](flux)
