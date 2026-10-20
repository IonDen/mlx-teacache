"""Detection refuses mflux pipeline classes whose call contract TeaCache does not implement."""

import types

import pytest

from mlx_teacache import IncompatibleModelError, apply_teacache
from mlx_teacache.variants import _REGISTRY


def _mflux_class(name: str) -> type:
    # A class that *claims* to live in mflux, exactly as the real pipelines do.
    return type(name, (), {"__module__": f"mflux.models.fake.{name.lower()}"})


_ALIASES_2512 = ["qwen-image", "qwen", "qwen-image-2512", "qwen-2512"]  # mflux 0.19+ qwen-image config


def _instance(cls: type, alias: str | list[str], model_name: str | None = None) -> object:
    obj = cls()
    aliases = [alias] if isinstance(alias, str) else alias
    obj.model_config = types.SimpleNamespace(aliases=aliases, model_name=model_name)
    return obj


@pytest.mark.parametrize(
    ("cls_name", "alias", "model_name"),
    [
        (
            "Flux2KleinEdit",
            "flux2-klein-base-4b",
            None,
        ),  # bug: Edit calls predict(image_latents=...) -> TypeError step 0
        ("Flux1Concept", "schnell", None),  # bug: unpacks (noise, attention) from a one-array proxy
        ("Flux1ConceptFromImage", "schnell", None),
        ("Flux1InContextDev", "dev", None),  # bug: silently runs at an unverified operating point
        ("Flux1InContextFill", "dev", None),
        ("Flux1Redux", "dev", None),
        ("Flux1Controlnet", "dev", None),
        ("QwenImageEdit", "qwen-image", None),
        ("QwenImageEdit", _ALIASES_2512, "Qwen/Qwen-Image-2512"),  # bug: the 2512 allowlist lets Edit through
    ],
)
def test_unsupported_mflux_pipeline_is_refused_before_patching(
    cls_name: str, alias: str | list[str], model_name: str | None
) -> None:
    """Bug caught: detect keyed on aliases only lets these classes through."""
    flux = _instance(_mflux_class(cls_name), alias, model_name)
    with pytest.raises(IncompatibleModelError, match=cls_name):
        apply_teacache(flux)
    assert not hasattr(flux, "_teacache_handle")


@pytest.mark.parametrize(
    ("cls_name", "alias", "model_name", "variant_id"),
    [
        ("Flux1", "dev", None, "flux1-dev"),
        ("Flux1", "krea-dev", None, "flux1-krea-dev"),
        ("Flux1", "schnell", None, "flux1-schnell"),
        ("Flux2Klein", "flux2-klein-4b", None, "flux2-klein-4b"),
        ("Flux2Klein", "flux2-klein-9b", None, "flux2-klein-9b"),
        ("Flux2Klein", "flux2-klein-base-4b", None, "flux2-klein-base-4b"),
        ("Flux2Klein", "flux2-klein-base-9b", None, "flux2-klein-base-9b"),
        ("ZImage", "z-image", None, "z-image-base"),
        ("QwenImage", "qwen-image", None, "qwen-image"),
        ("QwenImage", _ALIASES_2512, "Qwen/Qwen-Image-2512", "qwen-image-2512"),
    ],
)
def test_supported_mflux_pipeline_still_matches(
    cls_name: str, alias: str | list[str], model_name: str | None, variant_id: str
) -> None:
    """Bug caught: an allowlist typo refuses a supported model."""
    flux = _instance(_mflux_class(cls_name), alias, model_name)
    assert _REGISTRY[variant_id]["matches"](flux)


def test_non_mflux_subclass_is_duck_typed_by_alias() -> None:
    """Bug caught: the guard refuses user-defined wrappers that don't live in mflux."""
    user_cls = type("MyFluxWrapper", (), {"__module__": "my_project.models"})
    flux = _instance(user_cls, "dev")
    assert _REGISTRY["flux1-dev"]["matches"](flux)
