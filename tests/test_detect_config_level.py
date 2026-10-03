"""Each variant's detect.matches(flux) gives the same answer as its config-level
matches_config(model_config, pipeline_class), so a check made before loading (from the config and the class) and
apply_teacache's check on the loaded model cannot disagree."""

import importlib
import itertools
import types

import pytest

_FAMILY_CLASS = {
    "flux1_dev": "Flux1",
    "flux1_krea_dev": "Flux1",
    "flux1_schnell": "Flux1",
    "flux2_klein_4b": "Flux2Klein",
    "flux2_klein_9b": "Flux2Klein",
    "flux2_klein_base_4b": "Flux2Klein",
    "flux2_klein_base_9b": "Flux2Klein",
    "qwen_image": "QwenImage",
    "z_image_base": "ZImage",
}


def _mflux_class(name: str) -> type:
    return type(name, (), {"__module__": f"mflux.models.fake.{name.lower()}"})


_CLASSES = [
    _mflux_class(name)
    for name in (
        "Flux1",
        "Flux2Klein",
        "ZImage",
        "QwenImage",
        "Flux1Concept",
        "Flux2KleinEdit",
        "QwenImageEdit",
    )
] + [type("MyWrapper", (), {"__module__": "my_project.models"})]

_ALIAS_LISTS = [
    ["dev"],
    ["krea-dev", "dev-krea"],
    ["schnell"],
    ["flux2-klein-4b"],
    ["flux2-klein-9b"],
    ["flux2-klein-base-4b"],
    ["flux2-klein-base-9b"],
    ["qwen-image", "qwen"],
    ["z-image", "zimage"],
    ["z-image-turbo", "zimage-turbo"],
    ["dev-kontext"],
    [],
    None,
]


@pytest.mark.parametrize("subname", sorted(_FAMILY_CLASS))
def test_matches_and_matches_config_agree_everywhere(subname: str) -> None:
    """Bug: a detect.py keeps its own copy of the rule in matches() (for example without the pipeline-class guard),
    so a pre-load check says yes where apply_teacache says no."""
    detect = importlib.import_module(f"mlx_teacache.variants.{subname}.detect")
    disagreements = []
    matched = 0
    for cls, aliases in itertools.product(_CLASSES, _ALIAS_LISTS):
        flux = cls()
        flux.model_config = (
            None if aliases is None else types.SimpleNamespace(aliases=aliases, model_name=None)
        )
        post_load = detect.matches(flux)
        pre_load = detect.matches_config(flux.model_config, cls)
        matched += post_load
        if post_load != pre_load:
            disagreements.append((cls.__name__, aliases, post_load, pre_load))
    assert disagreements == []
    assert matched > 0


@pytest.mark.parametrize("subname", sorted(_FAMILY_CLASS))
def test_every_model_name_is_detected_on_its_own(subname: str) -> None:
    """Bug: MODEL_NAMES lists a name the detector does not key on, so users are told to type a name that is then
    refused."""
    detect = importlib.import_module(f"mlx_teacache.variants.{subname}.detect")
    cls = _mflux_class(_FAMILY_CLASS[subname])
    refused = [
        n for n in detect.MODEL_NAMES if not detect.matches_config(types.SimpleNamespace(aliases=[n]), cls)
    ]
    assert detect.MODEL_NAMES and refused == []
