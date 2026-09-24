"""Qwen-Image detect: accept the base aliases, reject every edit alias. Pure-core.

Aliases are the runtime hyphenated strings from mflux ModelConfig
(model_config.py:429-447), NOT the Python factory names.
"""

from types import SimpleNamespace

import pytest

from mlx_teacache.variants.qwen_image.detect import matches


def _flux(aliases: list[str]):
    return SimpleNamespace(model_config=SimpleNamespace(aliases=aliases))


@pytest.mark.parametrize("aliases", [["qwen-image", "qwen"], ["qwen"], ["qwen-image"]])
def test_matches_base(aliases: list[str]) -> None:
    assert matches(_flux(aliases)) is True


@pytest.mark.parametrize(
    "aliases",
    [
        ["qwen-image-edit", "qwen-edit", "qwen-edit-plus", "qwen-edit-2509"],
        ["qwen-image-edit"],
        ["qwen-edit"],
        ["qwen-edit-plus"],
        ["qwen-edit-2509"],
        ["z-image", "zimage"],
        [],
    ],
)
def test_rejects_edit_and_others(aliases: list[str]) -> None:
    assert matches(_flux(aliases)) is False


def test_rejects_qwen_image_21() -> None:
    """mflux 0.20 adds Qwen-Image-2.1 as its own architecture (`mflux.models.qwen21`), with
    aliases that all start with the Qwen-Image base's own names. Applying the Qwen-Image proxy
    to it would re-walk a transformer this variant was never written for. Goes red if the
    detector ever switches from element membership to substring or prefix matching."""
    assert matches(_flux(["qwen-image-2.1", "qwen-2.1", "qwen-image-21"])) is False


def test_missing_model_config_is_false() -> None:
    assert matches(SimpleNamespace()) is False
