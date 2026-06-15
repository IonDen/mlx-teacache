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


def test_missing_model_config_is_false() -> None:
    assert matches(SimpleNamespace()) is False
