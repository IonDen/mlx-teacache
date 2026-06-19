"""Weight-free real-mflux contract smoke.

The fast mflux lane otherwise runs against FAKE registries/transformers, so a
real-mflux structural change (a CallbackRegistry attribute rename, a model_config
alias retype, a ModelConfig.precision default flip) would pass undetected. These
assertions touch the REAL installed mflux with no weights, so they run on the
floor-pinned CI job and the ceiling lane and catch those touchpoints on every PR.

mflux-marked (added to conftest._MFLUX_FILES) — excluded from the pure-core lane.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_teacache.variants.flux1_dev import detect as flux1_detect
from mlx_teacache.variants.flux2_klein_base_4b import detect as flux2_detect
from mlx_teacache.variants.qwen_image import detect as qwen_detect
from mlx_teacache.variants.z_image_base import detect as zimage_detect


def test_callback_registry_exposes_list_attributes() -> None:
    """_remove_callback_by_identity walks these four lists; a rename/retype breaks restore()."""
    from mflux.callbacks.callback_registry import CallbackRegistry

    reg = CallbackRegistry()
    for attr in ("before_loop", "in_loop", "after_loop", "interrupt"):
        assert isinstance(getattr(reg, attr), list), f"CallbackRegistry.{attr} is not a list"


def test_modelconfig_precision_is_bfloat16() -> None:
    """The FLUX.2 forwards hard-cast temb to ModelConfig.precision; a default flip
    silently shifts compute precision (transparently passed through, but the audit
    must know if it changed)."""
    from mflux.models.common.config.model_config import ModelConfig

    assert ModelConfig.precision == mx.bfloat16


@pytest.mark.parametrize(
    "matches, factory_name",
    [
        (flux1_detect.matches, "dev"),
        (flux2_detect.matches, "flux2_klein_base_4b"),
        (qwen_detect.matches, "qwen_image"),
        (zimage_detect.matches, "z_image"),
    ],
)
def test_detect_matches_real_modelconfig(matches, factory_name: str) -> None:
    """Each variant's detect.matches() must return True against the REAL mflux
    ModelConfig it targets — catches a 0.18 alias rename/retype that import smoke
    (factory-resolves) does not exercise."""
    from mflux.models.common.config.model_config import ModelConfig

    model_config = getattr(ModelConfig, factory_name)()
    flux_stub = SimpleNamespace(model_config=model_config)
    assert matches(flux_stub) is True, f"{factory_name}: detect.matches() returned False on real ModelConfig"
