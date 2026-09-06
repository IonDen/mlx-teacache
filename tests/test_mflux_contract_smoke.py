"""Weight-free real-mflux contract smoke.

The fast mflux lane otherwise runs against FAKE registries/transformers, so a
real-mflux structural change (a CallbackRegistry attribute rename, a model_config
alias retype, a ModelConfig.precision default flip) would pass undetected. These
assertions touch the REAL installed mflux with no weights, so they run on the
floor-pinned CI job and the ceiling lane and catch those touchpoints on every PR.

mflux-marked (added to conftest._MFLUX_FILES) — excluded from the pure-core lane.
"""

import importlib
import inspect
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_teacache.variants.flux1_dev import detect as flux1_detect
from mlx_teacache.variants.flux1_krea_dev import detect as krea_detect
from mlx_teacache.variants.flux1_schnell import detect as flux1_schnell_detect
from mlx_teacache.variants.flux2_klein_4b import detect as flux2_klein_4b_detect
from mlx_teacache.variants.flux2_klein_9b import detect as flux2_klein_9b_detect
from mlx_teacache.variants.flux2_klein_base_4b import detect as flux2_klein_base_4b_detect
from mlx_teacache.variants.flux2_klein_base_9b import detect as flux2_klein_base_9b_detect
from mlx_teacache.variants.qwen_image import detect as qwen_detect
from mlx_teacache.variants.z_image_base import detect as zimage_detect
from tests._mflux_surface import assigned_attributes, return_tuple_arities


def test_callback_registry_exposes_list_attributes() -> None:
    """These four are the PRIMARY callback lists _remove_callback_by_identity walks during restore(). It also has a suffixed-name fallback (before_loop_callbacks, etc.), so a bare-name rename in mflux would red THIS test as a heads-up even if production's fallback still carries it — treat a failure as 'go re-read _remove_callback_by_identity', not necessarily a hard break."""
    from mflux.callbacks.callback_registry import CallbackRegistry

    reg = CallbackRegistry()
    for attr in ("before_loop", "in_loop", "after_loop", "interrupt"):
        assert isinstance(getattr(reg, attr), list), f"CallbackRegistry.{attr} is not a list"


def test_modelconfig_precision_is_bfloat16() -> None:
    """The FLUX.2 forwards hard-cast temb to ModelConfig.precision; a default flip
    silently shifts compute precision (transparently passed through, but the audit
    must know if it changed)."""
    # Consumed by variants/flux2_klein_base_4b/integration.py: temb.astype(ModelConfig.precision) (3 sites).
    from mflux.models.common.config.model_config import ModelConfig

    assert ModelConfig.precision == mx.bfloat16


@pytest.mark.parametrize(
    "matches, factory_name",
    [
        (flux1_detect.matches, "dev"),
        (flux1_schnell_detect.matches, "schnell"),
        (krea_detect.matches, "krea_dev"),
        (flux2_klein_base_4b_detect.matches, "flux2_klein_base_4b"),
        (flux2_klein_4b_detect.matches, "flux2_klein_4b"),
        (flux2_klein_9b_detect.matches, "flux2_klein_9b"),
        (flux2_klein_base_9b_detect.matches, "flux2_klein_base_9b"),
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


# ---------------------------------------------------------------------------
# v0.11.0: the private surface the integrations touch, pinned weight-free.
# Every name below is one the forwards read on a live transformer; mflux has no
# obligation to keep any of them, so a rename must turn CI red here instead of
# breaking a generation. Attribute names come from the class's own __init__ (by
# AST), methods from hasattr on the class; neither needs weights.
# ---------------------------------------------------------------------------


_TRANSFORMER_SURFACE = [
    (
        "mflux.models.flux.model.flux_transformer.transformer",
        "Transformer",
        {
            "x_embedder",
            "context_embedder",
            "time_text_embed",
            "pos_embed",
            "transformer_blocks",
            "single_transformer_blocks",
            "norm_out",
            "proj_out",
        },
        {
            "compute_text_embeddings",
            "compute_rotary_embeddings",
            "_apply_joint_transformer_block",
            "_apply_single_transformer_block",
        },
        "variants/flux1_dev/integration.py",
    ),
    (
        "mflux.models.flux2.model.flux2_transformer.transformer",
        "Flux2Transformer",
        {
            "x_embedder",
            "context_embedder",
            "pos_embed",
            "time_guidance_embed",
            "double_stream_modulation_img",
            "double_stream_modulation_txt",
            "single_stream_modulation",
            "transformer_blocks",
            "single_transformer_blocks",
            "norm_out",
            "proj_out",
        },
        set(),
        "variants/flux2_klein_base_4b/integration.py",
    ),
    (
        "mflux.models.z_image.model.z_image_transformer.transformer",
        "ZImageTransformer",
        {
            "layers",
            "patch_size",
            "f_patch_size",
            "rope_embedder",
            "cap_embedder",
            "x_pad_token",
            "cap_pad_token",
            "t_scale",
            "t_embedder",
            "out_channels",
            "noise_refiner",
            "context_refiner",
            "all_x_embedder",
            "all_final_layer",
        },
        {"_patchify", "_unpatchify"},
        "variants/z_image_base/integration.py",
    ),
    (
        "mflux.models.qwen.model.qwen_transformer.qwen_transformer",
        "QwenTransformer",
        {
            "img_in",
            "txt_in",
            "txt_norm",
            "time_text_embed",
            "pos_embed",
            "transformer_blocks",
            "norm_out",
            "proj_out",
        },
        {"_compute_timestep", "_compute_rotary_embeddings", "_apply_transformer_block"},
        "variants/qwen_image/integration.py",
    ),
]


@pytest.mark.parametrize("module, cls_name, attrs, methods, consumer", _TRANSFORMER_SURFACE)
def test_transformer_surface_the_integration_touches_still_exists(
    module: str, cls_name: str, attrs: set[str], methods: set[str], consumer: str
) -> None:
    cls = getattr(importlib.import_module(module), cls_name)
    missing_attrs = sorted(attrs - assigned_attributes(cls))
    missing_methods = sorted(m for m in methods if not hasattr(cls, m))
    assert not missing_attrs and not missing_methods, (
        f"{cls_name}: attributes {missing_attrs} / methods {missing_methods} are gone in the installed "
        f"mflux but {consumer} still reads them"
    )


def test_flux1_block0_norm1_returns_a_five_tuple() -> None:
    """_flux1_extract_mod_input takes [0] of block_0.norm1(...): (norm_hidden_states,
    gate_msa, shift_mlp, scale_mlp, gate_mlp)."""
    from mflux.models.flux.model.flux_transformer.ada_layer_norm_zero import AdaLayerNormZero
    from mflux.models.flux.model.flux_transformer.joint_transformer_block import JointTransformerBlock

    assert "norm1" in assigned_attributes(JointTransformerBlock)
    assert 5 in return_tuple_arities(AdaLayerNormZero.__call__)


def test_qwen_block0_modulation_surface_still_exists() -> None:
    """_qwen_signal_a reads block0.img_mod_linear / img_mod_silu / img_norm1 and calls
    block0._modulate, which returns (modulated, gate)."""
    from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock

    assert {"img_mod_linear", "img_mod_silu", "img_norm1"} <= assigned_attributes(QwenTransformerBlock)
    assert hasattr(QwenTransformerBlock, "_modulate")
    assert 2 in return_tuple_arities(QwenTransformerBlock._modulate)


@pytest.mark.parametrize(
    "module, cls_name",
    [
        ("mflux.models.flux2.variants.txt2img.flux2_klein", "Flux2Klein"),
        ("mflux.models.z_image.variants.z_image", "ZImage"),
    ],
)
def test_predict_is_a_staticmethod_the_eager_closure_can_shadow(module: str, cls_name: str) -> None:
    """The FLUX.2 / Z-Image patch strategy assigns an instance attribute `_predict`
    over a class-level staticmethod called as `self._predict(self.transformer)`."""
    cls = getattr(importlib.import_module(module), cls_name)
    assert isinstance(inspect.getattr_static(cls, "_predict"), staticmethod)
    assert "self._predict(self.transformer)" in inspect.getsource(cls.generate_image)


def test_qwen_loop_calls_the_transformer_exactly_twice_per_step() -> None:
    """CfgBranchPairer shares one gate decision across a positive call and a negative
    call; an mflux loop that made the second call conditional would desync it."""
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

    assert inspect.getsource(QwenImage.generate_image).count("self.transformer(") == 2


def test_no_kv_cache_model_config_matches_any_variant() -> None:
    """flux2-klein-9b-kv (mflux 0.18) fills a KV cache on the very steps the gate
    would skip; it must stay unmatched until that interaction is analysed."""
    from mflux.models.common.config.model_config import AVAILABLE_MODELS

    from mlx_teacache.variants import _REGISTRY

    kv_configs = {k: v for k, v in AVAILABLE_MODELS.items() if "kv" in k.lower()}
    for key, cfg in kv_configs.items():
        stub = SimpleNamespace(model_config=cfg)
        matched = [vid for vid, entry in _REGISTRY.items() if entry["matches"](stub)]
        assert matched == [], f"{key} matched {matched}"


def test_every_real_model_config_matches_at_most_one_variant() -> None:
    """apply_teacache takes the FIRST registry entry whose matches() is True, and the
    registry is in package (alphabetical) order; two matches would make dispatch an
    accident of directory naming. Runs the real walk over the real catalog."""
    from mflux.models.common.config.model_config import AVAILABLE_MODELS

    from mlx_teacache.variants import _REGISTRY

    ambiguous = {}
    for key, cfg in AVAILABLE_MODELS.items():
        stub = SimpleNamespace(model_config=cfg)
        matched = [vid for vid, entry in _REGISTRY.items() if entry["matches"](stub)]
        if len(matched) > 1:
            ambiguous[key] = matched
    assert ambiguous == {}, ambiguous
    assert list(_REGISTRY) == sorted(_REGISTRY), "registry walk order is no longer alphabetical"
