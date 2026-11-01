"""mflux-free detector for Qwen-Image base.

Base and Edit are distinguished by `model_config.aliases` (mflux
model_config.py:429-447): base = ["qwen-image", "qwen"], edit =
["qwen-image-edit", "qwen-edit", "qwen-edit-plus", "qwen-edit-2509"] (disjoint).
Element-membership on the bare "qwen-image"/"qwen" strings matches base only —
none of the edit aliases equals "qwen-image" or "qwen" as a list element, so the
edit model correctly falls through to IncompatibleModelError.

mflux 0.22 also builds Qwen-Image-Edit-2511 with `zero_cond_t` (its config sets
`transformer_overrides={"zero_cond_t": True}`): the reference-image tokens take the
timestep-0 modulation. The gated forward does not implement that path, so a config or a
transformer with it on is not this variant. Text-to-image Qwen-Image always has it off.

mflux 0.19 and later give every Qwen-Image text-to-image config the Qwen-Image-2512 aliases too, including the
original checkpoint loaded by its repo id (mflux resolves `Qwen/Qwen-Image` as a mirror of the `qwen-image` config,
recording base_model Qwen/Qwen-Image-2512). `is_2512_checkpoint` (in the shared `variants/_qwen_checkpoint.py`)
holds the one rule that splits the two variants: 2512 aliases mean Qwen-Image-2512 unless model_name is the
original checkpoint. This variant matches only configs the rule rejects, the qwen_image_2512 variant only configs it
accepts, so a config never matches both.
"""

from mlx_teacache.variants._pipeline_class import config_matches
from mlx_teacache.variants._qwen_checkpoint import has_zero_cond_t, is_2512_checkpoint
from mlx_teacache.variants._qwen_checkpoint import is_calibrated_checkpoint as is_calibrated_checkpoint

_ALLOWED = frozenset({"QwenImage"})

# The mflux model aliases this variant is detected by. They load the original checkpoint only on mflux 0.17.5 and
# 0.18; on mflux 0.19 and later they load Qwen-Image-2512 (the qwen_image_2512 variant), and only
# `ModelConfig.from_name("Qwen/Qwen-Image")` selects the original.
MODEL_NAMES: tuple[str, ...] = ("qwen-image", "qwen")


def matches_config(model_config: object, pipeline_class: type) -> bool:
    """True when `pipeline_class` building from `model_config` is this variant. Reads no weights."""
    if has_zero_cond_t(model_config) or is_2512_checkpoint(model_config):
        return False
    return config_matches(model_config, pipeline_class, _ALLOWED, MODEL_NAMES)


def matches(flux: object) -> bool:
    # Not a second copy of the config rule: this reads the BUILT transformer. A transformer built with
    # zero_cond_t, or swapped in after the model config was made, is invisible to matches_config.
    if getattr(getattr(flux, "transformer", None), "zero_cond_t", False):
        return False
    return matches_config(getattr(flux, "model_config", None), type(flux))
