"""mflux-free detector for Qwen-Image-2512.

mflux 0.19 and later give the `qwen-image` config the aliases ["qwen-image", "qwen", "qwen-image-2512",
"qwen-2512"] and load Qwen/Qwen-Image-2512 from it. The split from the original Qwen-Image variant is one rule,
`qwen_image.detect.is_2512_checkpoint`: a 2512 alias, unless model_name is the original checkpoint (mflux resolves
`Qwen/Qwen-Image` by repo id against the same config). The original variant matches exactly the configs that rule
rejects, so the two never both match. Edit-2511 (`zero_cond_t`) is refused like the original refuses it.
"""

from mlx_teacache.variants._pipeline_class import config_matches
from mlx_teacache.variants.qwen_image.detect import (
    ALIASES_2512,
    has_zero_cond_t,
    is_2512_checkpoint,
)
from mlx_teacache.variants.qwen_image.detect import (
    is_calibrated_checkpoint as is_calibrated_checkpoint,
)

_ALLOWED = frozenset({"QwenImage"})

# The mflux model aliases this variant is detected by (mflux 0.19 and later). mflux 0.17.5 and 0.18 have no 2512
# config: there these names infer a copy of Qwen/Qwen-Image and get the original variant.
MODEL_NAMES: tuple[str, ...] = ALIASES_2512

__all__ = ["MODEL_NAMES", "is_calibrated_checkpoint", "matches", "matches_config"]


def matches_config(model_config: object, pipeline_class: type) -> bool:
    """True when `pipeline_class` building from `model_config` is this variant. Reads no weights."""
    if has_zero_cond_t(model_config) or not is_2512_checkpoint(model_config):
        return False
    return config_matches(model_config, pipeline_class, _ALLOWED, MODEL_NAMES)


def matches(flux: object) -> bool:
    # Reads the BUILT transformer, which matches_config cannot see (see qwen_image.detect.matches).
    if getattr(getattr(flux, "transformer", None), "zero_cond_t", False):
        return False
    return matches_config(getattr(flux, "model_config", None), type(flux))
