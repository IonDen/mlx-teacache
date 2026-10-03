"""mflux-free detector for Qwen-Image base.

Base and Edit are distinguished by `model_config.aliases` (mflux
model_config.py:429-447): base = ["qwen-image", "qwen"], edit =
["qwen-image-edit", "qwen-edit", "qwen-edit-plus", "qwen-edit-2509"] (disjoint).
Element-membership on the bare "qwen-image"/"qwen" strings matches base only —
none of the edit aliases equals "qwen-image" or "qwen" as a list element, so the
edit model correctly falls through to IncompatibleModelError.
"""

from mlx_teacache.variants._pipeline_class import config_matches

_ALLOWED = frozenset({"QwenImage"})

# The mflux model aliases this variant is detected by; each is a valid `--model` value.
MODEL_NAMES: tuple[str, ...] = ("qwen-image", "qwen")


def matches_config(model_config: object, pipeline_class: type) -> bool:
    """True when `pipeline_class` building from `model_config` is this variant. Reads no weights."""
    return config_matches(model_config, pipeline_class, _ALLOWED, MODEL_NAMES)


def matches(flux: object) -> bool:
    return matches_config(getattr(flux, "model_config", None), type(flux))


def is_calibrated_checkpoint(model_config: object, calibrated: str) -> bool:
    """True unless the model reports a model_name that is neither the calibrated checkpoint nor declares it as its
    base. mflux resolves a local path or a pre-quantized mirror by copying the base config and recording the base's
    model_name in base_model, so such a mirror counts as calibrated. On mflux 0.19 and later a
    pre-quantized mirror resolves against the `qwen-image` alias config, which is Qwen-Image-2512, so it reports
    uncalibrated even when it re-quantizes the original Qwen-Image."""
    loaded = getattr(model_config, "model_name", None)
    base = getattr(model_config, "base_model", None)
    return not isinstance(loaded, str) or calibrated in (loaded, base)
