"""mflux-free detector for Qwen-Image base.

Base and Edit are distinguished by `model_config.aliases` (mflux
model_config.py:429-447): base = ["qwen-image", "qwen"], edit =
["qwen-image-edit", "qwen-edit", "qwen-edit-plus", "qwen-edit-2509"] (disjoint).
Element-membership on the bare "qwen-image"/"qwen" strings matches base only —
none of the edit aliases equals "qwen-image" or "qwen" as a list element, so the
edit model correctly falls through to IncompatibleModelError.
"""


def matches(flux: object) -> bool:
    model_config = getattr(flux, "model_config", None)
    if model_config is None:
        return False
    aliases = getattr(model_config, "aliases", None) or []
    return "qwen-image" in aliases or "qwen" in aliases


def is_calibrated_checkpoint(model_config: object, calibrated: str) -> bool:
    """True unless the model reports a model_name that is neither the calibrated checkpoint nor declares it as its
    base. mflux resolves a local path or a pre-quantized mirror by copying the base config and recording the base's
    model_name in base_model, so such a mirror counts as calibrated."""
    loaded = getattr(model_config, "model_name", None)
    base = getattr(model_config, "base_model", None)
    return not isinstance(loaded, str) or calibrated in (loaded, base)
