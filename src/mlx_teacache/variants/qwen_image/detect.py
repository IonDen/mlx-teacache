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
