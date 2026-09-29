"""mflux-free detector for Z-Image base.

Base and Turbo are the SAME `ZImage` Python class, distinguished only by
`model_config.aliases`: base = ["z-image", "zimage"], turbo =
["z-image-turbo", "zimage-turbo"] (disjoint). Element-membership on the bare
"z-image"/"zimage" strings matches base only — turbo's aliases contain neither
as an element, so it correctly falls through to IncompatibleModelError.
"""

from mlx_teacache.variants._pipeline_class import is_foreign_mflux_pipeline

_ALLOWED = frozenset({"ZImage"})


def matches(flux: object) -> bool:
    if is_foreign_mflux_pipeline(flux, _ALLOWED):
        return False
    model_config = getattr(flux, "model_config", None)
    if model_config is None:
        return False
    aliases = getattr(model_config, "aliases", None) or []
    return "z-image" in aliases or "zimage" in aliases
