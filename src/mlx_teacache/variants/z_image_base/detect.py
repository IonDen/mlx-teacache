"""mflux-free detector for Z-Image base.

Base and Turbo are the SAME `ZImage` Python class, distinguished only by
`model_config.aliases`: base = ["z-image", "zimage"], turbo =
["z-image-turbo", "zimage-turbo"] (disjoint). Element-membership on the bare
"z-image"/"zimage" strings matches base only — turbo's aliases contain neither
as an element, so it correctly falls through to IncompatibleModelError.
"""

from mlx_teacache.variants._pipeline_class import config_matches

_ALLOWED = frozenset({"ZImage"})

# The mflux model aliases this variant is detected by; each is a valid `--model` value.
MODEL_NAMES: tuple[str, ...] = ("z-image", "zimage")


def matches_config(model_config: object, pipeline_class: type) -> bool:
    """True when `pipeline_class` building from `model_config` is this variant. Reads no weights."""
    return config_matches(model_config, pipeline_class, _ALLOWED, MODEL_NAMES)


def matches(flux: object) -> bool:
    return matches_config(getattr(flux, "model_config", None), type(flux))
