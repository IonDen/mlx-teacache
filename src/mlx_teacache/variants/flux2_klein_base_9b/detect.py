"""mflux-free detector."""

from mlx_teacache.variants._pipeline_class import config_matches

_ALLOWED = frozenset({"Flux2Klein"})

# The mflux model aliases this variant is detected by; each is a valid `--model` value.
MODEL_NAMES: tuple[str, ...] = ("flux2-klein-base-9b",)


def matches_config(model_config: object, pipeline_class: type) -> bool:
    """True when `pipeline_class` building from `model_config` is this variant. Reads no weights."""
    return config_matches(model_config, pipeline_class, _ALLOWED, MODEL_NAMES)


def matches(flux: object) -> bool:
    return matches_config(getattr(flux, "model_config", None), type(flux))
