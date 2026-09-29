"""mflux-free detector."""

from mlx_teacache.variants._pipeline_class import is_foreign_mflux_pipeline

_ALLOWED = frozenset({"Flux2Klein"})


def matches(flux: object) -> bool:
    if is_foreign_mflux_pipeline(flux, _ALLOWED):
        return False
    model_config = getattr(flux, "model_config", None)
    if model_config is None:
        return False
    aliases = getattr(model_config, "aliases", None) or []
    return "flux2-klein-base-4b" in aliases
