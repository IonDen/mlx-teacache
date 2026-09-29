"""mflux-free detector. Krea's aliases are ["krea-dev", "dev-krea"]; list
membership keeps it disjoint from flux1_dev's `"dev" in aliases`."""

from mlx_teacache.variants._pipeline_class import is_foreign_mflux_pipeline

_ALLOWED = frozenset({"Flux1"})


def matches(flux: object) -> bool:
    if is_foreign_mflux_pipeline(flux, _ALLOWED):
        return False
    model_config = getattr(flux, "model_config", None)
    if model_config is None:
        return False
    aliases = getattr(model_config, "aliases", None) or []
    return "krea-dev" in aliases
