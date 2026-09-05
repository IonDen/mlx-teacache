"""mflux-free detector. Krea's aliases are ["krea-dev", "dev-krea"]; list
membership keeps it disjoint from flux1_dev's `"dev" in aliases`."""


def matches(flux: object) -> bool:
    model_config = getattr(flux, "model_config", None)
    if model_config is None:
        return False
    aliases = getattr(model_config, "aliases", None) or []
    return "krea-dev" in aliases
