"""mflux-free detector."""


def matches(flux: object) -> bool:
    model_config = getattr(flux, "model_config", None)
    if model_config is None:
        return False
    aliases = getattr(model_config, "aliases", None) or []
    return "flux2-klein-9b" in aliases
