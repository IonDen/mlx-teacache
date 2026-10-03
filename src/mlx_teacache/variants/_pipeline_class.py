"""mflux-free guard shared by every variant's detect.matches_config()."""


def is_foreign_mflux_class(cls: type, allowed: frozenset[str]) -> bool:
    """True when `cls` is an mflux pipeline class this variant does not implement.

    mflux builds several pipeline classes on the same ModelConfig (edit, concept,
    in-context, redux, controlnet ...). Their call contracts differ from the
    text-to-image pipeline the variant patches, so the alias alone is not enough.
    Classes defined outside mflux are left to alias matching."""
    module = getattr(cls, "__module__", "") or ""
    from_mflux = module == "mflux" or module.startswith("mflux.")
    return from_mflux and cls.__name__ not in allowed


def config_matches(
    model_config: object, pipeline_class: type, allowed: frozenset[str], model_names: tuple[str, ...]
) -> bool:
    """The detection rule every variant shares: not a foreign mflux pipeline class, a model config is present, and
    one of `model_names` is among its aliases. Reads no weights."""
    if is_foreign_mflux_class(pipeline_class, allowed) or model_config is None:
        return False
    aliases = getattr(model_config, "aliases", None) or []
    return any(name in aliases for name in model_names)
