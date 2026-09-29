"""mflux-free guard shared by every variant's detect.matches()."""


def is_foreign_mflux_pipeline(flux: object, allowed: frozenset[str]) -> bool:
    """True when `flux` is an mflux pipeline class this variant does not implement.

    mflux builds several pipeline classes on the same ModelConfig (edit, concept,
    in-context, redux, controlnet ...). Their call contracts differ from the
    text-to-image pipeline the variant patches, so the alias alone is not enough.
    Objects whose class is defined outside mflux are left to alias matching."""
    cls = type(flux)
    module = getattr(cls, "__module__", "") or ""
    from_mflux = module == "mflux" or module.startswith("mflux.")
    return from_mflux and cls.__name__ not in allowed
