# src/mlx_teacache/integrations/mflux/detect.py
"""Identify which supported variant a flux instance is, or raise IncompatibleModelError.

Variant ID is the canonical string used throughout the rest of the package
(coefficient lookup, stats, error messages). mflux classes are imported
lazily at module load time so this module can be unit-tested with stub
classes via monkeypatching."""

from __future__ import annotations

from typing import Literal

from mlx_teacache.errors import IncompatibleModelError

VariantId = Literal["flux1-dev", "flux1-schnell", "flux2-klein-4b"]

_SUPPORTED: tuple[str, ...] = ("flux1-dev", "flux1-schnell", "flux2-klein-4b")


def _import_mflux_types() -> tuple[type, type]:
    """Import mflux types at first use. Returns (Flux1, Flux2Klein) or raises
    IncompatibleModelError if mflux is not installed."""
    try:
        from mflux.models.flux.variants.txt2img.flux import Flux1
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    except ImportError as e:
        raise IncompatibleModelError(
            actual_type="(mflux not installed)",
            actual_model_name=None,
            supported=list(_SUPPORTED),
        ) from e
    return Flux1, Flux2Klein


# Module-level slots replaced lazily on first identify_variant call.
# Tests can monkeypatch these to fake types.
_Flux1Type: type | None = None
_Flux2KleinType: type | None = None


def identify_variant(flux: object) -> VariantId:
    """Return the variant_id for a supported mflux Flux1 / Flux2Klein instance.

    Raises IncompatibleModelError for unsupported model_name, unsupported
    Flux2Klein configuration (9b, base variants), or any non-Flux type."""
    global _Flux1Type, _Flux2KleinType
    if _Flux1Type is None or _Flux2KleinType is None:
        _Flux1Type, _Flux2KleinType = _import_mflux_types()

    actual_type = type(flux).__name__
    model_config = getattr(flux, "model_config", None)
    model_name = getattr(model_config, "model_name", None)
    # mflux 0.17 stores the HF repo path in `model_name` and the user-facing
    # short names (e.g. "dev", "schnell") in `aliases`. The plain "dev" config
    # shares its `model_name` with controlnet/upscaler variants, so `aliases`
    # is the only correct discriminator.
    aliases = getattr(model_config, "aliases", None) or []

    if isinstance(flux, _Flux1Type):
        if "dev" in aliases:
            return "flux1-dev"
        if "schnell" in aliases:
            return "flux1-schnell"
        raise IncompatibleModelError(
            actual_type=actual_type, actual_model_name=model_name, supported=list(_SUPPORTED),
        )

    if isinstance(flux, _Flux2KleinType):
        if "flux2-klein-4b" in aliases:
            return "flux2-klein-4b"
        raise IncompatibleModelError(
            actual_type=actual_type, actual_model_name=model_name, supported=list(_SUPPORTED),
        )

    raise IncompatibleModelError(
        actual_type=actual_type, actual_model_name=model_name, supported=list(_SUPPORTED),
    )
