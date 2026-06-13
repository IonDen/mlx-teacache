"""Variant registry. Walks every subpackage of variants/ at import time
to populate _REGISTRY with (META, matches, load_integration) entries.

config.py + detect.py are imported eagerly (they must be mflux-free).
integration.py is loaded lazily via load_integration() — apply_teacache
calls it after detect picks the winning variant. This is the contract
that keeps `import mlx_teacache` working without the [mflux] extra.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any, TypedDict, cast

from mlx_teacache._kernel.coefficients import validate_custom
from mlx_teacache.errors import CalibrationError, TeaCacheValueError


class _RegistryEntry(TypedDict):
    META: dict[str, Any]
    matches: Callable[[object], bool]
    load_integration: Callable[[], Callable[..., Any]]


_REGISTRY: dict[str, _RegistryEntry] = {}


def _make_lazy_loader(module_name: str) -> Callable[[], Callable[..., Any]]:
    def _load() -> Callable[..., Any]:
        integration = importlib.import_module(f"{module_name}.integration")
        return cast(Callable[..., Any], integration.apply)

    return _load


_REQUIRED_META_KEYS = ("variant_id", "display_name", "license")


def _validate_meta(meta: object, *, subname: str) -> dict[str, Any]:
    """Validate a variant's META mapping, raising a CalibrationError that names
    the subpackage so a malformed variant can't fail `import mlx_teacache` with an
    opaque AttributeError/KeyError (per 0031 #4)."""
    if not isinstance(meta, dict):
        raise CalibrationError(
            variant_id=subname,
            reason=f"variant {subname!r} has a missing or non-dict META",
        )
    missing = [key for key in _REQUIRED_META_KEYS if key not in meta]
    if missing:
        raise CalibrationError(
            variant_id=str(meta.get("variant_id", subname)),
            reason=f"variant {subname!r} META is missing required key(s): {missing}",
        )
    return meta


def _build_one(full: str, subname: str) -> tuple[str, _RegistryEntry]:
    """Import + validate a single variant subpackage. Any failure is surfaced as
    a CalibrationError naming the subpackage, so one broken variant can't take
    down the whole registry with an opaque error (per 0031 #4)."""
    try:
        config = importlib.import_module(f"{full}.config")
        detect = importlib.import_module(f"{full}.detect")
    except Exception as e:
        raise CalibrationError(
            variant_id=subname,
            reason=f"failed to import variant subpackage {subname!r}: {type(e).__name__}: {e}",
        ) from e

    meta = _validate_meta(getattr(config, "META", None), subname=subname)
    variant_id = str(meta["variant_id"])

    coeffs = getattr(config, "COEFFICIENTS", None)
    if coeffs is None:
        raise CalibrationError(
            variant_id=variant_id,
            reason="COEFFICIENTS attribute is missing from variant config",
        )
    try:
        validate_custom(coeffs)
    except TeaCacheValueError as e:
        raise CalibrationError(variant_id=variant_id, reason=str(e)) from e

    return variant_id, _RegistryEntry(
        META=meta,
        matches=detect.matches,
        load_integration=_make_lazy_loader(full),
    )


def _build_registry() -> None:
    package = importlib.import_module(__name__)
    for _, subname, ispkg in pkgutil.iter_modules(package.__path__):
        if not ispkg:
            continue
        variant_id, entry = _build_one(f"{__name__}.{subname}", subname)
        _REGISTRY[variant_id] = entry


_build_registry()

__all__ = ["_REGISTRY"]
