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


def _build_registry() -> None:
    package = importlib.import_module(__name__)
    for _, subname, ispkg in pkgutil.iter_modules(package.__path__):
        if not ispkg:
            continue
        full = f"{__name__}.{subname}"
        config = importlib.import_module(f"{full}.config")
        detect = importlib.import_module(f"{full}.detect")
        meta: dict[str, Any] = config.META
        variant_id = meta["variant_id"]
        _REGISTRY[variant_id] = _RegistryEntry(
            META=meta,
            matches=detect.matches,
            load_integration=_make_lazy_loader(full),
        )


_build_registry()

__all__ = ["_REGISTRY"]
