"""Attribute access for the deprecated root modules (PEP 562 module ``__getattr__``).

Each deprecated module imports nothing from ``_kernel/`` at run time; it imports only
``shim_getattr`` from here. The shim's own module-level ``__getattr__`` calls ``shim_getattr``
for every lookup, which warns at the caller's line and returns the object from its real home.
"""

import importlib
import warnings
from collections.abc import Mapping

# Names the package root exports, so the warning can point users there.
_ROOT_PUBLIC = frozenset(
    {"TeaCacheStats", "GenerationStats", "StepDecision", "StatsFrozenError", "Provenance"}
)


def shim_getattr(shim: str, name: str, sources: Mapping[str, str]) -> object:
    """Resolve ``name`` for the deprecated module ``shim``; ``sources`` maps names to their modules.

    An unknown name raises ``AttributeError`` before any warning, so ``hasattr`` probes stay silent.
    The warning uses stacklevel=3 (this helper, the module ``__getattr__``, then the caller), which
    lands on the user's ``from shim import name`` or ``shim.name`` line.
    """
    if name not in sources:
        raise AttributeError(f"module {shim!r} has no attribute {name!r}")
    if name in _ROOT_PUBLIC:
        advice = f"is deprecated; import {name} from mlx_teacache instead."
    else:
        advice = "is deprecated and has no public replacement."
    warnings.warn(
        f"{shim}.{name} {advice} The {shim} module is outside the 1.0 stability promise "
        "and will be removed in 2.0.",
        DeprecationWarning,
        stacklevel=3,
    )
    return getattr(importlib.import_module(sources[name]), name)
