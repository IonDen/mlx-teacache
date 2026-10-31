"""Deprecated. Import TeaCacheStats, GenerationStats, StepDecision and StatsFrozenError from
mlx_teacache instead; Decision has no public replacement. The module is removed in 2.0.

Every attribute access warns with a DeprecationWarning at the caller's line.
"""

from typing import TYPE_CHECKING

from mlx_teacache._deprecation import shim_getattr

if TYPE_CHECKING:
    from mlx_teacache._kernel.stats import (
        Decision,
        GenerationStats,
        StatsFrozenError,
        StepDecision,
        TeaCacheStats,
    )
    from mlx_teacache._kernel.stats import _Staging as _Staging

__all__ = ["Decision", "GenerationStats", "StatsFrozenError", "StepDecision", "TeaCacheStats"]

# _Staging is not in __all__ but was importable from here in 0.15.
_SOURCES = dict.fromkeys([*__all__, "_Staging"], "mlx_teacache._kernel.stats")


def __getattr__(name: str) -> object:
    return shim_getattr(__name__, name, _SOURCES)


def __dir__() -> list[str]:
    return sorted(__all__)
