"""Compatibility shim. Canonical home is `mlx_teacache._kernel.stats`."""

from mlx_teacache._kernel.stats import (
    Decision,
    GenerationStats,
    StatsFrozenError,
    StepDecision,
    TeaCacheStats,
)
from mlx_teacache._kernel.stats import (
    _Staging as _Staging,  # noqa: F401  # re-export for v0.5.x callers (tests/test_stats.py)
)

__all__ = ["Decision", "GenerationStats", "StatsFrozenError", "StepDecision", "TeaCacheStats"]
