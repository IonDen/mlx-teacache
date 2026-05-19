"""Compatibility shim. Canonical home is `mlx_teacache._kernel.stats`."""

from mlx_teacache._kernel.stats import (
    Decision,
    GenerationStats,
    StatsFrozenError,
    StepDecision,
    TeaCacheStats,
    _Staging,
)

__all__ = ["Decision", "GenerationStats", "StatsFrozenError", "StepDecision", "TeaCacheStats", "_Staging"]
