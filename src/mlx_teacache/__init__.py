# src/mlx_teacache/__init__.py
"""mlx-teacache — TeaCache step-skipping for FLUX diffusion on Apple Silicon.

Public API:
    apply_teacache(flux, *, rel_l1_thresh=..., ...)
        Enable TeaCache on an mflux Flux1 or Flux2Klein instance.
        rel_l1_thresh defaults to the variant's Provenance.default_thresh
        if set (e.g. 0.17 for flux2-klein-base-4b and flux2-klein-base-9b),
        otherwise 0.20.

    TeaCacheHandle
        Context-manager-compatible return value with .stats, .provenance, .restore().

    TeaCacheStats, StepDecision, GenerationStats
        Stats reporting types.

    Provenance
        Coefficient provenance metadata.

    TeaCacheError + subclasses
        Typed exception hierarchy. Catch TeaCacheError to handle anything.
"""

from __future__ import annotations

from mlx_teacache._version import __version__
from mlx_teacache.api import apply_teacache
from mlx_teacache.coefficients import Provenance
from mlx_teacache.errors import (
    AlreadyPatchedError,
    CalibrationError,
    IncompatibleModelError,
    InternalStateError,
    InvalidStepWindowError,
    MissingGenerationContextError,
    TeaCacheError,
    TeaCacheNoBenefitWarning,
    TransformerShapeError,
)
from mlx_teacache.handle import TeaCacheHandle
from mlx_teacache.stats import (
    GenerationStats,
    StatsFrozenError,
    StepDecision,
    TeaCacheStats,
)

__all__ = [
    "__version__",
    "apply_teacache",
    "TeaCacheHandle",
    "TeaCacheStats",
    "GenerationStats",
    "StepDecision",
    "StatsFrozenError",
    "Provenance",
    "TeaCacheError",
    "TeaCacheNoBenefitWarning",
    "IncompatibleModelError",
    "AlreadyPatchedError",
    "CalibrationError",
    "TransformerShapeError",
    "InternalStateError",
    "InvalidStepWindowError",
    "MissingGenerationContextError",
]
