# src/mlx_teacache/__init__.py
"""mlx-teacache — TeaCache step-skipping for FLUX, Qwen-Image, and Z-Image on Apple Silicon.

Public API:
    apply_teacache(flux, *, rel_l1_thresh=..., ...)
        Enable TeaCache on a supported mflux FLUX, Qwen-Image, or Z-Image model.
        rel_l1_thresh defaults to a per-variant value (see apply_teacache's docstring):
        0.20 flux1-dev / flux1-schnell, 0.30 flux1-krea-dev, 0.17 flux2-klein-base-4b /
        -base-9b, 0.12 z-image-base, 0.30 qwen-image. Distilled flux2-klein-4b / -9b have
        none and use 0.20.

    TeaCacheHandle
        Context-manager-compatible return value with .stats, .provenance, .restore().

    TeaCacheStats, StepDecision, GenerationStats
        Stats reporting types.

    Provenance
        Coefficient provenance metadata.

    TeaCacheError + subclasses
        Typed exception hierarchy. Catch TeaCacheError to handle anything.
"""

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.stats import (
    GenerationStats,
    StatsFrozenError,
    StepDecision,
    TeaCacheStats,
)
from mlx_teacache._version import __version__
from mlx_teacache.api import apply_teacache
from mlx_teacache.errors import (
    AlreadyPatchedError,
    CalibrationError,
    IncompatibleModelError,
    InternalStateError,
    InvalidStepWindowError,
    MissingGenerationContextError,
    TeaCacheDisabledWarning,
    TeaCacheError,
    TeaCacheNoBenefitWarning,
    TeaCacheUncalibratedCheckpointWarning,
    TeaCacheUntestedMfluxWarning,
    TeaCacheValueError,
    TransformerShapeError,
)
from mlx_teacache.handle import TeaCacheHandle

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
    "TeaCacheValueError",
    "TeaCacheDisabledWarning",
    "TeaCacheNoBenefitWarning",
    "TeaCacheUncalibratedCheckpointWarning",
    "TeaCacheUntestedMfluxWarning",
    "IncompatibleModelError",
    "AlreadyPatchedError",
    "CalibrationError",
    "TransformerShapeError",
    "InternalStateError",
    "InvalidStepWindowError",
    "MissingGenerationContextError",
]
