# src/mlx_teacache/__init__.py
"""mlx-teacache — TeaCache step-skipping for FLUX, Qwen-Image, and Z-Image on Apple Silicon.

Public API:
    apply_teacache(flux, *, rel_l1_thresh=..., ...)
        Enable TeaCache on a supported mflux FLUX, Qwen-Image, or Z-Image model.
        rel_l1_thresh defaults to the variant's Provenance.default_thresh
        if set: 0.17 for flux2-klein-base, 0.12 for z-image-base, and 0.30
        for qwen-image. Other variants use the 0.20 package fallback.

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
    "IncompatibleModelError",
    "AlreadyPatchedError",
    "CalibrationError",
    "TransformerShapeError",
    "InternalStateError",
    "InvalidStepWindowError",
    "MissingGenerationContextError",
]
