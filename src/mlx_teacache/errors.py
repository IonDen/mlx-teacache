# src/mlx_teacache/errors.py
"""Typed exception hierarchy for mlx-teacache. Every error names the parameter
and actual value; messages include remediation pointers where applicable."""

from __future__ import annotations


class TeaCacheError(Exception):
    """Base class — catch this to catch anything from mlx-teacache."""


class TeaCacheValueError(TeaCacheError, ValueError):
    """Raised for invalid arguments to apply_teacache / custom coefficients.
    IS-A ValueError, so existing ``except ValueError`` callers still catch it."""


class IncompatibleModelError(TeaCacheError):
    def __init__(self, *, actual_type: str, actual_model_name: str | None, supported: list[str]) -> None:
        super().__init__(
            f"Unsupported model: {actual_type} (model_name={actual_model_name!r}). "
            f"Supported variants: {', '.join(supported)}. "
            f"See https://github.com/IonDen/mlx-teacache#supported-models"
        )
        self.actual_type = actual_type
        self.actual_model_name = actual_model_name
        self.supported = supported


class AlreadyPatchedError(TeaCacheError):
    def __init__(self, *, variant_id: str, rel_l1_thresh: float) -> None:
        super().__init__(
            f"This flux instance is already patched by mlx-teacache "
            f"(variant_id={variant_id!r}, rel_l1_thresh={rel_l1_thresh}). "
            f"Call handle.restore() on the existing handle before applying again."
        )
        self.variant_id = variant_id
        self.rel_l1_thresh = rel_l1_thresh


class CalibrationError(TeaCacheError):
    def __init__(self, *, variant_id: str, reason: str) -> None:
        super().__init__(f"Coefficient calibration data for variant {variant_id!r} is invalid: {reason}.")
        self.variant_id = variant_id
        self.reason = reason


class TransformerShapeError(TeaCacheError):
    def __init__(self, step_idx: int, expected: tuple[int, ...], actual: tuple[int, ...]) -> None:
        super().__init__(
            f"Transformer output shape changed mid-generation: "
            f"step_idx={step_idx} expected={expected} actual={actual}. "
            f"This indicates an mflux internal inconsistency or a bug in mlx-teacache; "
            f"please file an issue."
        )
        self.step_idx = step_idx
        self.expected = expected
        self.actual = actual


class InvalidStepWindowError(TeaCacheError):
    def __init__(
        self,
        *,
        skip_first: int,
        skip_last: int,
        num_steps: int,  # legacy alias for active count
        nominal_num_inference_steps: int | None = None,
        active_num_steps: int | None = None,
    ) -> None:
        # Resolve which value represents the "active" denoising step count.
        # New callers may pass active_num_steps explicitly; old callers pass it
        # via the legacy `num_steps` keyword.
        active = num_steps if active_num_steps is None else active_num_steps

        if nominal_num_inference_steps is not None and nominal_num_inference_steps != active:
            tail = (
                f"; got skip_first={skip_first}, skip_last={skip_last}, "
                f"active_num_steps={active}, nominal_num_inference_steps={nominal_num_inference_steps} "
                f"(sum {skip_first + skip_last} >= {active})."
            )
        else:
            tail = (
                f". Got skip_first={skip_first}, skip_last={skip_last}, "
                f"active_num_steps={active} "
                f"(sum {skip_first + skip_last} >= {active})."
            )
        super().__init__("skip_first_n_steps + skip_last_n_steps must be < active denoising steps" + tail)
        self.skip_first = skip_first
        self.skip_last = skip_last
        self.num_steps = num_steps  # legacy attribute name preserved
        self.nominal_num_inference_steps = nominal_num_inference_steps
        self.active_num_steps = active


class MissingGenerationContextError(TeaCacheError):
    def __init__(self, detail: str | None = None) -> None:
        msg = (
            "FLUX.2 generation started but no fresh generation context was captured. "
            "This usually means flux.callbacks was replaced or cleared after apply_teacache(), "
            "or a previous generation crashed before lifecycle cleanup completed. "
            "Call handle.restore() and apply_teacache() again."
        )
        if detail:
            msg = f"{msg} (detail: {detail})"
        super().__init__(msg)


class InternalStateError(TeaCacheError):
    """Raised when an internal cache/state invariant is violated. Indicates
    a bug in mlx-teacache itself or a defensive guard tripping. Distinct
    from TransformerShapeError (which is about shape drift in real tensors)."""


class TeaCacheNoBenefitWarning(UserWarning):
    """Emitted once per TeaCacheHandle when the active step configuration
    leaves no possible cache skips (i.e., `active_num_steps - skip_first -
    skip_last <= 1`, so the polynomial gate cannot seed AND consume the
    cache within this generation). Triggers on very short schedules and
    aggressive skip windows.

    Suppress via the standard `warnings` module:
        warnings.filterwarnings("ignore", category=TeaCacheNoBenefitWarning)
    """


class TeaCacheDisabledWarning(UserWarning):
    """Emitted once at apply time when rel_l1_thresh=0.0 — caching is disabled
    (every step computes; no speedup). Higher threshold = MORE skips; 0.0 disables.

    Suppress via the standard `warnings` module:
        warnings.filterwarnings("ignore", category=TeaCacheDisabledWarning)
    """
