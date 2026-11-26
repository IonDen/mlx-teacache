# src/mlx_teacache/errors.py
"""Typed exception hierarchy for mlx-teacache. Every error names the parameter
and actual value; messages include remediation pointers where applicable."""

import warnings
from typing import Any, Literal


def _rebuild(cls: type, kwargs: dict[str, Any]) -> Any:
    """Unpickle helper for exceptions whose constructors are keyword-only.

    Each ``__reduce__`` below also returns the instance ``__dict__`` as state, so
    ``add_note`` notes and attributes set after construction survive the round trip.
    An InvalidStepWindowError pickled by 0.15 stores ``num_steps``; it is read as
    ``active_num_steps`` when that is missing, so the load does not warn."""
    if issubclass(cls, InvalidStepWindowError) and "num_steps" in kwargs:
        kwargs = dict(kwargs)
        legacy = kwargs.pop("num_steps")
        if kwargs.get("active_num_steps") is None:
            kwargs["active_num_steps"] = legacy
    return cls(**kwargs)


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

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _rebuild,
            (
                type(self),
                {
                    "actual_type": self.actual_type,
                    "actual_model_name": self.actual_model_name,
                    "supported": self.supported,
                },
            ),
            self.__dict__,
        )


class _StepCacheConflictError(IncompatibleModelError):
    """A patched model's generate_image call also turned on mflux's own step cache. Private: callers catch
    IncompatibleModelError. The model is supported, so `supported` holds the one variant TeaCache patched on it
    (empty if that id is unknown), and `requested` the arguments that turned the step cache on."""

    def __init__(
        self, *, actual_type: str, actual_model_name: str | None, supported: list[str], requested: str
    ) -> None:
        super().__init__(actual_type=actual_type, actual_model_name=actual_model_name, supported=supported)
        self.args = (
            f"This call turns on mflux's step cache ({requested}). mflux's step cache "
            "(step_cache_ratio / teacache_ratio / use_step_cache) and TeaCache both skip steps, so use one: "
            "call handle.restore() to use mflux's step cache, or turn it off to keep TeaCache: drop the argument, "
            "or pass None for the ratio and False for use_step_cache.",
        )
        self.requested = requested

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _rebuild,
            (
                type(self),
                {
                    "actual_type": self.actual_type,
                    "actual_model_name": self.actual_model_name,
                    "supported": self.supported,
                    "requested": self.requested,
                },
            ),
            self.__dict__,
        )


def _versions(mflux_version: str | None, teacache_version: str | None) -> str:
    """'mflux 0.24.0, mlx-teacache 0.18.1' for the predict errors; a missing one reads '(version unknown)'."""
    return ", ".join(
        f"{name} {found}" if found is not None else f"{name} (version unknown)"
        for name, found in (("mflux", mflux_version), ("mlx-teacache", teacache_version))
    )


class _UnknownPredictKeywordError(IncompatibleModelError):
    """The installed mflux called a patched model's `_predict` with keyword arguments TeaCache's replacement does not
    know. Private: callers catch IncompatibleModelError. `supported` is always empty and `actual_model_name` is None:
    the factory that raises sees only the session, not the model. `unknown` holds the keyword names, `mflux_version`
    the installed mflux and `teacache_version` the installed mlx-teacache (each None if unknown)."""

    def __init__(
        self,
        *,
        actual_type: str,
        actual_model_name: str | None,
        supported: list[str],
        unknown: list[str],
        mflux_version: str | None,
        teacache_version: str | None,
    ) -> None:
        super().__init__(actual_type=actual_type, actual_model_name=actual_model_name, supported=supported)
        installed = (
            f"mflux {mflux_version}" if mflux_version is not None else "the installed mflux (version unknown)"
        )
        self.args = (
            f"{installed} called {actual_type}._predict with keyword arguments TeaCache does not know: "
            f"{', '.join(unknown)}. TeaCache's replacement accepts compile_predict only. "
            "Call handle.restore() to generate without TeaCache, and please report both versions "
            f"({_versions(mflux_version, teacache_version)}) at https://github.com/IonDen/mlx-teacache/issues.",
        )
        self.unknown = unknown
        self.mflux_version = mflux_version
        self.teacache_version = teacache_version

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _rebuild,
            (
                type(self),
                {
                    "actual_type": self.actual_type,
                    "actual_model_name": self.actual_model_name,
                    "supported": self.supported,
                    "unknown": self.unknown,
                    "mflux_version": self.mflux_version,
                    "teacache_version": self.teacache_version,
                },
            ),
            self.__dict__,
        )


class _PredictNotCalledError(IncompatibleModelError):
    """A patched model's generate_image reached its first denoising step without asking the patched `_predict` for
    its step function, so that step was built without TeaCache. Private: callers catch IncompatibleModelError.
    `mflux_version` is the installed mflux and `teacache_version` the installed mlx-teacache (each None if
    unknown)."""

    def __init__(
        self,
        *,
        actual_type: str,
        actual_model_name: str | None,
        supported: list[str],
        mflux_version: str | None,
        teacache_version: str | None,
    ) -> None:
        super().__init__(actual_type=actual_type, actual_model_name=actual_model_name, supported=supported)
        installed = (
            f"mflux {mflux_version}" if mflux_version is not None else "the installed mflux (version unknown)"
        )
        self.args = (
            f"TeaCache is applied, but {actual_type}.generate_image built its first denoising step without "
            f"calling self._predict, so that step has no TeaCache and the generation was stopped. {installed} most "
            "likely no longer builds its step function through self._predict. Call handle.restore() to generate "
            "without TeaCache, and please report both versions "
            f"({_versions(mflux_version, teacache_version)}) at https://github.com/IonDen/mlx-teacache/issues.",
        )
        self.mflux_version = mflux_version
        self.teacache_version = teacache_version

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _rebuild,
            (
                type(self),
                {
                    "actual_type": self.actual_type,
                    "actual_model_name": self.actual_model_name,
                    "supported": self.supported,
                    "mflux_version": self.mflux_version,
                    "teacache_version": self.teacache_version,
                },
            ),
            self.__dict__,
        )


class _SsdStreamError(IncompatibleModelError):
    """apply_teacache was given a model built with mflux's SSD streaming (`ssd_stream=True`, `--ssd-stream`), which
    TeaCache has not been verified with. Private: callers catch IncompatibleModelError. `supported` holds the variant
    the model otherwise matched."""

    def __init__(self, *, actual_type: str, actual_model_name: str | None, supported: list[str]) -> None:
        super().__init__(actual_type=actual_type, actual_model_name=actual_model_name, supported=supported)
        self.args = (
            f"TeaCache does not support SSD streaming yet: this {actual_type} model "
            f"(model_name={actual_model_name!r}) was built with ssd_stream=True (--ssd-stream). "
            "Build the model without ssd_stream=True / --ssd-stream to use TeaCache.",
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _rebuild,
            (
                type(self),
                {
                    "actual_type": self.actual_type,
                    "actual_model_name": self.actual_model_name,
                    "supported": self.supported,
                },
            ),
            self.__dict__,
        )


class AlreadyPatchedError(TeaCacheError):
    def __init__(self, *, variant_id: str, rel_l1_thresh: float) -> None:
        super().__init__(
            f"This flux instance is already patched by mlx-teacache "
            f"(variant_id={variant_id!r}, rel_l1_thresh={rel_l1_thresh}). "
            f"Call handle.restore() on the existing handle before applying again."
        )
        self.variant_id = variant_id
        self.rel_l1_thresh = rel_l1_thresh

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _rebuild,
            (type(self), {"variant_id": self.variant_id, "rel_l1_thresh": self.rel_l1_thresh}),
            self.__dict__,
        )


class CalibrationError(TeaCacheError):
    def __init__(self, *, variant_id: str, reason: str) -> None:
        super().__init__(f"Coefficient calibration data for variant {variant_id!r} is invalid: {reason}.")
        self.variant_id = variant_id
        self.reason = reason

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuild, (type(self), {"variant_id": self.variant_id, "reason": self.reason}), self.__dict__)


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

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.step_idx, self.expected, self.actual), self.__dict__)


class InvalidStepWindowError(TeaCacheError):
    def __init__(
        self,
        *,
        skip_first: int,
        skip_last: int,
        num_steps: int | None = None,
        nominal_num_inference_steps: int | None = None,
        active_num_steps: int | None = None,
    ) -> None:
        # active_num_steps is the keyword for the active denoising step count. num_steps is a
        # deprecated alias: it warns, and active_num_steps wins when both are passed.
        if num_steps is not None:
            warnings.warn(
                "num_steps= is deprecated; pass active_num_steps=. It will be removed in 2.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        if active_num_steps is not None:
            active = active_num_steps
        elif num_steps is not None:
            active = num_steps
        else:
            raise TypeError("InvalidStepWindowError needs active_num_steps=")

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
        # The value passed as num_steps, or the active count when only active_num_steps was given.
        self.num_steps = num_steps if num_steps is not None else active
        self.nominal_num_inference_steps = nominal_num_inference_steps
        self.active_num_steps = active
        # Constructor inputs that re-derive the same message on a round trip. num_steps is left out
        # so unpickling never warns; the restored __dict__ brings back its value.
        self._ctor = {
            "skip_first": skip_first,
            "skip_last": skip_last,
            "nominal_num_inference_steps": nominal_num_inference_steps,
            "active_num_steps": active,
        }

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuild, (type(self), dict(self._ctor)), self.__dict__)


class MissingGenerationContextError(TeaCacheError):
    def __init__(self, detail: str | None = None) -> None:
        # A detail comes from a raiser whose text already carries the remedy.
        msg = detail or (
            "A generation started but no fresh generation context was captured. "
            "This usually means flux.callbacks was replaced or cleared after apply_teacache(), "
            "or a previous generation crashed before cleanup. "
            "Call handle.restore() and apply_teacache() again."
        )
        super().__init__(msg)
        self.detail = detail

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.detail,), self.__dict__)


class RestoreRefusedError(TeaCacheError):
    """Raised by `TeaCacheHandle.restore()` when undoing the patch now would break something; nothing is undone.

    reason "generation-in-flight": a generation still has steps to run (`remaining_steps` of `active_num_steps`).
    reason "patched-over": `flux.<attribute>` ("generate_image" or "_predict") was replaced after apply_teacache.
    Either way the handle is unchanged and `restore()` can be called again once the cause is gone."""

    def __init__(
        self,
        *,
        reason: Literal["generation-in-flight", "patched-over"],
        attribute: str | None = None,
        remaining_steps: int | None = None,
        active_num_steps: int | None = None,
    ) -> None:
        if reason == "generation-in-flight":
            msg = (
                f"restore() was called while a generation still had {remaining_steps} of {active_num_steps} steps "
                "to run; call it after generate_image returns or from an after-loop callback. "
                "The handle is unchanged."
            )
        else:
            msg = (
                f"flux.{attribute} was replaced after apply_teacache; restore() would remove that replacement. "
                "Undo it first (restore in reverse order of patching), then call restore() again. "
                "The handle is unchanged."
            )
        super().__init__(msg)
        self.reason = reason
        self.attribute = attribute
        self.remaining_steps = remaining_steps
        self.active_num_steps = active_num_steps

    def __reduce__(self) -> tuple[Any, ...]:
        kwargs = {
            "reason": self.reason,
            "attribute": self.attribute,
            "remaining_steps": self.remaining_steps,
            "active_num_steps": self.active_num_steps,
        }
        return (_rebuild, (type(self), kwargs), self.__dict__)


class InternalStateError(TeaCacheError):
    """Raised when an internal cache/state invariant is violated. Indicates
    a bug in mlx-teacache itself or a defensive guard tripping. Distinct
    from TransformerShapeError (which is about shape drift in real tensors)."""


class TeaCacheWarning(UserWarning):
    """The base class of mlx-teacache's run-time warnings; deprecation notices use
    DeprecationWarning and are not part of this family.

    Filtering it silences the whole family at once:
        warnings.filterwarnings("ignore", category=TeaCacheWarning)
    Each subclass can still be filtered on its own.
    """


class TeaCacheNoBenefitWarning(TeaCacheWarning):
    """Emitted when the gate cannot deliver step-skipping. Two triggers:

    At `apply_teacache()` time, on a distilled variant (FLUX.1 schnell,
    FLUX.2 Klein 4B and 9B) called without your own `coefficients`: its
    few-step schedule leaves the polynomial gate nothing to skip, so the
    wrapper adds per-step overhead and gives no speedup from step-skipping.

    Once per TeaCacheHandle during a generation, when the active step
    configuration leaves no possible cache skips (i.e., `active_num_steps -
    skip_first - skip_last <= 1`, so the polynomial gate cannot seed AND
    consume the cache within this generation). Triggers on very short
    schedules and aggressive skip windows.

    Suppress via the standard `warnings` module:
        warnings.filterwarnings("ignore", category=TeaCacheNoBenefitWarning)
    """


class TeaCacheUncalibratedCheckpointWarning(TeaCacheWarning):
    """Emitted once at apply time, for any variant, when the model's config names
    a checkpoint other than the one the variant's built-in coefficients were
    calibrated on (and does not declare that checkpoint as its base), and the
    caller supplied no coefficients of their own. A config without a model name
    is not checked.

    The polynomial gate predicts how much the transformer body changes per
    step; that prediction was fitted on a specific checkpoint's trajectories.
    A finetune or a newer release under the same alias can move those
    dynamics (FLUX.1 Krea [dev] scores R^2 -496 with FLUX.1-dev's tuple), so
    skip counts and image quality on the other checkpoint are unverified
    until it is calibrated. A local path or pre-quantized mirror of the
    calibrated checkpoint does not warn when it is loaded with that base model
    declared (mflux records it as `base_model`). Pass `coefficients=` from your
    own calibration to silence this, or suppress via the standard `warnings` module:
        warnings.filterwarnings("ignore", category=TeaCacheUncalibratedCheckpointWarning)
    """


class TeaCacheDisabledWarning(TeaCacheWarning):
    """Emitted once at apply time when rel_l1_thresh=0.0 — caching is disabled
    (every step computes; no speedup). Higher threshold = MORE skips; 0.0 disables.

    Suppress via the standard `warnings` module:
        warnings.filterwarnings("ignore", category=TeaCacheDisabledWarning)
    """


class TeaCacheUntestedMfluxWarning(TeaCacheWarning):
    """Emitted once per process by `apply_teacache` when the installed mflux is a later
    release than the newest one this version of mlx-teacache was verified on.

    TeaCache still applies. The `[mflux]` extra sets no upper bound on mflux, because most
    mflux releases change nothing TeaCache relies on. If a generation
    fails or looks wrong on that mflux, please report it with both version numbers at
    https://github.com/IonDen/mlx-teacache/issues.

    Suppress via the standard `warnings` module:
        warnings.filterwarnings("ignore", category=TeaCacheUntestedMfluxWarning)
    """
