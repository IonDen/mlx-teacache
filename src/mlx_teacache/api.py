# src/mlx_teacache/api.py
"""Public facade. `apply_teacache(flux, ...)` is the entire user-facing API.

Defers mflux imports until the function is actually called, so
`from mlx_teacache import apply_teacache` succeeds even on a machine without
mflux installed. If a caller passes a flux instance without mflux installed,
detect.identify_variant raises IncompatibleModelError with an install hint."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from mlx_teacache.cache import TeaCacheState
from mlx_teacache.coefficients import Provenance, load_builtin, validate_custom
from mlx_teacache.errors import (
    AlreadyPatchedError,
    IncompatibleModelError,
)
from mlx_teacache.stats import TeaCacheStats


@dataclass
class _HandleState:
    cache: TeaCacheState = field(default_factory=TeaCacheState)
    stats: TeaCacheStats = field(default_factory=TeaCacheStats)


@dataclass
class TeaCacheHandle:
    """Returned by apply_teacache. Context-manager-compatible. Holds live stats
    and a restore() method that reverses every mutation from apply_teacache."""
    variant_id: Literal["flux1-dev", "flux1-schnell", "flux2-klein-4b"]
    rel_l1_thresh: float
    coefficients: tuple[float, float, float, float, float]
    skip_first_n_steps: int
    skip_last_n_steps: int
    stats: TeaCacheStats
    provenance: Provenance
    _flux: Any
    _state: _HandleState
    _gen_ctx: Any
    _original_transformer: Any = None
    _original_generate_image: Any = None
    _generate_image_was_instance_attr: bool = False
    _callback_instance: Any = None
    _pending_finalize: Any = None
    _restored: bool = False

    def restore(self) -> None:
        if self._restored:
            return
        flux = self._flux
        # 1. Transformer / _predict reversal.
        if self.variant_id.startswith("flux1-"):
            flux.transformer = self._original_transformer
        else:
            if "_predict" in vars(flux):
                del flux._predict

        # 2. generate_image pristine restore.
        if self._generate_image_was_instance_attr:
            flux.generate_image = self._original_generate_image
        else:
            if "generate_image" in vars(flux):
                del flux.generate_image

        # 3. Callback removal by identity. Warn if registry was replaced.
        import warnings
        cb = self._callback_instance
        registry = getattr(flux, "callbacks", None)
        if registry is not None and cb is not None:
            removed = _remove_callback_by_identity(registry, cb)
            if not removed:
                warnings.warn(
                    "flux.callbacks was replaced or cleared after apply_teacache; "
                    "could not remove our generation callback by identity.",
                    stacklevel=2,
                )

        # 4. Clear sentinel.
        if getattr(flux, "_teacache_handle", None) is self:
            delattr(flux, "_teacache_handle")

        # 5. Discard any in-progress staging stats and freeze stats so further
        #    mutation raises.
        self._state.stats.discard_current_generation()
        self._state.stats._freeze()
        self._restored = True

    def __enter__(self) -> TeaCacheHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.restore()


def _remove_callback_by_identity(registry: Any, target: Any) -> bool:
    """Walk every callback list on the registry and remove `target` by identity.
    Returns True iff at least one removal succeeded. mflux 0.17 stores the
    actual lists on `before_loop` / `in_loop` / `after_loop` / `interrupt`;
    the suffixed names (`*_callbacks`) are methods returning those same lists.
    We try the real list names first, then the suffixed names (for backward
    compat with existing fake-registry test fixtures), then generic fallbacks."""
    removed_any = False
    for attr in ("before_loop", "in_loop", "after_loop", "interrupt",
                 "before_loop_callbacks", "in_loop_callbacks",
                 "after_loop_callbacks", "interrupt_callbacks",
                 "_callbacks", "callbacks"):
        lst = getattr(registry, attr, None)
        if isinstance(lst, list):
            for i in range(len(lst) - 1, -1, -1):
                if lst[i] is target:
                    del lst[i]
                    removed_any = True
    return removed_any


def apply_teacache(
    flux: Any,
    *,
    rel_l1_thresh: float = 0.20,
    coefficients: Sequence[float] | None = None,
    skip_first_n_steps: int = 1,
    skip_last_n_steps: int = 1,
) -> TeaCacheHandle:
    """Enable TeaCache step-skipping on an mflux Flux1 or Flux2Klein instance.

    Supported variants (detected via flux.model_config.model_name):
      - flux1-dev, flux1-schnell
      - flux2-klein-4b

    See docs/superpowers/specs/2026-05-14-mlx-teacache-design.md §6.1 for the
    full docstring; this is the runtime entry point."""
    # Eager static validation.
    if not (0.0 <= rel_l1_thresh <= 1.0):
        raise ValueError(f"rel_l1_thresh must be in [0.0, 1.0], got {rel_l1_thresh}")
    if skip_first_n_steps < 0:
        raise ValueError(f"skip_first_n_steps must be >= 0, got {skip_first_n_steps}")
    if skip_last_n_steps < 0:
        raise ValueError(f"skip_last_n_steps must be >= 0, got {skip_last_n_steps}")

    # Deferred mflux imports.
    from mlx_teacache.integrations.mflux.detect import identify_variant
    from mlx_teacache.integrations.mflux.flux1 import ProxyFlux1Transformer
    from mlx_teacache.integrations.mflux.flux2 import make_teacache_predict_factory
    from mlx_teacache.integrations.mflux.lifecycle import (
        GenerationContext,
        GenerationContextCallback,
        wrap_generate_image,
    )

    variant_id = identify_variant(flux)

    # Coefficient resolution.
    if coefficients is not None:
        coeffs = validate_custom(coefficients)
        prov = Provenance.for_user_supplied()
    else:
        coeffs, prov = load_builtin(variant_id)

    # Already-patched sentinel check.
    existing = getattr(flux, "_teacache_handle", None)
    if existing is not None:
        raise AlreadyPatchedError(
            variant_id=existing.variant_id, rel_l1_thresh=existing.rel_l1_thresh,
        )
    # Per audit Low #9: FLUX.1 secondary cross-check — proxy present but sentinel
    # missing indicates inconsistent state from a partial earlier patch.
    if variant_id.startswith("flux1-") and isinstance(flux.transformer, ProxyFlux1Transformer):
        from mlx_teacache.errors import InternalStateError
        raise InternalStateError(
            "flux.transformer is a ProxyFlux1Transformer but flux has no "
            "_teacache_handle sentinel. This indicates an inconsistent patch "
            "state (e.g., a partial apply that didn't complete). "
            "If you have a reference to the original handle, call handle.restore(); "
            "otherwise instantiate a fresh Flux1 model."
        )
    # Defensive: detect non-TeaCache _predict overrides on FLUX.2.
    if variant_id == "flux2-klein-4b" and "_predict" in vars(flux):
        raise IncompatibleModelError(
            actual_type=type(flux).__name__,
            actual_model_name=getattr(flux.model_config, "model_name", None),
            supported=["flux1-dev", "flux1-schnell", "flux2-klein-4b"],
        )

    # Build handle.
    handle = TeaCacheHandle(
        variant_id=variant_id,
        rel_l1_thresh=rel_l1_thresh,
        coefficients=coeffs,
        skip_first_n_steps=skip_first_n_steps,
        skip_last_n_steps=skip_last_n_steps,
        stats=TeaCacheStats(),  # replaced below by _state.stats
        provenance=prov,
        _flux=flux,
        _state=_HandleState(),
        _gen_ctx=GenerationContext(),
    )
    # Wire stats: handle.stats is the same object as handle._state.stats so
    # the public API exposes the live counters.
    handle.stats = handle._state.stats

    # --- transactional patch (per audit medium #3) ---
    rollback: list[Callable[[], None]] = []
    try:
        # Step A: register the lifecycle callback.
        if not hasattr(flux.callbacks, "register"):
            raise IncompatibleModelError(
                actual_type=type(flux).__name__,
                actual_model_name=getattr(getattr(flux, "model_config", None), "model_name", None),
                supported=["flux1-dev", "flux1-schnell", "flux2-klein-4b"],
            )
        callback = GenerationContextCallback(handle)
        handle._callback_instance = callback
        flux.callbacks.register(callback)
        def _rollback_callback() -> None:
            _remove_callback_by_identity(flux.callbacks, callback)
        rollback.append(_rollback_callback)

        # Step B: wrap generate_image (records _generate_image_was_instance_attr).
        wrap_generate_image(flux, handle)
        def _rollback_generate_image() -> None:
            if handle._generate_image_was_instance_attr:
                flux.generate_image = handle._original_generate_image
            else:
                if "generate_image" in vars(flux):
                    del flux.generate_image
        rollback.append(_rollback_generate_image)

        # Step C: patch transformer / _predict.
        if variant_id.startswith("flux1-"):
            handle._original_transformer = flux.transformer
            flux.transformer = ProxyFlux1Transformer(flux.transformer, handle)
            rollback.append(lambda: setattr(flux, "transformer", handle._original_transformer))
        else:
            flux._predict = make_teacache_predict_factory(handle)
            def _rollback_predict() -> None:
                if "_predict" in vars(flux):
                    del flux._predict
            rollback.append(_rollback_predict)

        # Step D: set sentinel.
        flux._teacache_handle = handle
        def _rollback_sentinel() -> None:
            if getattr(flux, "_teacache_handle", None) is handle:
                delattr(flux, "_teacache_handle")
        rollback.append(_rollback_sentinel)
    except BaseException:
        # Reverse every applied mutation, swallowing rollback failures so we
        # don't mask the original exception.
        for undo in reversed(rollback):
            with contextlib.suppress(Exception):
                undo()
        raise

    return handle
