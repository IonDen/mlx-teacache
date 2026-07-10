# src/mlx_teacache/api.py
"""Public entry point. Variant dispatch via _REGISTRY.

The 4-kwarg public signature (rel_l1_thresh, coefficients,
skip_first_n_steps, skip_last_n_steps) is preserved exactly from v0.5.x.
Each variant's apply() accepts all four; the dispatcher forwards them.
"""

import warnings
from collections.abc import Sequence
from typing import Any

from mlx_teacache._kernel.coefficients import validate_custom
from mlx_teacache.errors import IncompatibleModelError, TeaCacheDisabledWarning, TeaCacheValueError
from mlx_teacache.handle import TeaCacheHandle
from mlx_teacache.variants import _REGISTRY


def apply_teacache(
    flux: Any,
    *,
    rel_l1_thresh: float | None = None,
    coefficients: Sequence[float] | None = None,
    skip_first_n_steps: int = 1,
    skip_last_n_steps: int = 1,
) -> TeaCacheHandle:
    """Enable TeaCache step-skipping on an mflux Flux1 / Flux2Klein instance.

    Walks the variant registry; the first variant whose matches(flux) returns
    True wins, then its integration module is loaded lazily and dispatched with
    all four public kwargs.

    When rel_l1_thresh is left as None it resolves to the variant's default:
      - flux1-dev, flux1-schnell ......... 0.20 (the package fallback)
      - flux2-klein-base-4b, -base-9b .... 0.17
      - z-image-base ..................... 0.12
      - qwen-image ....................... 0.30
      - flux2-klein-4b, flux2-klein-9b ... no per-variant default; fall back to
        0.20 (these distilled 4-8 step schedules skip 0 steps at any reasonable
        threshold — see the "When to use" section of the README).
    Pass rel_l1_thresh=<float> to override. The resolved effective threshold is
    available afterwards as handle.rel_l1_thresh.

    Higher rel_l1_thresh = more steps skipped (larger speedup, some quality
    trade-off). rel_l1_thresh=0.0 disables caching entirely — every step
    computes, no speedup is gained, and a TeaCacheDisabledWarning is emitted.

    coefficients: any 5-element sequence of finite floats (list, tuple, etc.),
    coerced to a tuple before dispatch. nan or inf raises TeaCacheValueError.

    Returns a TeaCacheHandle (context-manager compatible; handle.restore()
    undoes the patch)."""
    # --- Static validation (model-independent) ---
    if skip_first_n_steps < 0:
        raise TeaCacheValueError(f"skip_first_n_steps must be >= 0, got {skip_first_n_steps}")
    if skip_last_n_steps < 0:
        raise TeaCacheValueError(f"skip_last_n_steps must be >= 0, got {skip_last_n_steps}")
    if coefficients is not None:
        coefficients = validate_custom(coefficients)
    if rel_l1_thresh is not None and not (0.0 <= rel_l1_thresh <= 1.0):
        raise TeaCacheValueError(f"rel_l1_thresh must be in [0.0, 1.0], got {rel_l1_thresh}")
    if rel_l1_thresh == 0.0:
        warnings.warn(
            "rel_l1_thresh=0.0 disables TeaCache caching (every step computes; "
            "no speedup). Higher threshold = more skips. Pass a positive value to enable.",
            TeaCacheDisabledWarning,
            stacklevel=2,
        )

    # --- Already-patched sentinel check ---
    existing = getattr(flux, "_teacache_handle", None)
    if existing is not None:
        from mlx_teacache.errors import AlreadyPatchedError

        raise AlreadyPatchedError(
            variant_id=getattr(existing, "variant_id", "unknown"),
            rel_l1_thresh=existing.rel_l1_thresh,
        )

    for variant_id, entry in _REGISTRY.items():
        if entry["matches"](flux):
            apply = entry["load_integration"]()
            handle: TeaCacheHandle = apply(
                flux,
                rel_l1_thresh=rel_l1_thresh,
                coefficients=coefficients,
                skip_first_n_steps=skip_first_n_steps,
                skip_last_n_steps=skip_last_n_steps,
            )
            # Attach dispatcher-level metadata. TeaCacheHandle is a regular
            # class, so dynamic attributes can be set without modifying handle.py.
            handle.variant_id = variant_id  # type: ignore[attr-defined]

            # Set the sentinel so AlreadyPatchedError fires on double-apply.
            # Clear it only after every teardown action succeeds, preventing a
            # re-apply from nesting on a half-restored model.
            flux._teacache_handle = handle

            def _clear_sentinel(_flux: Any = flux, _handle: Any = handle) -> None:
                if getattr(_flux, "_teacache_handle", None) is _handle:
                    delattr(_flux, "_teacache_handle")

            handle._patch.on_restored.append(_clear_sentinel)
            return handle

    model_config = getattr(flux, "model_config", None)
    model_name = getattr(model_config, "model_name", None)
    raise IncompatibleModelError(
        actual_type=type(flux).__name__,
        actual_model_name=model_name,
        supported=sorted(_REGISTRY.keys()),
    )
