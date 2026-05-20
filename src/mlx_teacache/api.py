# src/mlx_teacache/api.py
"""Public entry point. Variant dispatch via _REGISTRY.

The 4-kwarg public signature (rel_l1_thresh, coefficients,
skip_first_n_steps, skip_last_n_steps) is preserved exactly from v0.5.x.
Each variant's apply() accepts all four; the dispatcher forwards them.
"""

from typing import Any

from mlx_teacache.errors import IncompatibleModelError
from mlx_teacache.handle import TeaCacheHandle
from mlx_teacache.variants import _REGISTRY


def apply_teacache(
    flux: Any,
    *,
    rel_l1_thresh: float | None = None,
    coefficients: tuple[float, float, float, float, float] | None = None,
    skip_first_n_steps: int = 1,
    skip_last_n_steps: int = 1,
) -> TeaCacheHandle:
    """Enable TeaCache step-skipping. Walks the variant registry; the
    first variant whose matches(flux) returns True wins. Loads that
    variant's integration module lazily and dispatches with all four
    public kwargs."""
    # --- Static validation (model-independent) ---
    if skip_first_n_steps < 0:
        raise ValueError(f"skip_first_n_steps must be >= 0, got {skip_first_n_steps}")
    if skip_last_n_steps < 0:
        raise ValueError(f"skip_last_n_steps must be >= 0, got {skip_last_n_steps}")
    if coefficients is not None and len(coefficients) != 5:
        raise ValueError(
            f"coefficients must have length 5, got {len(coefficients)}"
        )
    if rel_l1_thresh is not None and not (0.0 <= rel_l1_thresh <= 1.0):
        raise ValueError(f"rel_l1_thresh must be in [0.0, 1.0], got {rel_l1_thresh}")

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

            # Set the sentinel on flux so AlreadyPatchedError fires on
            # double-apply, and register a rollback that clears it when
            # restore() runs. Rollbacks execute in reverse; appending here
            # means this rollback runs FIRST (after all variant rollbacks).
            flux._teacache_handle = handle

            def _clear_sentinel(
                _flux: Any = flux, _handle: Any = handle
            ) -> None:
                if getattr(_flux, "_teacache_handle", None) is _handle:
                    delattr(_flux, "_teacache_handle")

            handle._patch.rollbacks.append(_clear_sentinel)
            return handle

    model_config = getattr(flux, "model_config", None)
    model_name = getattr(model_config, "model_name", None)
    raise IncompatibleModelError(
        actual_type=type(flux).__name__,
        actual_model_name=model_name,
        supported=sorted(_REGISTRY.keys()),
    )
