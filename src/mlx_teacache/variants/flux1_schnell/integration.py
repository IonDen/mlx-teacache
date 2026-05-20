"""FLUX.1 schnell integration. Reuses FLUX.1 dev's proxy + forward
verbatim; same transformer architecture. Only the public-facing
metadata (provenance) and the apply() defaults differ.
"""
from __future__ import annotations

from typing import Any

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache.handle import TeaCacheHandle, VariantPatch
from mlx_teacache.integrations.mflux.lifecycle import wrap_generate_image

# Reuse the verbatim port from flux1_dev — identical forward code.
from mlx_teacache.variants.flux1_dev.integration import (
    ProxyFlux1Transformer,
    _InternalHandle,
)

from .config import COEFFICIENTS, DEFAULT_THRESH

_PROVENANCE = Provenance(
    source="builtin",
    revision="upstream-flux-v1-shared",
    calibration_dataset="upstream ali-vilab TeaCache (FLUX architecture is shared between dev and schnell)",
    reference_url="https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py",
)


def apply(
    flux: Any,
    *,
    rel_l1_thresh: float | None = None,
    coefficients: tuple[float, float, float, float, float] | None = None,
    skip_first_n_steps: int = 1,
    skip_last_n_steps: int = 1,
) -> TeaCacheHandle:
    """FLUX.1 schnell apply. Public-API-equivalent of the FLUX.1 schnell
    branch of v0.5.x apply_teacache."""
    # 1. Resolve rel_l1_thresh (caller > DEFAULT_THRESH).
    resolved_thresh: float = rel_l1_thresh if rel_l1_thresh is not None else DEFAULT_THRESH

    # 2. Resolve coefficients (caller > COEFFICIENTS).
    resolved_coeffs: tuple[float, float, float, float, float] = coefficients if coefficients is not None else COEFFICIENTS

    # 3. Build internal handle (carries state, gen context, and per-generation
    #    fields that lifecycle.py and the forward block reference).
    internal = _InternalHandle(
        rel_l1_thresh=resolved_thresh,
        coefficients=resolved_coeffs,
        skip_first_n_steps=skip_first_n_steps,
        skip_last_n_steps=skip_last_n_steps,
    )

    # 4. Save original transformer for rollback.
    original_transformer = flux.transformer

    # 5. Build proxy and swap onto flux.transformer.
    proxy = ProxyFlux1Transformer(inner=original_transformer, handle=internal)
    flux.transformer = proxy

    # 6. Register lifecycle callback via wrap_generate_image. This registers
    #    GenerationContextCallback and wraps flux.generate_image.
    from mlx_teacache.integrations.mflux.lifecycle import GenerationContextCallback

    callback = GenerationContextCallback(internal)
    internal._callback_instance = callback
    flux.callbacks.register(callback)
    wrap_generate_image(flux, internal)

    # 7. Build VariantPatch: rollback restores transformer + unsubscribes the
    #    callback + restores generate_image. NO stats finalize call (audit F2).
    def _restore_transformer() -> None:
        flux.transformer = original_transformer

    def _unsubscribe_callback() -> None:
        from mlx_teacache.api import _remove_callback_by_identity
        _remove_callback_by_identity(flux.callbacks, callback)

    def _restore_generate_image() -> None:
        if internal._generate_image_was_instance_attr:
            flux.generate_image = internal._original_generate_image
        else:
            if "generate_image" in vars(flux):
                del flux.generate_image

    patch = VariantPatch(
        rollbacks=[_restore_transformer, _restore_generate_image],
        finalizers=[_unsubscribe_callback],
    )

    # 8. Return public TeaCacheHandle (variant-agnostic, audit F3).
    return TeaCacheHandle(
        patch=patch,
        stats=internal._state.stats,
        provenance=_PROVENANCE,
        rel_l1_thresh=resolved_thresh,
    )
