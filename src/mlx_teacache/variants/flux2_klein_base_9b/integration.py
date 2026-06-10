"""FLUX.2 Klein base 9B integration. Reuses the FLUX.2 forward + factory
verbatim from flux2_klein_base_4b. The 9B variant uses the SAME forward
path (both no-CFG at low_step recipe and CFG-per-branch at the canonical
50-step + g=4.0 recipe) — only the metadata (Provenance) and memory_cap
hint differ.
"""

from __future__ import annotations

from typing import Any

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache.handle import TeaCacheHandle, VariantPatch
from mlx_teacache.integrations.mflux.lifecycle import (
    GenerationContextCallback,
    wrap_generate_image,
)

# Cross-import everything heavy from base_4b. The factory + forward + helpers
# are byte-equivalent; only the per-variant constants (COEFFICIENTS,
# DEFAULT_THRESH) and provenance change.
from mlx_teacache.variants.flux1_dev.integration import _InternalHandle
from mlx_teacache.variants.flux2_klein_base_4b.integration import (
    make_teacache_predict_factory,
)

from .config import COEFFICIENTS, DEFAULT_THRESH

_PROVENANCE = Provenance(
    source="builtin",
    revision="in-repo-2026-05-18-reuse-base-4b",
    calibration_dataset=(
        "REUSED from flux2-klein-base-4b — same architecture family + same recipe "
        "(10 prompts × 25 steps × seed=42, M1 Max 32GB, bf16, 512x512, guidance=1.0, "
        "origin-constrained polyfit); validated empirically at 50 steps + guidance=4.0"
    ),
    fit_metric="constrained-LSQ R^2 on consecutive-step (mod_in, body_out) rel-L1 pairs (poly(0)=0)",
    fit_metric_value=0.10643408169124158,
    reference_url="https://github.com/IonDen/mlx-teacache/blob/main/scripts/validate_klein_base_9b.py",
    default_thresh=0.17,
)


def apply(
    flux: Any,
    *,
    rel_l1_thresh: float | None = None,
    coefficients: tuple[float, float, float, float, float] | None = None,
    skip_first_n_steps: int = 1,
    skip_last_n_steps: int = 1,
) -> TeaCacheHandle:
    """FLUX.2 Klein base 9B apply. Mirrors flux2_klein_base_4b.apply() but
    with this variant's COEFFICIENTS, DEFAULT_THRESH, and _PROVENANCE."""
    # 1. Resolve rel_l1_thresh.
    # Priority: explicit caller > per-variant DEFAULT_THRESH (only when using
    # builtin coefficients) > package fallback 0.20.
    if rel_l1_thresh is not None:
        resolved_thresh: float = rel_l1_thresh
    elif coefficients is None and DEFAULT_THRESH is not None:
        resolved_thresh = DEFAULT_THRESH
    else:
        resolved_thresh = 0.20

    # 2. Resolve coefficients and provenance (caller > COEFFICIENTS).
    # User-supplied coefficients get a user provenance; builtin get _PROVENANCE.
    if coefficients is not None:
        resolved_coeffs: tuple[float, float, float, float, float] = coefficients
        resolved_provenance = Provenance.for_user_supplied()
    else:
        resolved_coeffs = COEFFICIENTS
        resolved_provenance = _PROVENANCE

    # 3. Build internal handle (carries state, gen context, and per-generation
    #    fields that lifecycle.py and the forward block reference).
    internal = _InternalHandle(
        rel_l1_thresh=resolved_thresh,
        coefficients=resolved_coeffs,
        skip_first_n_steps=skip_first_n_steps,
        skip_last_n_steps=skip_last_n_steps,
    )

    import contextlib

    # 4. Register lifecycle callback.
    callback = GenerationContextCallback(internal)
    internal._callback_instance = callback
    flux.callbacks.register(callback)

    # Eager rollback list for the transactional patch (per audit medium #3):
    # if any mutation after callback registration raises, preceding mutations
    # are reversed. Start with the callback unregister.
    from mlx_teacache.integrations.mflux.lifecycle import _remove_callback_by_identity

    _rollbacks_so_far: list[Any] = [lambda: _remove_callback_by_identity(flux.callbacks, callback)]

    # 5. Wrap generate_image (records _generate_image_was_instance_attr, sets
    #    internal._original_generate_image).
    try:
        wrap_generate_image(flux, internal)
    except BaseException:
        for _undo in reversed(_rollbacks_so_far):
            with contextlib.suppress(Exception):
                _undo()
        raise

    # wrap_generate_image succeeded — add its rollback before the next mutation.
    def _restore_generate_image() -> None:
        if internal._generate_image_was_instance_attr:
            flux.generate_image = internal._original_generate_image
        else:
            if "generate_image" in vars(flux):
                del flux.generate_image

    _rollbacks_so_far.append(_restore_generate_image)

    # 6. Patch flux._predict with the factory. FLUX.2 uses _predict replacement,
    #    NOT flux.transformer — the factory is called as
    #    predict = self._predict(self.transformer) inside generate_image.
    flux._predict = make_teacache_predict_factory(internal)
    # No try needed: _predict assignment is the last mutation; fall through.

    # 7. Build VariantPatch: rollback deletes _predict + restores generate_image.
    #    Finalizer unsubscribes the callback. NO stats finalize (audit F2).
    def _restore_predict() -> None:
        if "_predict" in vars(flux):
            del flux._predict

    def _unsubscribe_callback() -> None:
        from mlx_teacache.integrations.mflux.lifecycle import _remove_callback_by_identity as _rcbi

        _rcbi(flux.callbacks, callback)

    patch = VariantPatch(
        rollbacks=[_restore_predict, _restore_generate_image],
        finalizers=[_unsubscribe_callback],
    )

    # 8. Return public TeaCacheHandle (variant-agnostic, audit F3).
    handle = TeaCacheHandle(
        patch=patch,
        stats=internal._state.stats,
        provenance=resolved_provenance,
        rel_l1_thresh=resolved_thresh,
    )
    # Expose resolved coefficients and callback instance as dynamic attributes
    # so callers and tests can inspect them (mirrors v0.5.x TeaCacheHandle).
    handle.coefficients = resolved_coeffs  # type: ignore[attr-defined]
    handle._callback_instance = internal._callback_instance  # type: ignore[attr-defined]
    return handle
