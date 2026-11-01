# src/mlx_teacache/integrations/mflux/patch.py
"""Patch installers for mflux pipelines, one per strategy, and the rollbacks they share.

- install_proxy_patch: swaps `flux.transformer` for a proxy first, then registers the lifecycle callback and wraps
  `generate_image` inside one transaction (FLUX.1, Qwen-Image).
- install_predict_patch: registers the callback and wraps `generate_image` first, then assigns `flux._predict` last,
  outside the transaction (FLUX.2 Klein, Z-Image).

Both build the per-apply session from a resolved rule and return the public handle. `wrap_generate_image` is looked
up on the lifecycle module at call time, so patching that module reaches every variant. Imports no mflux.
"""

import contextlib
import weakref
from collections.abc import Callable
from typing import Any

from mlx_teacache._kernel.resolve import ResolvedRule
from mlx_teacache.handle import TeaCacheHandle, VariantPatch
from mlx_teacache.integrations._session import InternalHandle
from mlx_teacache.integrations.mflux import lifecycle


def make_restore_transformer(flux: Any, proxy: Any) -> Callable[[], None]:
    """A rollback that puts the proxy's inner transformer back on `flux` while the proxy is still installed.

    It holds the proxy only weakly, so the proxy in `flux.transformer` is the only strong reference to it and, through
    it, to the original transformer. When mflux's memory saver sets `model.transformer = None` before the decode
    (--low-ram or --pid-decode, for a single seed), both are collected. The slot is restored only while it still holds
    the proxy, so the rollback never brings back a transformer mflux dropped, even if something else still references
    the proxy, and never replaces a transformer the caller installed after apply."""
    proxy_ref = weakref.ref(proxy)

    def _restore_transformer() -> None:
        live = proxy_ref()
        if live is None:
            return
        installed: Any = getattr(flux, "transformer", None)
        if installed is live:
            flux.transformer = live._inner

    return _restore_transformer


def _new_session(rule: ResolvedRule, *, skip_first_n_steps: int, skip_last_n_steps: int) -> InternalHandle:
    return InternalHandle(
        rel_l1_thresh=rule.rel_l1_thresh,
        coefficients=rule.coefficients,
        calibrated_range=rule.calibrated_range,
        skip_first_n_steps=skip_first_n_steps,
        skip_last_n_steps=skip_last_n_steps,
    )


def _register_and_wrap(flux: Any, session: InternalHandle, rollbacks: list[Callable[[], object]]) -> Any:
    """Register the lifecycle callback and wrap `generate_image`; on any failure undo `rollbacks` (plus the
    registration) in reverse and re-raise. Returns the registered callback."""
    callback = lifecycle.GenerationContextCallback(session)
    session._callback_instance = callback
    rollbacks.append(lambda: lifecycle._remove_callback_by_identity(flux.callbacks, callback))
    try:
        flux.callbacks.register(callback)
        lifecycle.wrap_generate_image(flux, session)
    except BaseException:
        for undo in reversed(rollbacks):
            with contextlib.suppress(Exception):
                undo()
        raise
    return callback


def _restore_generate_image(flux: Any, session: InternalHandle) -> Callable[[], None]:
    def _restore() -> None:
        if session._generate_image_was_instance_attr:
            flux.generate_image = session._original_generate_image
        elif "generate_image" in vars(flux):
            del flux.generate_image

    return _restore


def _public_handle(
    flux: Any,
    session: InternalHandle,
    callback: Any,
    rule: ResolvedRule,
    *,
    rollbacks: list[Callable[[], None]],
) -> TeaCacheHandle:
    def _unsubscribe_callback() -> None:
        lifecycle._remove_callback_by_identity(flux.callbacks, callback)

    patch = VariantPatch(rollbacks=rollbacks, finalizers=[_unsubscribe_callback])
    handle = TeaCacheHandle(
        patch=patch,
        stats=session._state.stats,
        provenance=rule.provenance,
        rel_l1_thresh=rule.rel_l1_thresh,
    )
    handle._coefficients = rule.coefficients
    handle._session = session
    handle._callback_instance = session._callback_instance
    return handle


def install_proxy_patch(
    flux: Any,
    *,
    make_proxy: Callable[[Any, InternalHandle], Any],
    rule: ResolvedRule,
    skip_first_n_steps: int,
    skip_last_n_steps: int,
) -> TeaCacheHandle:
    """Swap `flux.transformer` for `make_proxy(inner, session)`, then register and wrap in one transaction.

    A failed registration or wrap puts the original transformer back. `restore()` undoes the swap through a weak
    reference to the proxy (see `make_restore_transformer`) and restores `generate_image`; the callback is
    unsubscribed as a finalizer."""
    session = _new_session(rule, skip_first_n_steps=skip_first_n_steps, skip_last_n_steps=skip_last_n_steps)
    original_transformer = flux.transformer
    # Unwinds mutations made during the install if a later one raises. Distinct from the handle's rollbacks (the
    # restore() teardown set), where the callback unsubscribe is a finalizer.
    rollbacks: list[Callable[[], object]] = []
    proxy = make_proxy(original_transformer, session)
    flux.transformer = proxy
    rollbacks.append(lambda: setattr(flux, "transformer", original_transformer))
    callback = _register_and_wrap(flux, session, rollbacks)
    return _public_handle(
        flux,
        session,
        callback,
        rule,
        rollbacks=[make_restore_transformer(flux, proxy), _restore_generate_image(flux, session)],
    )


def install_predict_patch(
    flux: Any,
    *,
    make_factory: Callable[[InternalHandle], Callable[[Any], Any]],
    rule: ResolvedRule,
    skip_first_n_steps: int,
    skip_last_n_steps: int,
) -> TeaCacheHandle:
    """Register and wrap in one transaction, then assign `flux._predict = make_factory(session)` last.

    mflux calls the factory as `predict = self._predict(self.transformer)` inside `generate_image`. The assignment is
    the last mutation, so it sits outside the transaction. `restore()` puts back (or deletes) the `_predict` instance
    attribute and restores `generate_image`; the callback is unsubscribed as a finalizer."""
    session = _new_session(rule, skip_first_n_steps=skip_first_n_steps, skip_last_n_steps=skip_last_n_steps)
    callback = _register_and_wrap(flux, session, [])
    restore_generate_image = _restore_generate_image(flux, session)

    predict_was_instance_attr = "_predict" in vars(flux)
    original_predict = flux.__dict__.get("_predict")
    flux._predict = make_factory(session)

    def _restore_predict() -> None:
        if predict_was_instance_attr:
            flux._predict = original_predict
        elif "_predict" in vars(flux):
            del flux._predict

    return _public_handle(flux, session, callback, rule, rollbacks=[_restore_predict, restore_generate_image])
