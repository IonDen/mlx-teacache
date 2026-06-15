"""Qwen-Image integration — proxy-transformer TeaCache. mflux imported lazily
(registry loads this only after detect.matches() wins).

Qwen has no `_predict` and no `mx.compile`: we proxy `flux.transformer` (FLUX.1
pattern). generate_image calls the transformer TWICE per step (positive then
negative, external CFG combine), so the forward threads a CfgBranchPairer.
"""

from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

from mlx_teacache._kernel.cache import TeaCacheState
from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.stats import TeaCacheStats
from mlx_teacache.handle import TeaCacheHandle, VariantPatch

from .config import COEFFICIENTS, DEFAULT_THRESH
from .pairing import CfgBranchPairer

_PROVENANCE = Provenance(
    source="builtin",
    revision="PLACEHOLDER-pre-calibration",  # replaced when calibration lands
    calibration_dataset="PENDING in-repo calibration (scripts/calibrate_qwen.py)",
    reference_url="https://github.com/IonDen/mlx-teacache/blob/main/scripts/calibrate_qwen.py",
)


class _InternalHandleState:
    def __init__(self) -> None:
        self.cache: TeaCacheState = TeaCacheState()
        self.stats: TeaCacheStats = TeaCacheStats()
        self.no_benefit_warned: bool = False


class _GenerationContext:
    def __init__(self) -> None:
        self.token: int = 0
        self.active_num_steps: int | None = None
        self.consumed_at_token: int | None = None


class _InternalHandle:
    def __init__(
        self,
        *,
        rel_l1_thresh: float,
        coefficients: tuple[float, float, float, float, float],
        skip_first_n_steps: int,
        skip_last_n_steps: int,
    ) -> None:
        self._state = _InternalHandleState()
        self._gen_ctx = _GenerationContext()
        self._pairer = CfgBranchPairer()
        self.rel_l1_thresh = rel_l1_thresh
        self.coefficients = coefficients
        self.skip_first_n_steps = skip_first_n_steps
        self.skip_last_n_steps = skip_last_n_steps
        self._generate_image_was_instance_attr: bool = False
        self._original_generate_image: Any = None
        self._pending_finalize: Any = None
        self._callback_instance: Any = None


class ProxyQwenTransformer(nn.Module):  # type: ignore[misc,name-defined]
    def __init__(self, inner: Any, handle: Any) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_handle", handle)

    def __call__(
        self,
        *,
        t: int,
        config: Any,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
        qwen_image_ids: Any = None,
        cond_image_grid: Any = None,
    ) -> Any:
        return qwen_forward_with_gate(
            self._inner,
            self._handle,
            t=t,
            config=config,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            qwen_image_ids=qwen_image_ids,
            cond_image_grid=cond_image_grid,
        )

    def freeze(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.freeze(*args, **kwargs)

    def parameters(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._inner.parameters())

    def trainable_parameters(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._inner.trainable_parameters())

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = object.__getattribute__(self, "_inner")
            return getattr(inner, name)


def qwen_forward_with_gate(inner: Any, handle: Any, **call_kwargs: Any) -> Any:
    raise NotImplementedError("qwen_forward_with_gate is implemented in a later task")


def apply(
    flux: Any,
    *,
    rel_l1_thresh: float | None = None,
    coefficients: tuple[float, float, float, float, float] | None = None,
    skip_first_n_steps: int = 1,
    skip_last_n_steps: int = 1,
) -> TeaCacheHandle:
    """Qwen-Image apply. Proxies flux.transformer; lifecycle-callback driven."""
    if rel_l1_thresh is not None:
        resolved_thresh: float = rel_l1_thresh
    elif coefficients is None and DEFAULT_THRESH is not None:
        resolved_thresh = DEFAULT_THRESH
    else:
        resolved_thresh = 0.20

    if coefficients is not None:
        resolved_coeffs: tuple[float, float, float, float, float] = coefficients
        resolved_provenance = Provenance.for_user_supplied()
    else:
        resolved_coeffs = COEFFICIENTS
        resolved_provenance = _PROVENANCE

    import contextlib

    internal = _InternalHandle(
        rel_l1_thresh=resolved_thresh,
        coefficients=resolved_coeffs,
        skip_first_n_steps=skip_first_n_steps,
        skip_last_n_steps=skip_last_n_steps,
    )

    original_transformer = flux.transformer
    # Eager rollback list: unwinds mutations made DURING apply() if a later
    # mutation raises. Distinct from VariantPatch.rollbacks below (the clean
    # restore() teardown set) — the callback unsubscribe is a finalizer there,
    # not a rollback. Do not merge the two lists. (Matches FLUX.1.)
    _rollbacks_so_far: list[Any] = []

    proxy = ProxyQwenTransformer(inner=original_transformer, handle=internal)
    flux.transformer = proxy
    _rollbacks_so_far.append(lambda: setattr(flux, "transformer", original_transformer))

    from mlx_teacache.integrations.mflux import lifecycle as _lifecycle
    from mlx_teacache.integrations.mflux.lifecycle import (
        GenerationContextCallback,
        _remove_callback_by_identity,
    )

    callback = GenerationContextCallback(internal)
    internal._callback_instance = callback
    flux.callbacks.register(callback)
    _rollbacks_so_far.append(lambda: _remove_callback_by_identity(flux.callbacks, callback))

    try:
        _lifecycle.wrap_generate_image(flux, internal)
    except BaseException:
        for _undo in reversed(_rollbacks_so_far):
            with contextlib.suppress(Exception):
                _undo()
        raise

    def _restore_transformer() -> None:
        flux.transformer = original_transformer

    def _restore_generate_image() -> None:
        if internal._generate_image_was_instance_attr:
            flux.generate_image = internal._original_generate_image
        elif "generate_image" in vars(flux):
            del flux.generate_image

    def _unsubscribe_callback() -> None:
        _remove_callback_by_identity(flux.callbacks, callback)

    patch = VariantPatch(
        rollbacks=[_restore_transformer, _restore_generate_image],
        finalizers=[_unsubscribe_callback],
    )
    handle = TeaCacheHandle(
        patch=patch,
        stats=internal._state.stats,
        provenance=resolved_provenance,
        rel_l1_thresh=resolved_thresh,
    )
    handle.coefficients = resolved_coeffs  # type: ignore[attr-defined]
    handle._callback_instance = internal._callback_instance  # type: ignore[attr-defined]
    return handle
