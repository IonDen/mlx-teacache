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
from mlx_teacache._kernel.gate import gate_step
from mlx_teacache._kernel.stats import StepDecision, TeaCacheStats
from mlx_teacache.errors import InternalStateError, InvalidStepWindowError, TransformerShapeError
from mlx_teacache.handle import TeaCacheHandle, VariantPatch
from mlx_teacache.integrations.mflux.lifecycle import _active_step_count

from .config import COEFFICIENTS, DEFAULT_THRESH
from .pairing import CfgBranchPairer

_PROVENANCE = Provenance(
    source="builtin",
    revision="in-repo-2026-06-17-origin-signalA-768-50",
    calibration_dataset=(
        "10 prompts (7 fit / 3 held-out) x 50 steps x seed=42, M1 Max 32GB, q4, "
        "768x768, guidance=4.0 (CFG), origin-constrained polyfit, chunked per-prompt"
    ),
    fit_metric=(
        "constrained-LSQ R^2 on consecutive-step Signal-A (modulated block-0 input, "
        "worst-branch body_out) rel-L1 pairs (poly(0)=0)"
    ),
    fit_metric_value=0.8490,
    reference_url="https://github.com/IonDen/mlx-teacache/blob/main/scripts/calibrate_qwen.py",
    default_thresh=DEFAULT_THRESH,
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


def _step_decision_from_gate(decision: Any, *, step_idx: int, timestep: float) -> StepDecision:
    return StepDecision(
        step_idx=step_idx,
        timestep=timestep,
        rel_l1=decision.rel_l1,
        accumulated_distance=decision.accumulated_distance,
        decision=decision.kind,
    )


class _Prelude:
    """Per-step prelude: the image-stream residual base + the timestep-only
    text embedding. Rope + encoder prep are NOT here — they're body-only inputs
    (computed in _qwen_run_body), so a skip step avoids them."""

    __slots__ = ("h_in", "text_embeddings")

    def __init__(self, *, h_in: mx.array, text_embeddings: mx.array) -> None:
        self.h_in = h_in
        self.text_embeddings = text_embeddings


def _qwen_prelude(inner: Any, t: int, config: Any, hidden_states: mx.array) -> _Prelude:
    """img_in + timestep-only text embedding. Mirrors qwen_transformer.py:47-53
    (the img_in + _compute_timestep + time_text_embed portion only)."""
    h_in = inner.img_in(hidden_states)
    timestep = inner._compute_timestep(t, config)
    timestep = mx.broadcast_to(timestep, (h_in.shape[0],)).astype(h_in.dtype)
    text_embeddings = inner.time_text_embed(timestep, h_in)
    return _Prelude(h_in=h_in, text_embeddings=text_embeddings)


def _qwen_signal_a(inner: Any, pre: _Prelude) -> mx.array:
    """FLUX-canonical modulated block-0 image input (the gate signal). Two-stage
    modulation split; _modulate returns (modulated, gate) → take [0].
    qwen_transformer_block.py:36-43,73-76."""
    block0 = inner.transformer_blocks[0]
    img_mod_params = block0.img_mod_linear(block0.img_mod_silu(pre.text_embeddings))
    img_mod1, _img_mod2 = mx.split(img_mod_params, 2, axis=-1)
    img_modulated_0, _gate = block0._modulate(block0.img_norm1(pre.h_in), img_mod1)
    out: mx.array = img_modulated_0
    return out


def _qwen_run_body(
    inner: Any,
    pre: _Prelude,
    *,
    config: Any,
    encoder_hidden_states: mx.array,
    encoder_hidden_states_mask: mx.array,
    cond_image_grid: Any,
) -> mx.array:
    """All 60 dual-stream blocks over the image stream; returns the image stream
    pre-tail (the residual = this − pre.h_in). Encoder prep + rope are computed
    HERE (compute-only) so skip steps don't pay them. Mirrors qwen_transformer.py
    :51-69."""
    encoder = inner.txt_in(inner.txt_norm(encoder_hidden_states))
    image_rotary_embeddings = inner._compute_rotary_embeddings(
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        pos_embed=inner.pos_embed,
        config=config,
        cond_image_grid=cond_image_grid,
    )
    hidden_states = pre.h_in
    for idx, block in enumerate(inner.transformer_blocks):
        encoder, hidden_states = inner._apply_transformer_block(
            idx=idx,
            block=block,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            text_embeddings=pre.text_embeddings,
            image_rotary_embeddings=image_rotary_embeddings,
        )
    return hidden_states


def _qwen_tail(inner: Any, body_out: mx.array, pre: _Prelude) -> mx.array:
    """norm_out + proj_out. Uses only the image stream + timestep text_embeddings
    (the evolved text stream is discarded). qwen_transformer.py:70-71."""
    out = inner.norm_out(body_out, pre.text_embeddings)
    proj: mx.array = inner.proj_out(out)
    return proj


def qwen_forward_with_gate(
    inner: Any,
    handle: Any,
    *,
    t: int,
    config: Any,
    hidden_states: mx.array,
    encoder_hidden_states: mx.array,
    encoder_hidden_states_mask: mx.array,
    qwen_image_ids: Any = None,
    cond_image_grid: Any = None,
) -> Any:
    """Gated proxy forward. Qwen calls the transformer twice per step (positive
    then negative); one shared gate decision (computed on the positive branch's
    branch-independent Signal A) drives both, with two cached residuals."""
    state = handle._state.cache
    stats = handle._state.stats
    pairer = handle._pairer
    pairer.on_generation_token(handle._gen_ctx.token)
    positive = pairer.is_positive()

    # Resolve the active denoising window once — needed by both the skip-window
    # validation below and the gate's forced-window indexing (slow path).
    active_num_steps = handle._gen_ctx.active_num_steps
    if active_num_steps is None:
        active_num_steps = _active_step_count(config)

    # Per-generation skip-window validation (once, on the first call). Mirrors
    # FLUX.1 / FLUX.2 / Z-Image: an over-wide skip window raises rather than
    # silently running at vanilla speed. Reset each generation by the lifecycle's
    # reset_for_new_generation (skip_window_validated=False).
    if state.step_counter == 0 and not state.skip_window_validated:
        if handle.skip_first_n_steps + handle.skip_last_n_steps >= active_num_steps:
            raise InvalidStepWindowError(
                skip_first=handle.skip_first_n_steps,
                skip_last=handle.skip_last_n_steps,
                num_steps=active_num_steps,
                nominal_num_inference_steps=config.num_inference_steps,
            )
        state.skip_window_validated = True

    pre = _qwen_prelude(inner, t, config, hidden_states)

    # thresh<=0: never cache (mirrors gate_step's short-circuit); step recorded+advanced once per step, like the slow path.
    # Fast path: thresh <= 0 ⇒ always compute, never cache. Record + advance
    # ONCE per step (on the positive branch) to keep len(decisions)==num_steps.
    if handle.rel_l1_thresh <= 0.0:
        body_out = _qwen_run_body(
            inner,
            pre,
            config=config,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            cond_image_grid=cond_image_grid,
        )
        if positive:
            # Fast path mirrors the gate's threshold-0 contract: always compute, never cache (see _kernel/gate.py).
            stats._staging.cfg_was_active = True
            stats.record(
                StepDecision(
                    step_idx=state.step_counter,
                    timestep=float(t),
                    rel_l1=None,
                    accumulated_distance=state.accumulated_distance,
                    decision="computed",
                )
            )
        out = _qwen_tail(inner, body_out, pre)
        if not positive:
            state.step_counter += 1
        pairer.advance()
        return out

    if positive:
        stats._staging.cfg_was_active = True
        mod_in = _qwen_signal_a(inner, pre)
        if state.previous_mod_input is not None and mod_in.shape != state.previous_mod_input.shape:
            raise TransformerShapeError(
                step_idx=state.step_counter,
                expected=state.previous_mod_input.shape,
                actual=mod_in.shape,
            )
        decision = gate_step(
            state,
            rel_l1_thresh=handle.rel_l1_thresh,
            coefficients=handle.coefficients,
            skip_first=handle.skip_first_n_steps,
            skip_last=handle.skip_last_n_steps,
            num_steps=active_num_steps,
            step_idx=state.step_counter,
            mod_in=mod_in,
        )
        pairer.shared_decision = decision
        stats.record(_step_decision_from_gate(decision, step_idx=state.step_counter, timestep=float(t)))
        if decision.should_compute:
            body_out = _qwen_run_body(
                inner,
                pre,
                config=config,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                cond_image_grid=cond_image_grid,
            )
            if decision.should_update_cache:
                state.cached_residual = body_out - pre.h_in
                state.previous_mod_input = mod_in
        else:
            if state.cached_residual is None:
                raise InternalStateError(
                    "cached_residual is None on a skipped positive step (qwen); gate logic bug."
                )
            body_out = pre.h_in + state.cached_residual
        out = _qwen_tail(inner, body_out, pre)
        pairer.advance()
        return out

    # Negative branch: reuse the shared decision; cache the NEGATIVE residual.
    decision = pairer.shared_decision
    if decision is None:
        raise InternalStateError("negative branch with no shared decision (qwen pairing bug).")
    if decision.should_compute:
        body_out = _qwen_run_body(
            inner,
            pre,
            config=config,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            cond_image_grid=cond_image_grid,
        )
        if decision.should_update_cache:
            state.cached_residual_neg = body_out - pre.h_in
    else:
        if state.cached_residual_neg is None:
            raise InternalStateError(
                "cached_residual_neg is None on a skipped negative step (qwen); gate logic bug."
            )
        body_out = pre.h_in + state.cached_residual_neg
    out = _qwen_tail(inner, body_out, pre)
    state.step_counter += 1
    pairer.advance()
    return out


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

    try:
        _rollbacks_so_far.append(lambda: _remove_callback_by_identity(flux.callbacks, callback))
        flux.callbacks.register(callback)
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
