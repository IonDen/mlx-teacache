"""FLUX.1 dev integration. Byte-for-byte port from v0.5.x:
- src/mlx_teacache/integrations/mflux/flux1.py::ProxyFlux1Transformer
- src/mlx_teacache/integrations/mflux/forward.py FLUX.1 forward block
- src/mlx_teacache/api.py::apply_teacache FLUX.1 branch

mflux is imported only inside this module. The package registry loads
this lazily, after detect.matches() wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

from mlx_teacache._kernel.cache import TeaCacheState
from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.gate import gate_step
from mlx_teacache._kernel.stats import StepDecision, TeaCacheStats
from mlx_teacache.handle import TeaCacheHandle, VariantPatch
from mlx_teacache.integrations.mflux.lifecycle import _active_step_count

from .config import COEFFICIENTS, DEFAULT_THRESH

_PROVENANCE = Provenance(
    source="builtin",
    revision="upstream-flux-v1",
    calibration_dataset="upstream ali-vilab TeaCache (no in-repo calibration)",
    reference_url="https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py",
)


# ----- Internal handle shape (mirrors v0.5.x TeaCacheHandle fields used by
#       forward.py and lifecycle.py). Not part of the public API. -----


@dataclass
class _InternalHandleState:
    cache: TeaCacheState = field(default_factory=TeaCacheState)
    stats: TeaCacheStats = field(default_factory=TeaCacheStats)
    no_benefit_warned: bool = False


@dataclass
class _GenerationContext:
    token: int = 0
    active_num_steps: int | None = None
    consumed_at_token: int | None = None


class _InternalHandle:
    """Duck-type-compatible with the v0.5.x handle fields that forward.py and
    lifecycle.py reference. Not returned to the user."""

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
        self.rel_l1_thresh = rel_l1_thresh
        self.coefficients = coefficients
        self.skip_first_n_steps = skip_first_n_steps
        self.skip_last_n_steps = skip_last_n_steps
        # Fields set by wrap_generate_image
        self._generate_image_was_instance_attr: bool = False
        self._original_generate_image: Any = None
        self._pending_finalize: Any = None
        # Field set by GenerationContextCallback registration
        self._callback_instance: Any = None


# ----- PORTED VERBATIM from src/mlx_teacache/integrations/mflux/flux1.py -----


class ProxyFlux1Transformer(nn.Module):  # type: ignore[misc,name-defined]
    def __init__(self, inner: Any, handle: Any) -> None:
        super().__init__()
        # Store inner under a leading-underscore name. This makes _inner
        # accessible via attribute lookup and our delegated methods, but
        # MLX's valid_parameter_filter excludes underscore-prefixed keys,
        # so parent-level flux.parameters() does NOT recurse into _inner.
        # We rely on mflux calling flux.transformer.parameters() (the
        # proxy's overridden method) directly during save, which delegates
        # to inner.parameters() correctly. See limitations note below.
        # Use object.__setattr__ to bypass nn.Module.__setattr__, which calls
        # hasattr(self, key) -> __getattr__ -> getattr(self._inner, ...) before
        # _inner is set, causing infinite recursion. Both _inner and _handle are
        # stored directly on the instance __dict__ this way.
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_handle", handle)

    def __call__(
        self,
        t: int,
        config: Any,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        **kwargs: Any,
    ) -> Any:
        return flux1_forward_with_gate(
            self._inner,
            self._handle,
            t=t,
            config=config,
            hidden_states=hidden_states,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            **kwargs,
        )

    # --- explicit method delegation for mflux-used ops ---
    # MLX's nn.Module exposes parameters() and trainable_parameters() as
    # methods (verified against mlx/nn/layers/base.py), not properties.

    def freeze(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.freeze(*args, **kwargs)

    def parameters(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._inner.parameters())

    def trainable_parameters(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._inner.trainable_parameters())

    def __getattr__(self, name: str) -> Any:
        # nn.Module's __getattr__ handles dict children + parameters. Fall
        # back to the inner module for anything else (x_embedder,
        # transformer_blocks, time_text_embed, etc.).
        # Use object.__getattribute__ to fetch _inner to avoid re-entering
        # __getattr__ if _inner hasn't been set yet.
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = object.__getattribute__(self, "_inner")
            return getattr(inner, name)


# ----- PORTED VERBATIM from src/mlx_teacache/integrations/mflux/forward.py -----
# FLUX.1 block only.

from mlx_teacache.errors import (  # noqa: E402
    InternalStateError,
    InvalidStepWindowError,
    TransformerShapeError,
)


def _step_decision_from_gate(decision: Any, *, step_idx: int, timestep: float) -> StepDecision:
    return StepDecision(
        step_idx=step_idx,
        timestep=timestep,
        rel_l1=decision.rel_l1,
        accumulated_distance=decision.accumulated_distance,
        decision=decision.kind,
    )


def _flux1_extract_mod_input(block_0: Any, hidden_states_pre: mx.array, text_embeddings: mx.array) -> Any:
    """Extract the modulated block-0 input — the TeaCache gating signal.
    For FLUX.1's JointTransformerBlock, the gating signal is the output of
    the AdaLayerNormZero modulation applied to hidden_states_pre. We extract
    it without running the attention to keep the gating cost ~µs.

    mflux JointTransformerBlock has .norm1: AdaLayerNormZero. Its __call__
    returns (norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp);
    norm_hidden_states is what we use as mod_in."""
    # AdaLayerNormZero.__call__(hidden_states, text_embeddings) — positional,
    # no `emb=` keyword (mflux 0.17).
    return block_0.norm1(hidden_states_pre, text_embeddings)[0]


def _flux1_run_body(
    inner: Any,
    body_in: mx.array,
    encoder_hidden_states: mx.array,
    text_embeddings: mx.array,
    image_rotary_embeddings: mx.array,
    kwargs: dict[str, Any],
) -> Any:
    """Run all joint transformer blocks, concat, then all single transformer
    blocks. Returns the post-concat post-single-blocks tensor that the tail
    operates on. Mirrors mflux Transformer.__call__ lines 50-74."""
    hidden_states = body_in
    for idx, block in enumerate(inner.transformer_blocks):
        encoder_hidden_states, hidden_states = inner._apply_joint_transformer_block(
            idx=idx,
            block=block,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            text_embeddings=text_embeddings,
            image_rotary_embeddings=image_rotary_embeddings,
            controlnet_block_samples=kwargs.get("controlnet_block_samples"),
        )
    hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
    for idx, block in enumerate(inner.single_transformer_blocks):
        hidden_states = inner._apply_single_transformer_block(
            idx=idx,
            block=block,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            text_embeddings=text_embeddings,
            image_rotary_embeddings=image_rotary_embeddings,
            controlnet_single_block_samples=kwargs.get("controlnet_single_block_samples"),
        )
    return hidden_states


def flux1_forward_with_gate(
    inner: Any,
    handle: Any,
    *,
    t: int,
    config: Any,
    hidden_states: mx.array,
    prompt_embeds: mx.array,
    pooled_prompt_embeds: mx.array,
    **kwargs: Any,
) -> Any:
    """Replacement for mflux.models.flux.model.flux_transformer.transformer.Transformer.__call__
    with TeaCache gating inserted between body and tail.

    img2img is supported as of v0.2.0. The forward path uses state.step_counter
    (0-based, per-generation) rather than the scheduler's absolute `t` for gate
    indexing, so img2img runs starting mid-schedule still index correctly.

    `rel_l1_thresh <= 0` takes a fast-path branch that does NOT build the
    gating tensors (`body_in_concat`, `mod_in`, `cached_residual`,
    `previous_mod_input`). At non-positive threshold no future step can ever
    skip, so the cache can never be consumed — these tensors would be
    unused work. The fast path also removes one previously suspected source
    of MLX allocation/refcount perturbation (the cached_residual array
    holding refs to body intermediates past the tail). Empirically this did
    NOT restore cross-process byte parity with vanilla mflux, but it does
    shrink the numerical surface area we're testing.
    """
    state = handle._state.cache
    stats = handle._state.stats

    # 1. Per-generation skip-window validation. Cache reset is lifecycle-owned
    #    (see lifecycle.py call_before_loop). We use state.step_counter == 0 as
    #    the once-per-generation marker for validation — not absolute `t == 0`,
    #    which would never fire under img2img where the first call has t > 0.
    if state.step_counter == 0 and not state.skip_window_validated:
        active_num_steps = handle._gen_ctx.active_num_steps
        if active_num_steps is None:
            # Defensive: lifecycle should have set this. Fall back to the ACTIVE
            # window (num_inference_steps - init_time_step), not the nominal
            # schedule, so an img2img run validates against the real denoising count.
            active_num_steps = _active_step_count(config)
        if handle.skip_first_n_steps + handle.skip_last_n_steps >= active_num_steps:
            raise InvalidStepWindowError(
                skip_first=handle.skip_first_n_steps,
                skip_last=handle.skip_last_n_steps,
                num_steps=active_num_steps,  # legacy alias; carries the active count
                nominal_num_inference_steps=config.num_inference_steps,
            )
        state.skip_window_validated = True

    # 2. Prelude (mirrors mflux Transformer.__call__ lines 44-47). Both paths
    # below need these intermediates.
    body_in = inner.x_embedder(hidden_states)
    encoder_hidden_states = inner.context_embedder(prompt_embeds)
    text_embeddings = inner.compute_text_embeddings(
        t,
        pooled_prompt_embeds,
        inner.time_text_embed,
        config,
    )
    image_rotary_embeddings = inner.compute_rotary_embeddings(
        prompt_embeds,
        inner.pos_embed,
        config,
        kwargs.get("kontext_image_ids"),
    )

    # 2a. Threshold-zero fast path: every step is "computed", cache is never
    # consumed. Skip mod_in extraction, body_in_concat, and the cached_residual
    # subtraction to avoid keeping intermediates alive past the tail.
    if handle.rel_l1_thresh <= 0.0:
        body_out_concat = _flux1_run_body(
            inner,
            body_in,
            encoder_hidden_states,
            text_embeddings,
            image_rotary_embeddings,
            kwargs,
        )
        stats.record(
            StepDecision(
                step_idx=state.step_counter,
                timestep=float(t),
                rel_l1=None,
                accumulated_distance=state.accumulated_distance,
                decision="computed",
            )
        )
        out = body_out_concat[:, encoder_hidden_states.shape[1] :, ...]
        out = inner.norm_out(out, text_embeddings)
        out = inner.proj_out(out)
        state.step_counter += 1
        return out

    # 2b. Slow path: TeaCache gating is live. Build the gating tensors.
    body_in_concat = mx.concatenate([encoder_hidden_states, body_in], axis=1)

    # 3. Extract mod_in for gating.
    mod_in = _flux1_extract_mod_input(inner.transformer_blocks[0], body_in, text_embeddings)

    # 4. Defensive shape check.
    if state.previous_mod_input is not None and mod_in.shape != state.previous_mod_input.shape:
        raise TransformerShapeError(
            step_idx=t,
            expected=state.previous_mod_input.shape,
            actual=mod_in.shape,
        )

    # 5. Gate. Use state.step_counter (0-based, per-generation) as step_idx
    #    so img2img generations starting at t > 0 still index the gate
    #    relative to the start of this generation. num_steps uses the active
    #    window (from lifecycle's _gen_ctx) so skip_last is measured from the
    #    actual end of denoising, not the nominal schedule.
    active_num_steps = handle._gen_ctx.active_num_steps
    if active_num_steps is None:
        active_num_steps = _active_step_count(config)  # defensive: active window, not nominal
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

    # 6. Stats — staging only; commit happens in call_after_loop.
    stats.record(_step_decision_from_gate(decision, step_idx=state.step_counter, timestep=float(t)))

    # 7. Compute path driven by decision.
    if decision.should_compute:
        body_out_concat = _flux1_run_body(
            inner,
            body_in,
            encoder_hidden_states,
            text_embeddings,
            image_rotary_embeddings,
            kwargs,
        )
        if decision.should_update_cache:
            state.cached_residual = body_out_concat - body_in_concat
            state.previous_mod_input = mod_in
    else:
        # decision.kind == "skipped"
        if state.cached_residual is None:
            raise InternalStateError(
                "cached_residual is None on a skipped step; "
                "this indicates a gate.py logic bug (gate_step should never return "
                "should_compute=False before the first computed step seeds the cache)."
            )
        body_out_concat = body_in_concat + state.cached_residual

    # 8. Tail (mirrors mflux Transformer.__call__ lines 77-80, always runs).
    out = body_out_concat[:, encoder_hidden_states.shape[1] :, ...]
    out = inner.norm_out(out, text_embeddings)
    out = inner.proj_out(out)

    # 9. Bump step counter for the next call.
    state.step_counter += 1
    return out


# ----- apply() — translated from v0.5.x api.py::apply_teacache FLUX.1 branch -----


def apply(
    flux: Any,
    *,
    rel_l1_thresh: float | None = None,
    coefficients: tuple[float, float, float, float, float] | None = None,
    skip_first_n_steps: int = 1,
    skip_last_n_steps: int = 1,
) -> TeaCacheHandle:
    """FLUX.1 dev apply. Public-API-equivalent of the FLUX.1 branch of
    v0.5.x apply_teacache."""
    # 1. Resolve rel_l1_thresh (caller > DEFAULT_THRESH).
    resolved_thresh: float = rel_l1_thresh if rel_l1_thresh is not None else DEFAULT_THRESH

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

    # 4. Save original transformer for rollback.
    original_transformer = flux.transformer

    import contextlib

    # Eager rollback list for the transactional patch (per audit medium #3):
    # if any mutation raises after the first, all preceding mutations are reversed.
    _rollbacks_so_far: list[Any] = []

    # 5. Build proxy and swap onto flux.transformer.
    proxy = ProxyFlux1Transformer(inner=original_transformer, handle=internal)
    flux.transformer = proxy
    _rollbacks_so_far.append(lambda: setattr(flux, "transformer", original_transformer))

    # 6. Register lifecycle callback via wrap_generate_image. This registers
    #    GenerationContextCallback and wraps flux.generate_image.
    # Import GenerationContextCallback lazily here (same module as lifecycle).
    # Call wrap_generate_image via the module so monkeypatching lifecycle in
    # tests affects this call site (top-level import caches the original ref).
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

    # 7. Build VariantPatch: rollback restores transformer + unsubscribes the
    #    callback + restores generate_image. NO stats finalize call (audit F2).
    def _restore_transformer() -> None:
        flux.transformer = original_transformer

    def _unsubscribe_callback() -> None:
        from mlx_teacache.integrations.mflux.lifecycle import _remove_callback_by_identity as _rcbi

        _rcbi(flux.callbacks, callback)

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
