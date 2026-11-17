"""The shared FLUX.1 forward, used by every FLUX.1 variant (dev, schnell, Krea [dev]):
- ProxyFlux1Transformer: an nn.Module standing in for flux.transformer.
- flux1_forward_with_gate: the FLUX.1 forward with the gate between body and tail.
- _flux1_prelude, _flux1_extract_mod_input, _flux1_run_body, _flux1_tail: the pieces of the
  FLUX.1 transformer call that the gate splits apart (the calibration script reuses them).

Imports no mflux: the forward only calls the transformer it wraps.
"""

from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

from mlx_teacache._kernel.gate import gate_step
from mlx_teacache._kernel.step import (
    cache_residuals,
    check_signal_shape,
    fast_path_decision,
    is_disabled,
    reconstruct,
    step_decision,
    validate_window_once,
)
from mlx_teacache.integrations.mflux.lifecycle import resolve_active_num_steps

# ----- Proxy transformer -----


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

    # Everything nn.Module walks through update()/children()/filter_and_map()
    # (load_weights, set_dtype, quantize, eval/train) must see the inner tree,
    # not the proxy's own empty dict. Signatures match mlx 0.31.2 nn/layers/base.py.
    def update(self, parameters: dict[str, Any], strict: bool = True) -> "ProxyFlux1Transformer":
        self._inner.update(parameters, strict=strict)
        return self

    def update_modules(self, modules: dict[str, Any], strict: bool = True) -> "ProxyFlux1Transformer":
        self._inner.update_modules(modules, strict=strict)
        return self

    def filter_and_map(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.filter_and_map(*args, **kwargs)

    def children(self) -> Any:
        return self._inner.children()

    def leaf_modules(self) -> Any:
        return self._inner.leaf_modules()

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


# ----- Gated FLUX.1 forward -----


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


def _flux1_prelude(
    inner: Any,
    *,
    t: int,
    config: Any,
    hidden_states: mx.array,
    prompt_embeds: mx.array,
    pooled_prompt_embeds: mx.array,
    kwargs: dict[str, Any],
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """The four intermediates every FLUX.1 step needs before the body: embedded
    image tokens, embedded text tokens, the timestep/pooled-text embedding, and
    the rotary embeddings. Mirrors mflux Transformer.__call__ lines 44-47. Shared
    with scripts/calibrate_flux1.py so the calibrator never re-walks it."""
    body_in = inner.x_embedder(hidden_states)
    encoder_hidden_states = inner.context_embedder(prompt_embeds)
    text_embeddings = inner.compute_text_embeddings(t, pooled_prompt_embeds, inner.time_text_embed, config)
    image_rotary_embeddings = inner.compute_rotary_embeddings(
        prompt_embeds, inner.pos_embed, config, kwargs.get("kontext_image_ids")
    )
    return body_in, encoder_hidden_states, text_embeddings, image_rotary_embeddings


def _flux1_tail(
    inner: Any, body_out_concat: mx.array, encoder_hidden_states: mx.array, text_embeddings: mx.array
) -> Any:
    """Drop the text tokens, final norm, projection (mflux Transformer.__call__ lines 77-80)."""
    out = body_out_concat[:, encoder_hidden_states.shape[1] :, ...]
    out = inner.norm_out(out, text_embeddings)
    return inner.proj_out(out)


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

    img2img is supported. The forward path uses state.step_counter
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
    #    The active count is resolved once per call, for this check and the gate below; it falls
    #    back (defensively) to the ACTIVE window, not the nominal schedule.
    active_num_steps = resolve_active_num_steps(handle, config)
    validate_window_once(
        state,
        active_num_steps=active_num_steps,
        skip_first_n_steps=handle.skip_first_n_steps,
        skip_last_n_steps=handle.skip_last_n_steps,
        nominal_num_inference_steps=config.num_inference_steps,
        first_step_only=True,
    )

    # 2. Prelude (mirrors mflux Transformer.__call__ lines 44-47). Both paths
    # below need these intermediates.
    body_in, encoder_hidden_states, text_embeddings, image_rotary_embeddings = _flux1_prelude(
        inner,
        t=t,
        config=config,
        hidden_states=hidden_states,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        kwargs=kwargs,
    )

    # 2a. Threshold-zero fast path: every step is "computed", cache is never
    # consumed. Skip mod_in extraction, body_in_concat, and the cached_residual
    # subtraction to avoid keeping intermediates alive past the tail.
    if is_disabled(handle.rel_l1_thresh):
        body_out_concat = _flux1_run_body(
            inner,
            body_in,
            encoder_hidden_states,
            text_embeddings,
            image_rotary_embeddings,
            kwargs,
        )
        stats.record(fast_path_decision(state, timestep=float(t)))
        out = _flux1_tail(inner, body_out_concat, encoder_hidden_states, text_embeddings)
        state.step_counter += 1
        return out

    # 2b. Slow path: TeaCache gating is live. Build the gating tensors.
    body_in_concat = mx.concatenate([encoder_hidden_states, body_in], axis=1)

    # 3. Extract mod_in for gating.
    mod_in = _flux1_extract_mod_input(inner.transformer_blocks[0], body_in, text_embeddings)

    # 4. Defensive shape check.
    check_signal_shape(state, mod_in)

    # 5. Gate. Use state.step_counter (0-based, per-generation) as step_idx
    #    so img2img generations starting at t > 0 still index the gate
    #    relative to the start of this generation. num_steps uses the active
    #    window (from lifecycle's _gen_ctx) so skip_last is measured from the
    #    actual end of denoising, not the nominal schedule.
    decision = gate_step(
        state,
        rel_l1_thresh=handle.rel_l1_thresh,
        coefficients=handle.coefficients,
        skip_first=handle.skip_first_n_steps,
        skip_last=handle.skip_last_n_steps,
        num_steps=active_num_steps,
        step_idx=state.step_counter,
        mod_in=mod_in,
        calibrated_range=handle.calibrated_range,
    )

    # 6. Stats — staging only; commit happens in call_after_loop.
    stats.record(step_decision(decision, step_idx=state.step_counter, timestep=float(t)))

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
        cache_residuals(state, decision, pos=(body_out_concat, body_in_concat))
    else:
        # decision.kind == "skipped"
        body_out_concat = reconstruct(body_in_concat, state, decision)

    # 8. Tail (mirrors mflux Transformer.__call__ lines 77-80, always runs).
    out = _flux1_tail(inner, body_out_concat, encoder_hidden_states, text_embeddings)

    # 9. Bump step counter for the next call.
    state.step_counter += 1
    return out
