# src/mlx_teacache/integrations/mflux/forward.py
"""Reimplementation of mflux's transformer forward passes with TeaCache
gating inserted between body and tail.

Why we have to reimplement: TeaCache caches the residual across the
transformer blocks. The only way to insert a gate at the body/tail boundary
is to know mflux's internal structure. We don't fork mflux; we pin it tight
(>=0.17,<0.18) and the threshold=0 parity test (test_parity_flux1.py /
test_parity_flux2.py) is the bit-exact correctness gate.

Reference: mflux/models/flux/model/flux_transformer/transformer.py:32-80 (FLUX.1)
           mflux/models/flux2/model/flux2_transformer/transformer.py:67-133 (FLUX.2)
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from mlx_teacache.errors import (
    InternalStateError,
    InvalidStepWindowError,
    TransformerShapeError,
)
from mlx_teacache.gate import gate_step
from mlx_teacache.stats import StepDecision


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
            idx=idx, block=block,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            text_embeddings=text_embeddings,
            image_rotary_embeddings=image_rotary_embeddings,
            controlnet_block_samples=kwargs.get("controlnet_block_samples"),
        )
    hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
    for idx, block in enumerate(inner.single_transformer_blocks):
        hidden_states = inner._apply_single_transformer_block(
            idx=idx, block=block,
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

    img2img has already been rejected by GenerationContextCallback before any
    transformer call. We don't re-check config.image_path here.

    `rel_l1_thresh <= 0` takes a fast-path branch that does NOT build the
    gating tensors (`body_in_concat`, `mod_in`, `cached_residual`,
    `previous_mod_input`). At non-positive threshold no future step can ever
    skip, so the cache can never be consumed — these tensors would be
    unused work. The fast path also removes one previously suspected source
    of MLX allocation/refcount perturbation (the cached_residual array
    holding refs to body intermediates past the tail). Empirically this did
    NOT restore cross-process byte parity with vanilla mflux, but it does
    shrink the numerical surface area we're testing. See
    docs/superpowers/notes/2026-05-14-task-25-{mlx-nondeterminism,fast-path-measurement}.md
    and the 2026-05-15 audit addendum for the measurement detail.
    """
    state = handle._state.cache
    stats = handle._state.stats

    # 1. Per-generation reset (§5.2): unconditional on every t == 0.
    if t == 0:
        state.reset_for_new_generation(num_steps=config.num_inference_steps)
        # FLUX.1 lazy skip-window validation (§5.6): every step is TeaCache-active
        # (no CFG path), so validate at t == 0 unconditionally.
        if handle.skip_first_n_steps + handle.skip_last_n_steps >= config.num_inference_steps:
            raise InvalidStepWindowError(
                skip_first=handle.skip_first_n_steps,
                skip_last=handle.skip_last_n_steps,
                num_steps=config.num_inference_steps,
            )
        state.skip_window_validated = True

    # 2. Prelude (mirrors mflux Transformer.__call__ lines 44-47). Both paths
    # below need these intermediates.
    body_in = inner.x_embedder(hidden_states)
    encoder_hidden_states = inner.context_embedder(prompt_embeds)
    text_embeddings = inner.compute_text_embeddings(
        t, pooled_prompt_embeds, inner.time_text_embed, config,
    )
    image_rotary_embeddings = inner.compute_rotary_embeddings(
        prompt_embeds, inner.pos_embed, config, kwargs.get("kontext_image_ids"),
    )

    # 2a. Threshold-zero fast path: every step is "computed", cache is never
    # consumed. Skip mod_in extraction, body_in_concat, and the cached_residual
    # subtraction to avoid keeping intermediates alive past the tail.
    if handle.rel_l1_thresh <= 0.0:
        body_out_concat = _flux1_run_body(
            inner, body_in, encoder_hidden_states, text_embeddings,
            image_rotary_embeddings, kwargs,
        )
        stats.record(StepDecision(
            step_idx=t, timestep=float(t),
            rel_l1=None, accumulated_distance=state.accumulated_distance,
            decision="computed",
        ))
        out = body_out_concat[:, encoder_hidden_states.shape[1]:, ...]
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
            step_idx=t, expected=state.previous_mod_input.shape, actual=mod_in.shape,
        )

    # 5. Gate.
    decision = gate_step(
        state,
        rel_l1_thresh=handle.rel_l1_thresh,
        coefficients=handle.coefficients,
        skip_first=handle.skip_first_n_steps,
        skip_last=handle.skip_last_n_steps,
        num_steps=config.num_inference_steps,
        step_idx=t,
        mod_in=mod_in,
    )

    # 6. Stats — staging only; commit happens in call_after_loop.
    stats.record(_step_decision_from_gate(decision, step_idx=t, timestep=float(t)))

    # 7. Compute path driven by decision.
    if decision.should_compute:
        body_out_concat = _flux1_run_body(
            inner, body_in, encoder_hidden_states, text_embeddings,
            image_rotary_embeddings, kwargs,
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
    out = body_out_concat[:, encoder_hidden_states.shape[1]:, ...]
    out = inner.norm_out(out, text_embeddings)
    out = inner.proj_out(out)

    # 9. Bump step counter for the next call.
    state.step_counter += 1
    return out


# --- FLUX.2 ---


def _flux2_compute_mod_in(inner: Any, hidden_states: mx.array, timestep: mx.array) -> Any:
    """Shared helper: compute the TeaCache gating signal from raw inputs.

    Performs the exact same preprocessing as flux2_forward_with_gate up to
    the modulation-extraction point, so calibration recorders and the runtime
    gate see byte-identical mod_in tensors. (Per audit medium #6.)"""
    from mflux.models.common.config.model_config import ModelConfig
    if not isinstance(timestep, mx.array):
        timestep = mx.array(timestep, dtype=hidden_states.dtype)
    if timestep.ndim == 0:
        timestep = mx.full((hidden_states.shape[0],), timestep, dtype=hidden_states.dtype)
    timestep = timestep.astype(hidden_states.dtype)
    timestep_scale = mx.where(mx.max(timestep) <= 1.0, 1000.0, 1.0).astype(hidden_states.dtype)
    timestep = timestep * timestep_scale
    temb = inner.time_guidance_embed(timestep, None)
    temb = temb.astype(ModelConfig.precision)
    body_in = inner.x_embedder(hidden_states)
    temb_mod_params_img = inner.double_stream_modulation_img(temb)
    return _flux2_extract_mod_input(inner, body_in, temb_mod_params_img)


def _flux2_extract_mod_input(
    inner: Any, body_in: mx.array, temb_mod_params_img: tuple[mx.array, ...],
) -> Any:
    """Extract the modulated input for FLUX.2.

    Flux2Modulation produces a tuple of modulation parameters. The first
    set is applied to body_in inside Flux2TransformerBlock; we replicate
    the modulation step here without running attention so we can use the
    result as the gating signal.

    Reference: mflux flux2_transformer/modulation.py + transformer_block.py.

    Flux2Modulation returns (shift, scale, gate, ...). We apply ada-LN-zero
    to body_in: out = body_in * (1 + scale) + shift."""
    scale = temb_mod_params_img[1][:, None, :]
    shift = temb_mod_params_img[0][:, None, :]
    return body_in * (1.0 + scale) + shift


def _flux2_run_body(
    inner: Any,
    body_in: mx.array,
    encoder_hidden_states: mx.array,
    temb: mx.array,
    temb_mod_params_img: Any,
    temb_mod_params_txt: Any,
    image_rotary_emb: Any,
) -> mx.array:
    """Run all Flux2TransformerBlocks then all Flux2SingleTransformerBlocks.
    Mirrors Flux2Transformer.__call__ lines 111-128 EXACTLY, including the
    intra-body computation of `temb_mod_params_single` between the joint
    and single block loops. Computing single_stream_modulation upfront
    instead of inline produces a different MLX graph topology even though
    the math is equivalent — under the eager (non-compiled) wrapper that
    suffices to shift Metal dispatch and accumulate divergence vs vanilla
    (verified 2026-05-15: upfront computation gave max_abs ~3.77 after 8
    steps vs inline matching bit-exact)."""
    hidden_states = body_in
    for block in inner.transformer_blocks:
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod_params_img=temb_mod_params_img,
            temb_mod_params_txt=temb_mod_params_txt,
            image_rotary_emb=image_rotary_emb,
        )
    hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
    # Compute single-stream modulation HERE (matching vanilla line 122), not
    # in the caller, to preserve identical graph topology to vanilla.
    temb_mod_params_single = inner.single_stream_modulation(temb)[0]
    for block in inner.single_transformer_blocks:
        hidden_states = block(
            hidden_states=hidden_states,
            temb_mod_params=temb_mod_params_single,
            image_rotary_emb=image_rotary_emb,
        )
    return hidden_states


def flux2_forward_with_gate(
    inner: Any,
    handle: Any,
    *,
    hidden_states: mx.array,
    encoder_hidden_states: mx.array,
    timestep: mx.array,
    img_ids: mx.array,
    txt_ids: mx.array,
) -> Any:
    """Replacement for Flux2Transformer.__call__ with gating.

    img2img is rejected pre-loop. CFG is handled in the predict closure (we
    only reach this function for non-CFG steps). Skip-window validation is
    handled in the predict closure too (this function is the 'first non-CFG
    gated step' boundary — see §5.6)."""
    state = handle._state.cache
    stats = handle._state.stats

    # 1. Prelude (mirrors Flux2Transformer.__call__ lines 76-109).
    #    Uses the shared _flux2_compute_mod_in helper for mod_in extraction so
    #    calibration and production see byte-identical signals (audit medium #6).
    from mflux.models.common.config.model_config import ModelConfig
    if not isinstance(timestep, mx.array):
        timestep = mx.array(timestep, dtype=hidden_states.dtype)
    if timestep.ndim == 0:
        timestep = mx.full((hidden_states.shape[0],), timestep, dtype=hidden_states.dtype)
    timestep = timestep.astype(hidden_states.dtype)
    timestep_scale = mx.where(mx.max(timestep) <= 1.0, 1000.0, 1.0).astype(hidden_states.dtype)
    timestep = timestep * timestep_scale
    temb = inner.time_guidance_embed(timestep, None)
    temb = temb.astype(ModelConfig.precision)

    body_in = inner.x_embedder(hidden_states)
    encoder_hidden_states = inner.context_embedder(encoder_hidden_states)
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    image_rotary_emb = inner.pos_embed(img_ids)
    text_rotary_emb = inner.pos_embed(txt_ids)
    concat_rotary_emb = (
        mx.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
        mx.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
    )
    temb_mod_params_img = inner.double_stream_modulation_img(temb)
    temb_mod_params_txt = inner.double_stream_modulation_txt(temb)
    # NOTE: `temb_mod_params_single = single_stream_modulation(temb)[0]` is
    # computed INSIDE _flux2_run_body between the joint loop and the single
    # loop — exactly where vanilla Flux2Transformer.__call__ computes it
    # (line 122). Computing it upfront changes MLX graph topology under
    # eager dispatch and produces a different output. See _flux2_run_body
    # docstring for the measured impact.

    # 1a. Threshold-zero fast path (FLUX.2 mirror of the FLUX.1 fast path).
    # No future step can ever skip at non-positive threshold, so the cache is
    # never consumed. Skip building mod_in, body_in_concat, and the
    # cached_residual subtraction to avoid keeping body intermediates alive
    # past the tail and shrink the numerical surface area we test against.
    # See docs/superpowers/notes/2026-05-14-task-25-fast-path-measurement.md.
    if handle.rel_l1_thresh <= 0.0:
        body_out_concat = _flux2_run_body(
            inner, body_in, encoder_hidden_states, temb,
            temb_mod_params_img, temb_mod_params_txt,
            concat_rotary_emb,
        )
        stats.record(StepDecision(
            step_idx=state.step_counter,
            timestep=float(timestep.flatten()[0]),
            rel_l1=None,
            accumulated_distance=state.accumulated_distance,
            decision="computed",
        ))
        state.last_timestep = float(timestep.flatten()[0])
        out = body_out_concat[:, encoder_hidden_states.shape[1]:, ...]
        out = inner.norm_out(out, temb)
        out = inner.proj_out(out)
        state.step_counter += 1
        return out

    # 1b. Slow path: TeaCache gating live. Build the gating tensors.
    body_in_concat = mx.concatenate([encoder_hidden_states, body_in], axis=1)

    # 2. Extract mod_in (same value _flux2_compute_mod_in would return).
    mod_in = _flux2_extract_mod_input(inner, body_in, temb_mod_params_img)

    # 3. Defensive shape check.
    if state.previous_mod_input is not None and mod_in.shape != state.previous_mod_input.shape:
        raise TransformerShapeError(
            step_idx=state.step_counter,
            expected=state.previous_mod_input.shape, actual=mod_in.shape,
        )

    # 4. Gate.
    decision = gate_step(
        state,
        rel_l1_thresh=handle.rel_l1_thresh,
        coefficients=handle.coefficients,
        skip_first=handle.skip_first_n_steps,
        skip_last=handle.skip_last_n_steps,
        num_steps=handle._gen_ctx.active_num_steps,
        step_idx=state.step_counter,
        mod_in=mod_in,
    )

    # 5. Stats record.
    stats.record(_step_decision_from_gate(
        decision, step_idx=state.step_counter, timestep=float(timestep.flatten()[0]),
    ))

    # 6. Debug-only timestep tracking.
    state.last_timestep = float(timestep.flatten()[0])

    # 7. Compute path.
    if decision.should_compute:
        body_out_concat = _flux2_run_body(
            inner, body_in, encoder_hidden_states, temb,
            temb_mod_params_img, temb_mod_params_txt,
            concat_rotary_emb,
        )
        if decision.should_update_cache:
            state.cached_residual = body_out_concat - body_in_concat
            state.previous_mod_input = mod_in
    else:
        if state.cached_residual is None:
            raise InternalStateError(
                "cached_residual is None on a skipped step (FLUX.2); "
                "this indicates a gate.py logic bug."
            )
        body_out_concat = body_in_concat + state.cached_residual

    # 8. Tail.
    out = body_out_concat[:, encoder_hidden_states.shape[1]:, ...]
    out = inner.norm_out(out, temb)
    out = inner.proj_out(out)

    # 9. Bump step counter.
    state.step_counter += 1
    return out
