"""The shared FLUX.2 forwards and the `_predict` replacement, used by every FLUX.2 Klein variant:

- _flux2_extract_mod_input, _flux2_run_body, _flux2_apply_tail_and_combine:
  the pieces of the FLUX.2 transformer call that the gate splits apart.
- flux2_forward_with_gate: the no-CFG path.
- flux2_cfg_forward_with_gate: the CFG path, one shared gate decision per step
  and one cached residual per branch.
- make_teacache_predict_factory: the callable assigned to flux._predict, which picks the
  forward from the guidance scale.

mflux is imported only inside the forwards (ModelConfig, at call time).
"""

from collections.abc import Callable
from typing import Any, cast

import mlx.core as mx

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

# ---------- Gated FLUX.2 forwards ----------

# The old name, kept only for the variant module that still re-exports it.
_step_decision_from_gate = step_decision


def _flux2_extract_mod_input(
    inner: Any,
    body_in: mx.array,
    temb_mod_params_img: Any,
) -> Any:
    """Extract the modulated input for FLUX.2 — the TeaCache gating signal.

    Mirrors the first ada-LN-zero block of `Flux2TransformerBlock.__call__`
    exactly so the signal we threshold against is the same tensor the
    attention path actually sees:

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

    Reference: mflux flux2_transformer/transformer_block.py lines 32-36.

    `Flux2Modulation` with `mod_param_sets=2` returns a nested tuple:
        temb_mod_params_img = ((shift_msa, scale_msa, gate_msa),
                                (shift_mlp, scale_mlp, gate_mlp))
    We use the MSA (first) set — that's the one applied to the body input
    before attention. Each element is already shaped [B, 1, D] (from
    `expand_dims` inside `Flux2Modulation.__call__`)."""
    (shift_msa, scale_msa, _gate_msa), *_ = temb_mod_params_img
    norm_in = inner.transformer_blocks[0].norm1(body_in)
    return (1.0 + scale_msa) * norm_in + shift_msa


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

    img2img is supported: the active denoising window is set up
    by lifecycle's call_before_loop and consumed via handle._gen_ctx. CFG is
    handled in the predict closure (we only reach this function for non-CFG
    steps). Skip-window validation is handled in the predict closure too: it
    runs once per generation, on the first gated call, CFG or not, before this
    function or the CFG forward is reached."""
    state = handle._state.cache
    stats = handle._state.stats

    # 1. Prelude (mirrors Flux2Transformer.__call__ lines 76-109).
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
    if is_disabled(handle.rel_l1_thresh):
        body_out_concat = _flux2_run_body(
            inner,
            body_in,
            encoder_hidden_states,
            temb,
            temb_mod_params_img,
            temb_mod_params_txt,
            concat_rotary_emb,
        )
        timestep_val = float(timestep.flatten()[0])
        stats.record(fast_path_decision(state, timestep=timestep_val))
        state.last_timestep = timestep_val
        out = body_out_concat[:, encoder_hidden_states.shape[1] :, ...]
        out = inner.norm_out(out, temb)
        out = inner.proj_out(out)
        state.step_counter += 1
        return out

    # 1b. Slow path: TeaCache gating live. Build the gating tensors.
    body_in_concat = mx.concatenate([encoder_hidden_states, body_in], axis=1)

    # 2. Extract mod_in from the first block's modulated input.
    mod_in = _flux2_extract_mod_input(inner, body_in, temb_mod_params_img)

    # 3. Defensive shape check.
    check_signal_shape(state, mod_in)

    # 4. Gate. `active_num_steps` is guaranteed non-None here: the predict closure
    #    consumes + validates the generation context before any forward call (see
    #    make_teacache_predict_factory). FLUX.2's forward has no `config` to fall
    #    back to the way FLUX.1 does, so that closure check is the SOLE guarantee —
    #    don't add a None-coalesce here expecting a nominal fallback.
    decision = gate_step(
        state,
        rel_l1_thresh=handle.rel_l1_thresh,
        coefficients=handle.coefficients,
        skip_first=handle.skip_first_n_steps,
        skip_last=handle.skip_last_n_steps,
        num_steps=handle._gen_ctx.active_num_steps,
        step_idx=state.step_counter,
        mod_in=mod_in,
        calibrated_range=handle.calibrated_range,
    )

    # 5. Stats record and timestep tracking (one host read for both).
    timestep_val = float(timestep.flatten()[0])
    stats.record(step_decision(decision, step_idx=state.step_counter, timestep=timestep_val))
    state.last_timestep = timestep_val

    # 6. Compute path.
    if decision.should_compute:
        body_out_concat = _flux2_run_body(
            inner,
            body_in,
            encoder_hidden_states,
            temb,
            temb_mod_params_img,
            temb_mod_params_txt,
            concat_rotary_emb,
        )
        cache_residuals(state, decision, pos=(body_out_concat, body_in_concat))
    else:
        body_out_concat = reconstruct(body_in_concat, state, decision)

    # 8. Tail.
    out = body_out_concat[:, encoder_hidden_states.shape[1] :, ...]
    out = inner.norm_out(out, temb)
    out = inner.proj_out(out)

    # 9. Bump step counter.
    state.step_counter += 1
    return out


def flux2_cfg_forward_with_gate(
    inner: Any,
    handle: Any,
    *,
    hidden_states: mx.array,
    prompt_embeds: mx.array,
    text_ids: mx.array,
    negative_prompt_embeds: mx.array,
    negative_text_ids: mx.array,
    guidance: float,
    timestep: mx.array,
    img_ids: mx.array,
) -> Any:
    """Gated CFG forward for FLUX.2.

    One shared polynomial-gate decision per step: mod_in is built from the
    image latents and the timestep embedding only, never the prompt
    embeddings, so it is the same on both branches. Two cached residuals
    (positive + negative). CFG combination math runs after the tail on
    both branches.

    In production this stands in for mflux's own CFG predict, which calls
    the transformer once per branch and then combines the two outputs."""
    from mflux.models.common.config.model_config import ModelConfig

    state = handle._state.cache
    stats = handle._state.stats

    # 1. Shared prelude (mirrors flux2_forward_with_gate's step-1 prelude
    #    minus the encoder_hidden_states and text-rotary handling, which are per-branch).
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
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    image_rotary_emb = inner.pos_embed(img_ids)
    temb_mod_params_img = inner.double_stream_modulation_img(temb)
    temb_mod_params_txt = inner.double_stream_modulation_txt(temb)

    # 2. Per-branch encoder + text-rotary prep. These differ per branch.
    enc_pos = inner.context_embedder(prompt_embeds)
    txt_ids_pos = text_ids[0] if text_ids.ndim == 3 else text_ids
    txt_rot_pos = inner.pos_embed(txt_ids_pos)
    concat_rot_pos = (
        mx.concatenate([txt_rot_pos[0], image_rotary_emb[0]], axis=0),
        mx.concatenate([txt_rot_pos[1], image_rotary_emb[1]], axis=0),
    )

    enc_neg = inner.context_embedder(negative_prompt_embeds)
    txt_ids_neg = negative_text_ids[0] if negative_text_ids.ndim == 3 else negative_text_ids
    txt_rot_neg = inner.pos_embed(txt_ids_neg)
    concat_rot_neg = (
        mx.concatenate([txt_rot_neg[0], image_rotary_emb[0]], axis=0),
        mx.concatenate([txt_rot_neg[1], image_rotary_emb[1]], axis=0),
    )

    timestep_val = float(timestep.flatten()[0])

    # 3. Fast path (threshold <= 0): run both bodies, no caching.
    if is_disabled(handle.rel_l1_thresh):
        body_out_pos = _flux2_run_body(
            inner, body_in, enc_pos, temb, temb_mod_params_img, temb_mod_params_txt, concat_rot_pos
        )
        body_out_neg = _flux2_run_body(
            inner, body_in, enc_neg, temb, temb_mod_params_img, temb_mod_params_txt, concat_rot_neg
        )
        stats.record(fast_path_decision(state, timestep=timestep_val))
        state.last_timestep = timestep_val
        state.step_counter += 1
        return _flux2_apply_tail_and_combine(
            inner, body_out_pos, body_out_neg, enc_pos, enc_neg, temb, guidance
        )

    # 4. Slow path: build mod_in, run gate ONCE on shared signal.
    mod_in = _flux2_extract_mod_input(inner, body_in, temb_mod_params_img)
    check_signal_shape(state, mod_in)

    # active_num_steps is non-None by the predict-closure guarantee (see the note
    # at the non-CFG gate above) — FLUX.2 has no nominal fallback.
    decision = gate_step(
        state,
        rel_l1_thresh=handle.rel_l1_thresh,
        coefficients=handle.coefficients,
        skip_first=handle.skip_first_n_steps,
        skip_last=handle.skip_last_n_steps,
        num_steps=handle._gen_ctx.active_num_steps,
        step_idx=state.step_counter,
        mod_in=mod_in,
        calibrated_range=handle.calibrated_range,
    )
    stats.record(step_decision(decision, step_idx=state.step_counter, timestep=timestep_val))
    state.last_timestep = timestep_val

    # 5. Compute / skip — applied uniformly across both branches.
    body_in_concat_pos = mx.concatenate([enc_pos, body_in], axis=1)
    body_in_concat_neg = mx.concatenate([enc_neg, body_in], axis=1)
    if decision.should_compute:
        body_out_pos = _flux2_run_body(
            inner, body_in, enc_pos, temb, temb_mod_params_img, temb_mod_params_txt, concat_rot_pos
        )
        body_out_neg = _flux2_run_body(
            inner, body_in, enc_neg, temb, temb_mod_params_img, temb_mod_params_txt, concat_rot_neg
        )
        cache_residuals(
            state,
            decision,
            pos=(body_out_pos, body_in_concat_pos),
            neg=(body_out_neg, body_in_concat_neg),
        )
    else:
        body_out_pos = reconstruct(body_in_concat_pos, state, decision)
        body_out_neg = reconstruct(body_in_concat_neg, state, decision, branch="neg")

    state.step_counter += 1
    return _flux2_apply_tail_and_combine(inner, body_out_pos, body_out_neg, enc_pos, enc_neg, temb, guidance)


def _flux2_apply_tail_and_combine(
    inner: Any,
    body_out_pos: mx.array,
    body_out_neg: mx.array,
    enc_pos: mx.array,
    enc_neg: mx.array,
    temb: mx.array,
    guidance: float,
) -> Any:
    """Apply Flux2 norm_out + proj_out tail to each branch independently,
    then combine via CFG math: negative + guidance * (positive - negative).

    norm_out + proj_out are branch-independent ops parameterized by temb;
    we apply them once per branch because the body_out per branch differs.
    The CFG math is the same triplet mflux uses (flux2_klein.py:267-276)."""
    noise_pos = body_out_pos[:, enc_pos.shape[1] :, ...]
    noise_pos = inner.norm_out(noise_pos, temb)
    noise_pos = inner.proj_out(noise_pos)

    noise_neg = body_out_neg[:, enc_neg.shape[1] :, ...]
    noise_neg = inner.norm_out(noise_neg, temb)
    noise_neg = inner.proj_out(noise_neg)

    return noise_neg + guidance * (noise_pos - noise_neg)


# ---------- _predict replacement ----------

PredictFn = Callable[
    [mx.array, mx.array, mx.array, mx.array, "mx.array | None", "mx.array | None", float, mx.array],
    Any,
]
PredictFactory = Callable[[Any], PredictFn]


def make_teacache_predict_factory(handle: Any) -> PredictFactory:
    """Return a callable assignable to `flux._predict`. Mflux will call it
    as `predict = self._predict(self.transformer)` at the top of each
    generate_image."""
    from mlx_teacache.errors import MissingGenerationContextError

    def predict_factory(transformer: Any) -> PredictFn:
        # Closure-local — fresh per generation. Survives mid-loop crashes of
        # previous runs because Flux2Klein.generate_image builds a new closure
        # every time it calls self._predict(self.transformer).
        context_consumed = False

        def predict(
            latents: mx.array,
            latent_ids: mx.array,
            prompt_embeds: mx.array,
            text_ids: mx.array,
            negative_prompt_embeds: mx.array | None,
            negative_text_ids: mx.array | None,
            guidance: float,
            timestep: mx.array,
        ) -> Any:
            nonlocal context_consumed
            ctx = handle._gen_ctx

            # 1. Consume context on the first call of THIS generation's closure.
            if not context_consumed:
                if ctx.active_num_steps is None or ctx.consumed_at_token == ctx.token:
                    raise MissingGenerationContextError()
                ctx.consumed_at_token = ctx.token
                context_consumed = True

            # 2. Lazy skip-window validation. It runs on the first gated call
            #    regardless of CFG, so an all-CFG generation with a bad window
            #    raises instead of silently running vanilla.
            validate_window_once(
                handle._state.cache,
                active_num_steps=cast(int, handle._gen_ctx.active_num_steps),
                skip_first_n_steps=handle.skip_first_n_steps,
                skip_last_n_steps=handle.skip_last_n_steps,
            )

            # 3. CFG branch: gated per-branch caching.
            cfg_active = negative_prompt_embeds is not None and negative_text_ids is not None
            if cfg_active:
                assert negative_prompt_embeds is not None
                assert negative_text_ids is not None
                handle._state.stats._staging.cfg_was_active = True
                return flux2_cfg_forward_with_gate(
                    transformer,
                    handle,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_text_ids=negative_text_ids,
                    guidance=guidance,
                    timestep=timestep,
                    img_ids=latent_ids,
                )

            # 4. Non-CFG branch: unchanged from v0.4.0.
            return flux2_forward_with_gate(
                transformer,
                handle,
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=latent_ids,
                txt_ids=text_ids,
            )

        return predict

    return predict_factory
