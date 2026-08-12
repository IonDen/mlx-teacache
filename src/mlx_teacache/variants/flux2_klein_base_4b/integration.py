"""FLUX.2 Klein base 4B integration. Byte-for-byte port from v0.5.x:
- src/mlx_teacache/integrations/mflux/flux2.py (proxy / factory)
- src/mlx_teacache/integrations/mflux/forward.py FLUX.2 block:
  * _flux2_extract_mod_input
  * _flux2_run_body
  * flux2_forward_with_gate (no-CFG path)
  * flux2_cfg_forward_with_gate (CFG-per-branch, v0.4.1)
  * _flux2_apply_tail_and_combine
- src/mlx_teacache/api.py::apply_teacache FLUX.2 branch (apply logic +
  guidance-based forward selection)

mflux is imported only inside this module. The registry loads this
lazily, after detect.matches() wins.
"""

from collections.abc import Callable
from typing import Any

import mlx.core as mx

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.gate import gate_step
from mlx_teacache._kernel.stats import StepDecision
from mlx_teacache.handle import TeaCacheHandle, VariantPatch
from mlx_teacache.integrations.mflux.lifecycle import (
    GenerationContextCallback,
    wrap_generate_image,
)

# Reuse the _InternalHandle bridge from flux1_dev — its shape is variant-agnostic
# at the integration level (just carries state, gen_ctx, lifecycle hooks).
from mlx_teacache.variants.flux1_dev.integration import _InternalHandle

from .config import COEFFICIENTS, DEFAULT_THRESH

_PROVENANCE = Provenance(
    source="builtin",
    revision="in-repo-2026-05-17-origin",
    calibration_dataset=(
        "10 prompts × 25 steps × seed=42, M1 Max 32GB, bf16, 512x512, "
        "guidance=1.0, origin-constrained polyfit"
    ),
    fit_metric="constrained-LSQ R^2 on consecutive-step (mod_in, body_out) rel-L1 pairs (poly(0)=0)",
    fit_metric_value=0.10643408169124158,
    reference_url="https://github.com/IonDen/mlx-teacache/blob/main/scripts/calibrate_flux2.py",
    default_thresh=0.17,
)


# ---------- PORTED VERBATIM from src/mlx_teacache/integrations/mflux/forward.py ----------
# FLUX.2 block: lines 258-664.

from mlx_teacache.errors import (  # noqa: E402
    InternalStateError,
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

    img2img is supported as of v0.2.0; the active denoising window is set up
    by lifecycle's call_before_loop and consumed via handle._gen_ctx. CFG is
    handled in the predict closure (we only reach this function for non-CFG
    steps). Skip-window validation is handled in the predict closure too
    (this function is the 'first non-CFG gated step' boundary — see §5.6)."""
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
    if handle.rel_l1_thresh <= 0.0:
        body_out_concat = _flux2_run_body(
            inner,
            body_in,
            encoder_hidden_states,
            temb,
            temb_mod_params_img,
            temb_mod_params_txt,
            concat_rotary_emb,
        )
        stats.record(
            StepDecision(
                step_idx=state.step_counter,
                timestep=float(timestep.flatten()[0]),
                rel_l1=None,
                accumulated_distance=state.accumulated_distance,
                decision="computed",
            )
        )
        state.last_timestep = float(timestep.flatten()[0])
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
    if state.previous_mod_input is not None and mod_in.shape != state.previous_mod_input.shape:
        raise TransformerShapeError(
            step_idx=state.step_counter,
            expected=state.previous_mod_input.shape,
            actual=mod_in.shape,
        )

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
    )

    # 5. Stats record.
    stats.record(
        _step_decision_from_gate(
            decision,
            step_idx=state.step_counter,
            timestep=float(timestep.flatten()[0]),
        )
    )

    # 6. Debug-only timestep tracking.
    state.last_timestep = float(timestep.flatten()[0])

    # 7. Compute path.
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
        if decision.should_update_cache:
            state.cached_residual = body_out_concat - body_in_concat
    else:
        if state.cached_residual is None:
            raise InternalStateError(
                "cached_residual is None on a skipped step (FLUX.2); this indicates a gate.py logic bug."
            )
        body_out_concat = body_in_concat + state.cached_residual

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
    """v0.4.1: gated CFG forward for FLUX.2.

    One shared polynomial-gate decision per step (mod_in is encoder-
    independent — see forward.py:258-304). Two cached residuals
    (positive + negative). CFG combination math runs after the tail on
    both branches.

    Replaces _vanilla_flux2_cfg_predict in production. The vanilla helper
    stays in flux2.py as a test-only diagnostic reference."""
    from mflux.models.common.config.model_config import ModelConfig

    state = handle._state.cache
    stats = handle._state.stats

    # 1. Shared prelude (mirrors flux2_forward_with_gate lines 367-394
    #    minus encoder_hidden_states handling, which is per-branch).
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
    if handle.rel_l1_thresh <= 0.0:
        body_out_pos = _flux2_run_body(
            inner, body_in, enc_pos, temb, temb_mod_params_img, temb_mod_params_txt, concat_rot_pos
        )
        body_out_neg = _flux2_run_body(
            inner, body_in, enc_neg, temb, temb_mod_params_img, temb_mod_params_txt, concat_rot_neg
        )
        stats.record(
            StepDecision(
                step_idx=state.step_counter,
                timestep=timestep_val,
                rel_l1=None,
                accumulated_distance=state.accumulated_distance,
                decision="computed",
            )
        )
        state.last_timestep = timestep_val
        state.step_counter += 1
        return _flux2_apply_tail_and_combine(
            inner, body_out_pos, body_out_neg, enc_pos, enc_neg, temb, guidance
        )

    # 4. Slow path: build mod_in, run gate ONCE on shared signal.
    mod_in = _flux2_extract_mod_input(inner, body_in, temb_mod_params_img)
    if state.previous_mod_input is not None and mod_in.shape != state.previous_mod_input.shape:
        raise TransformerShapeError(
            step_idx=state.step_counter,
            expected=state.previous_mod_input.shape,
            actual=mod_in.shape,
        )

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
    )
    stats.record(_step_decision_from_gate(decision, step_idx=state.step_counter, timestep=timestep_val))
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
        if decision.should_update_cache:
            state.cached_residual = body_out_pos - body_in_concat_pos
            state.cached_residual_neg = body_out_neg - body_in_concat_neg
    else:
        if state.cached_residual is None or state.cached_residual_neg is None:
            raise InternalStateError(
                "cached_residual or cached_residual_neg is None on a skipped CFG step; "
                "gate logic should guarantee seed-step caching before any skip."
            )
        body_out_pos = body_in_concat_pos + state.cached_residual
        body_out_neg = body_in_concat_neg + state.cached_residual_neg

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


# ---------- PORTED VERBATIM from src/mlx_teacache/integrations/mflux/flux2.py ----------

PredictFn = Callable[
    [mx.array, mx.array, mx.array, mx.array, "mx.array | None", "mx.array | None", float, mx.array],
    Any,
]
PredictFactory = Callable[[Any], PredictFn]


def make_teacache_predict_factory(handle: Any) -> PredictFactory:
    """Return a callable assignable to `flux._predict`. Mflux will call it
    as `predict = self._predict(self.transformer)` at the top of each
    generate_image."""
    from mlx_teacache.errors import InvalidStepWindowError, MissingGenerationContextError

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

            # 2. Lazy skip-window validation. v0.4.1: lifted up so it runs on
            #    the first gated call regardless of CFG. In v0.4.0 the CFG
            #    branch bypassed this; an all-CFG generation with a bad
            #    window silently ran vanilla. This is now a behavior change
            #    documented in CHANGELOG.
            if not handle._state.cache.skip_window_validated:
                if handle.skip_first_n_steps + handle.skip_last_n_steps >= ctx.active_num_steps:
                    raise InvalidStepWindowError(
                        skip_first=handle.skip_first_n_steps,
                        skip_last=handle.skip_last_n_steps,
                        num_steps=ctx.active_num_steps,
                    )
                handle._state.cache.skip_window_validated = True

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


# ---------- apply() — translated from v0.5.x api.py::apply_teacache FLUX.2 branch ----------


def apply(
    flux: Any,
    *,
    rel_l1_thresh: float | None = None,
    coefficients: tuple[float, float, float, float, float] | None = None,
    skip_first_n_steps: int = 1,
    skip_last_n_steps: int = 1,
) -> TeaCacheHandle:
    """FLUX.2 Klein base 4B apply. Public-API-equivalent of the FLUX.2 branch
    of v0.5.x apply_teacache.

    FLUX.2 wraps flux._predict (not flux.transformer). VariantPatch rollback
    deletes the _predict instance attribute; finalizer unsubscribes callback.
    No stats finalize in either (audit F2)."""
    # 1. Resolve rel_l1_thresh.
    # Priority: explicit caller > per-variant DEFAULT_THRESH (only when using
    # builtin coefficients) > package fallback 0.20.
    # User-supplied coefficients skip the per-variant default because it was
    # tuned for the bundled polynomial; a custom polynomial gets 0.20.
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
    _predict_was_instance_attr = "_predict" in vars(flux)
    _original_predict = flux.__dict__.get("_predict")
    flux._predict = make_teacache_predict_factory(internal)
    # No try needed: _predict assignment is the last mutation; fall through.

    # 7. Build VariantPatch: rollback deletes _predict + restores generate_image.
    #    Finalizer unsubscribes the callback. NO stats finalize (audit F2).
    def _restore_predict() -> None:
        if _predict_was_instance_attr:
            flux._predict = _original_predict
        else:
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
