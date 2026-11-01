"""Z-Image base integration — self-contained TeaCache mini-kernel.

Re-walks ZImageTransformer.__call__ (mflux 0.17.5, z_image transformer.py:57-139)
with a TeaCache gate. The gate signal is Signal B: the first-main-layer residual
rel-L1 (calibrated 2026-05-31; see config.py). The cache stores the 30-main-layer
residual `main_out - unified_in` per CFG branch; a skipped step reconstructs
`main_out = unified_in + cached_residual` (so the current step's prelude flows
through). The timestep-only adaLN modulation rules out a cheap caption-independent
prelude signal, so Signal B is tapped one layer into the unified stream.

No sibling-variant imports: this variant defines its own forward, depending
only on the model-agnostic _kernel/, the public handle, the shared internal
session and the shared mflux lifecycle helpers. mflux is imported lazily — the registry
loads this module only after detect.matches() wins.

Vanilla Z-Image wraps `_predict` in `mx.compile` (eager on base and Pro M1/M2). We
replace `_predict` with an eager factory so the per-step gate runs every step;
threshold=0 parity vs vanilla is therefore cosine, not bit-exact (see
tests/test_parity_z_image.py).
"""

from collections.abc import Callable
from typing import Any, cast

import mlx.core as mx

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.gate import gate_step
from mlx_teacache._kernel.resolve import BuiltinFit, resolve_rule
from mlx_teacache._kernel.step import (
    cache_residuals,
    check_signal_shape,
    fast_path_decision,
    is_disabled,
    reconstruct,
    step_decision,
    validate_window_once,
)
from mlx_teacache.handle import TeaCacheHandle
from mlx_teacache.integrations._session import InternalHandle
from mlx_teacache.integrations.mflux.patch import install_predict_patch

from .config import CALIBRATED_RANGE, COEFFICIENTS, DEFAULT_THRESH

# The shared session type under the name the tests still import from here.
_InternalHandle = InternalHandle


_PROVENANCE = Provenance(
    source="builtin",
    revision="in-repo-2026-05-31-origin-signalB",
    calibration_dataset=(
        "10 prompts (7 fit / 3 held-out) × 50 steps × seed=42, M1 Max 32GB, q8, "
        "512x512, guidance=4.0 (CFG), origin-constrained polyfit"
    ),
    fit_metric=(
        "constrained-LSQ R^2 on consecutive-step (signal-B first-main-layer-residual, "
        "worst-branch main_out) rel-L1 pairs (poly(0)=0)"
    ),
    fit_metric_value=0.3997548048012458,
    reference_url="https://github.com/IonDen/mlx-teacache/blob/main/scripts/calibrate_z_image.py",
    default_thresh=DEFAULT_THRESH,
)


# ---------- Re-walk of ZImageTransformer.__call__ (transformer.py:57-139). ----------
# Mirrors the calibration capture forward, split so the
# main-layer loop can stop after layer 0 on a skip step.


def _zimage_t_emb(transformer: Any, timestep: Any, sigmas: mx.array) -> mx.array:
    """Replicate the timestep -> t_emb path (mflux 0.17.5 transformer.py:66-75; builds with mflux PR 803 add the
    stream_t_emb cast)."""
    if not isinstance(timestep, mx.array):
        if isinstance(timestep, int):
            sigma_t = sigmas[timestep].reshape((1,))
            timestep = mx.ones_like(sigma_t) - sigma_t
        else:
            timestep = mx.array(timestep, dtype=mx.float32)
    if timestep.ndim == 0:
        timestep = timestep.reshape((1,))
    t_emb: mx.array = transformer.t_embedder(timestep.astype(mx.float32) * transformer.t_scale)
    # mflux builds after 0.21.0 run the hidden stream in bfloat16 and cast t_emb through this
    # staticmethod (honouring the transformer's `float32` flag); earlier releases have none.
    stream_t_emb = getattr(transformer, "stream_t_emb", None)
    if callable(stream_t_emb):
        t_emb = stream_t_emb(t_emb, getattr(transformer, "float32", False))
    return t_emb


class _Prelude:
    """Per-branch prelude outputs: the unified-stream input + everything the
    main-layer loop and the tail need. unified_in is the residual base."""

    __slots__ = ("unified_in", "freqs_cis", "attn_mask", "x_len", "x_size")

    def __init__(
        self,
        *,
        unified_in: mx.array,
        freqs_cis: mx.array,
        attn_mask: mx.array,
        x_len: int,
        x_size: Any,
    ) -> None:
        self.unified_in = unified_in
        self.freqs_cis = freqs_cis
        self.attn_mask = attn_mask
        self.x_len = x_len
        self.x_size = x_size


def _zimage_prelude(transformer: Any, latents: mx.array, t_emb: mx.array, cap_feats: mx.array) -> _Prelude:
    """Patchify -> image embed + noise refiner -> caption embed + context refiner
    -> unify. Mirrors transformer.py:78-120 verbatim (no taps)."""
    ZImageTransformer = type(transformer)
    key = f"{transformer.patch_size}-{transformer.f_patch_size}"

    x_emb, cap_emb, x_size, x_pos_ids, cap_pos_ids, x_pad_mask, cap_pad_mask = ZImageTransformer._patchify(
        image=latents,
        cap_feats=cap_feats,
        patch_size=transformer.patch_size,
        f_patch_size=transformer.f_patch_size,
    )
    x_emb = transformer.all_x_embedder[key](x_emb)
    x_emb = mx.where(x_pad_mask[:, None], transformer.x_pad_token, x_emb)
    x_freqs_cis = transformer.rope_embedder(x_pos_ids)
    x_attn_mask = mx.ones((1, x_emb.shape[0]), dtype=mx.bool_)
    x_emb = mx.expand_dims(x_emb, axis=0)
    for layer in transformer.noise_refiner:
        x_emb = layer(x=x_emb, attn_mask=x_attn_mask, freqs_cis=x_freqs_cis, t_emb=t_emb)

    cap_emb = transformer.cap_embedder[1](transformer.cap_embedder[0](cap_emb))
    cap_emb = mx.where(cap_pad_mask[:, None], transformer.cap_pad_token, cap_emb)
    cap_freqs_cis = transformer.rope_embedder(cap_pos_ids)
    cap_attn_mask = mx.ones((1, cap_emb.shape[0]), dtype=mx.bool_)
    cap_emb = mx.expand_dims(cap_emb, axis=0)
    for layer in transformer.context_refiner:
        cap_emb = layer(x=cap_emb, attn_mask=cap_attn_mask, freqs_cis=cap_freqs_cis)

    x_len = x_emb.shape[1]
    unified_in = mx.concatenate([x_emb, cap_emb], axis=1)
    unified_freqs_cis = mx.concatenate([x_freqs_cis, cap_freqs_cis], axis=0)
    unified_attn_mask = mx.ones((1, unified_in.shape[1]), dtype=mx.bool_)
    return _Prelude(
        unified_in=unified_in,
        freqs_cis=unified_freqs_cis,
        attn_mask=unified_attn_mask,
        x_len=x_len,
        x_size=x_size,
    )


def _run_main_layers(
    transformer: Any, h: mx.array, pre: _Prelude, t_emb: mx.array, *, start: int
) -> mx.array:
    """Run transformer.layers[start:] over the unified stream (transformer.py:122-128)."""
    for layer in transformer.layers[start:]:
        h = layer(x=h, attn_mask=pre.attn_mask, freqs_cis=pre.freqs_cis, t_emb=t_emb)
    return h


def _zimage_tail(transformer: Any, main_out: mx.array, t_emb: mx.array, pre: _Prelude) -> mx.array:
    """Final layer + unpatchify + negation (transformer.py:130-139). Returns the
    per-branch noise prediction (-output)."""
    ZImageTransformer = type(transformer)
    key = f"{transformer.patch_size}-{transformer.f_patch_size}"
    final = transformer.all_final_layer[key](main_out, t_emb)
    output = ZImageTransformer._unpatchify(
        x=final[0, : pre.x_len],
        size=pre.x_size,
        patch_size=transformer.patch_size,
        f_patch_size=transformer.f_patch_size,
        out_channels=transformer.out_channels,
    )
    noise: mx.array = -output
    return noise


# ---------- Gated forwards ----------


def zimage_forward_with_gate(
    transformer: Any,
    handle: Any,
    *,
    latents: mx.array,
    timestep: mx.array,
    sigmas: mx.array,
    cap_feats: mx.array,
) -> mx.array:
    """Non-CFG gated forward (guidance <= 1.0; negative_encodings is None)."""
    state = handle._state.cache
    stats = handle._state.stats
    t_emb = _zimage_t_emb(transformer, timestep, sigmas)
    timestep_val = float(timestep.flatten()[0])
    pre = _zimage_prelude(transformer, latents, t_emb, cap_feats)

    # Fast path: threshold <= 0 ⇒ always compute, never cache. Run the full
    # 30-layer body in one loop (closest topology to vanilla __call__).
    if is_disabled(handle.rel_l1_thresh):
        main_out = _run_main_layers(transformer, pre.unified_in, pre, t_emb, start=0)
        stats.record(fast_path_decision(state, timestep=timestep_val))
        state.last_timestep = timestep_val
        state.step_counter += 1
        return _zimage_tail(transformer, main_out, t_emb, pre)

    # Slow path: run layer 0 to get Signal B (the gate signal).
    h1 = transformer.layers[0](
        x=pre.unified_in, attn_mask=pre.attn_mask, freqs_cis=pre.freqs_cis, t_emb=t_emb
    )
    signal_b = h1 - pre.unified_in
    check_signal_shape(state, signal_b)

    decision = gate_step(
        state,
        rel_l1_thresh=handle.rel_l1_thresh,
        coefficients=handle.coefficients,
        skip_first=handle.skip_first_n_steps,
        skip_last=handle.skip_last_n_steps,
        num_steps=handle._gen_ctx.active_num_steps,
        step_idx=state.step_counter,
        mod_in=signal_b,
        calibrated_range=handle.calibrated_range,
    )
    stats.record(step_decision(decision, step_idx=state.step_counter, timestep=timestep_val))
    state.last_timestep = timestep_val

    if decision.should_compute:
        main_out = _run_main_layers(transformer, h1, pre, t_emb, start=1)
        cache_residuals(state, decision, pos=(main_out, pre.unified_in))
    else:
        main_out = reconstruct(pre.unified_in, state, decision)

    state.step_counter += 1
    return _zimage_tail(transformer, main_out, t_emb, pre)


def zimage_cfg_forward_with_gate(
    transformer: Any,
    handle: Any,
    *,
    latents: mx.array,
    timestep: mx.array,
    sigmas: mx.array,
    cap_feats_pos: mx.array,
    cap_feats_neg: mx.array,
    guidance: float,
) -> mx.array:
    """CFG gated forward (guidance > 1.0). One shared gate decision driven by the
    positive branch's Signal B (calibrated on branch='pos'); two cached residuals.
    CFG combine matches z_image.py:209 exactly: pos + g*(pos - neg)."""
    state = handle._state.cache
    stats = handle._state.stats
    t_emb = _zimage_t_emb(transformer, timestep, sigmas)
    timestep_val = float(timestep.flatten()[0])
    pre_pos = _zimage_prelude(transformer, latents, t_emb, cap_feats_pos)
    pre_neg = _zimage_prelude(transformer, latents, t_emb, cap_feats_neg)

    # Fast path: run both full bodies, no caching.
    if is_disabled(handle.rel_l1_thresh):
        main_out_pos = _run_main_layers(transformer, pre_pos.unified_in, pre_pos, t_emb, start=0)
        main_out_neg = _run_main_layers(transformer, pre_neg.unified_in, pre_neg, t_emb, start=0)
        stats.record(fast_path_decision(state, timestep=timestep_val))
        state.last_timestep = timestep_val
        state.step_counter += 1
        noise_pos = _zimage_tail(transformer, main_out_pos, t_emb, pre_pos)
        noise_neg = _zimage_tail(transformer, main_out_neg, t_emb, pre_neg)
        return noise_pos + guidance * (noise_pos - noise_neg)

    # Slow path: layer 0 on the POSITIVE branch supplies the shared gate signal.
    h1_pos = transformer.layers[0](
        x=pre_pos.unified_in, attn_mask=pre_pos.attn_mask, freqs_cis=pre_pos.freqs_cis, t_emb=t_emb
    )
    signal_b = h1_pos - pre_pos.unified_in
    check_signal_shape(state, signal_b)

    decision = gate_step(
        state,
        rel_l1_thresh=handle.rel_l1_thresh,
        coefficients=handle.coefficients,
        skip_first=handle.skip_first_n_steps,
        skip_last=handle.skip_last_n_steps,
        num_steps=handle._gen_ctx.active_num_steps,
        step_idx=state.step_counter,
        mod_in=signal_b,
        calibrated_range=handle.calibrated_range,
    )
    stats.record(step_decision(decision, step_idx=state.step_counter, timestep=timestep_val))
    state.last_timestep = timestep_val

    if decision.should_compute:
        main_out_pos = _run_main_layers(transformer, h1_pos, pre_pos, t_emb, start=1)
        # Negative branch's layer 0 is only needed on a compute step.
        h1_neg = transformer.layers[0](
            x=pre_neg.unified_in, attn_mask=pre_neg.attn_mask, freqs_cis=pre_neg.freqs_cis, t_emb=t_emb
        )
        main_out_neg = _run_main_layers(transformer, h1_neg, pre_neg, t_emb, start=1)
        cache_residuals(
            state,
            decision,
            pos=(main_out_pos, pre_pos.unified_in),
            neg=(main_out_neg, pre_neg.unified_in),
        )
    else:
        main_out_pos = reconstruct(pre_pos.unified_in, state, decision)
        main_out_neg = reconstruct(pre_neg.unified_in, state, decision, branch="neg")

    state.step_counter += 1
    noise_pos = _zimage_tail(transformer, main_out_pos, t_emb, pre_pos)
    noise_neg = _zimage_tail(transformer, main_out_neg, t_emb, pre_neg)
    return noise_pos + guidance * (noise_pos - noise_neg)


# ---------- predict factory (replaces ZImage._predict) ----------

PredictFn = Callable[[mx.array, mx.array, mx.array, mx.array, "mx.array | None", float], Any]
PredictFactory = Callable[[Any], PredictFn]


def make_teacache_predict_factory(handle: Any) -> PredictFactory:
    """Return a callable assignable to `flux._predict`. mflux calls it as
    `predict = self._predict(self.transformer)` at the top of generate_image.
    Returns an EAGER closure (no mx.compile) so the per-step gate runs every step."""
    from mlx_teacache.errors import MissingGenerationContextError

    def predict_factory(transformer: Any) -> PredictFn:
        context_consumed = False

        def predict(
            latents: mx.array,
            timestep: mx.array,
            sigmas: mx.array,
            text_encodings: mx.array,
            negative_encodings: mx.array | None,
            guidance: float,
        ) -> Any:
            nonlocal context_consumed
            ctx = handle._gen_ctx

            # 1. Consume context on the first call of THIS generation's closure.
            if not context_consumed:
                if ctx.active_num_steps is None or ctx.consumed_at_token == ctx.token:
                    raise MissingGenerationContextError()
                ctx.consumed_at_token = ctx.token
                context_consumed = True

            # 2. Lazy skip-window validation (first gated call).
            validate_window_once(
                handle._state.cache,
                active_num_steps=cast(int, handle._gen_ctx.active_num_steps),
                skip_first_n_steps=handle.skip_first_n_steps,
                skip_last_n_steps=handle.skip_last_n_steps,
            )

            # 3. CFG branch.
            if negative_encodings is not None:
                handle._state.stats._staging.cfg_was_active = True
                return zimage_cfg_forward_with_gate(
                    transformer,
                    handle,
                    latents=latents,
                    timestep=timestep,
                    sigmas=sigmas,
                    cap_feats_pos=text_encodings,
                    cap_feats_neg=negative_encodings,
                    guidance=guidance,
                )

            # 4. Non-CFG branch.
            return zimage_forward_with_gate(
                transformer,
                handle,
                latents=latents,
                timestep=timestep,
                sigmas=sigmas,
                cap_feats=text_encodings,
            )

        return predict

    return predict_factory


# ---------- apply() ----------

_BUILTIN = BuiltinFit(
    coefficients=COEFFICIENTS,
    provenance=_PROVENANCE,
    default_thresh=DEFAULT_THRESH,
    calibrated_range=CALIBRATED_RANGE,
)


def apply(
    flux: Any,
    *,
    rel_l1_thresh: float | None = None,
    coefficients: tuple[float, float, float, float, float] | None = None,
    skip_first_n_steps: int = 1,
    skip_last_n_steps: int = 1,
) -> TeaCacheHandle:
    """Z-Image base apply. Wraps flux._predict (instance-attribute replacement);
    rollback deletes it so the class staticmethod is exposed again. Finalizer
    unsubscribes the lifecycle callback."""
    # Caller > the variant default with the built-in coefficients > the package fallback 0.20. A custom polynomial
    # gets the fallback because the default was tuned for the bundled one.
    rule = resolve_rule(
        rel_l1_thresh=rel_l1_thresh,
        coefficients=coefficients,
        builtin=_BUILTIN,
        policy="fallback-for-caller-coefficients",
    )
    return install_predict_patch(
        flux,
        make_factory=make_teacache_predict_factory,
        rule=rule,
        skip_first_n_steps=skip_first_n_steps,
        skip_last_n_steps=skip_last_n_steps,
    )
