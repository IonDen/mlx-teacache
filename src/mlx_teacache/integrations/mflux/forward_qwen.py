"""The shared Qwen-Image forward and apply, used by both Qwen-Image variants:
- QwenCalibration: the per-checkpoint values apply_qwen reads.
- ProxyQwenTransformer: an nn.Module standing in for flux.transformer.
- qwen_forward_with_gate: the gated forward. generate_image calls the transformer TWICE per
  step (positive then negative, external CFG combine), so the forward threads a CfgBranchPairer.
- _qwen_prelude, _qwen_signal_a, _qwen_run_body, _qwen_tail: the pieces of the Qwen
  transformer call that the gate splits apart (the calibration script reuses them).
- apply_qwen: installs the proxy for one calibration.

Imports no mflux: the forward only calls the transformer it wraps.
"""

import warnings
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

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
from mlx_teacache.errors import (
    InternalStateError,
    TeaCacheUncalibratedCheckpointWarning,
)
from mlx_teacache.handle import TeaCacheHandle
from mlx_teacache.integrations.mflux.lifecycle import resolve_active_num_steps
from mlx_teacache.integrations.mflux.patch import install_proxy_patch
from mlx_teacache.variants.qwen_image.detect import is_calibrated_checkpoint


@dataclass(frozen=True, slots=True, kw_only=True)
class QwenCalibration:
    """Everything the shared Qwen apply reads per checkpoint: the fit, the range it covers,
    the default operating point, where the fit came from and the checkpoint it was measured on."""

    variant_id: str
    coefficients: tuple[float, float, float, float, float]
    calibrated_range: tuple[float, float]
    default_thresh: float
    provenance: Provenance
    hf_model_id: str


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

    # Everything nn.Module walks through update()/children()/filter_and_map()
    # (load_weights, set_dtype, quantize, eval/train) must see the inner tree,
    # not the proxy's own empty dict. Signatures match mlx 0.31.2 nn/layers/base.py.
    def update(self, parameters: dict[str, Any], strict: bool = True) -> "ProxyQwenTransformer":
        self._inner.update(parameters, strict=strict)
        return self

    def update_modules(self, modules: dict[str, Any], strict: bool = True) -> "ProxyQwenTransformer":
        self._inner.update_modules(modules, strict=strict)
        return self

    def filter_and_map(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.filter_and_map(*args, **kwargs)

    def children(self) -> Any:
        return self._inner.children()

    def leaf_modules(self) -> Any:
        return self._inner.leaf_modules()

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = object.__getattribute__(self, "_inner")
            return getattr(inner, name)


# The old name, kept only for the variant module that still re-exports it.
_step_decision_from_gate = step_decision


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

    # Resolve the active denoising window and the nominal step count once — needed by
    # the skip-window validation below; the active count also drives the gate's
    # forced-window indexing (slow path).
    active_num_steps = resolve_active_num_steps(handle, config)
    nominal_num_inference_steps = config.num_inference_steps

    # Per-generation skip-window validation (once, on the first call). Mirrors
    # FLUX.1 / FLUX.2 / Z-Image: an over-wide skip window raises rather than
    # silently running at vanilla speed. Reset each generation by the lifecycle's
    # reset_for_new_generation (skip_window_validated=False).
    validate_window_once(
        state,
        active_num_steps=active_num_steps,
        skip_first_n_steps=handle.skip_first_n_steps,
        skip_last_n_steps=handle.skip_last_n_steps,
        nominal_num_inference_steps=nominal_num_inference_steps,
        first_step_only=True,
    )

    pre = _qwen_prelude(inner, t, config, hidden_states)

    # thresh<=0: never cache (mirrors gate_step's short-circuit); step recorded+advanced once per step, like the slow path.
    # Fast path: thresh <= 0 ⇒ always compute, never cache. Record + advance
    # ONCE per step (on the positive branch) to keep len(decisions)==num_steps.
    if is_disabled(handle.rel_l1_thresh):
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
            stats.record(fast_path_decision(state, timestep=float(t)))
        out = _qwen_tail(inner, body_out, pre)
        if not positive:
            state.step_counter += 1
        pairer.advance()
        return out

    if positive:
        stats._staging.cfg_was_active = True
        mod_in = _qwen_signal_a(inner, pre)
        check_signal_shape(state, mod_in)
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
        pairer.shared_decision = decision
        stats.record(step_decision(decision, step_idx=state.step_counter, timestep=float(t)))
        if decision.should_compute:
            body_out = _qwen_run_body(
                inner,
                pre,
                config=config,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                cond_image_grid=cond_image_grid,
            )
            cache_residuals(state, decision, pos=(body_out, pre.h_in))
        else:
            body_out = reconstruct(pre.h_in, state, decision)
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
        cache_residuals(state, decision, neg=(body_out, pre.h_in))
    else:
        body_out = reconstruct(pre.h_in, state, decision, branch="neg")
    out = _qwen_tail(inner, body_out, pre)
    state.step_counter += 1
    pairer.advance()
    return out


def apply_qwen(
    flux: Any,
    calibration: QwenCalibration,
    *,
    rel_l1_thresh: float | None = None,
    coefficients: tuple[float, float, float, float, float] | None = None,
    skip_first_n_steps: int = 1,
    skip_last_n_steps: int = 1,
) -> TeaCacheHandle:
    """Shared Qwen-Image apply: proxies flux.transformer and reads every per-checkpoint
    value (fit, range, default threshold, provenance, checkpoint id) from `calibration`."""
    builtin = BuiltinFit(
        coefficients=calibration.coefficients,
        provenance=calibration.provenance,
        default_thresh=calibration.default_thresh,
        calibrated_range=calibration.calibrated_range,
    )
    # Caller > the checkpoint's default with the built-in coefficients > the package fallback 0.20.
    rule = resolve_rule(
        rel_l1_thresh=rel_l1_thresh,
        coefficients=coefficients,
        builtin=builtin,
        policy="fallback-for-caller-coefficients",
    )
    if coefficients is None:
        # The built-in polynomial was fitted on calibration.hf_model_id. mflux 0.19
        # resolves the `qwen-image` alias to Qwen/Qwen-Image-2512, a newer
        # checkpoint with the same aliases, so the detector still matches and
        # the user would otherwise get an unverified operating point silently.
        # mflux resolves a local path or a pre-quantized mirror by copying the base
        # config and recording the base's model_name in base_model, so a mirror
        # declared through --base-model counts as the calibrated checkpoint.
        model_config = getattr(flux, "model_config", None)
        loaded = getattr(model_config, "model_name", None)
        calibrated = calibration.hf_model_id
        if not is_calibrated_checkpoint(model_config, calibrated):
            # stacklevel 4: apply_qwen -> variant apply -> apply_teacache -> the caller.
            warnings.warn(
                TeaCacheUncalibratedCheckpointWarning(
                    f"variant {calibration.variant_id!r} loaded {loaded!r}, but its coefficients were "
                    f"calibrated on {calibrated!r}; skip counts and image quality on this "
                    "checkpoint are not verified. If this is a mirror of the calibrated "
                    "checkpoint, load it with that base model declared or filter this "
                    "category; otherwise pass coefficients= from your own calibration "
                    "(scripts/calibrate_qwen.py)."
                ),
                stacklevel=4,
            )

    return install_proxy_patch(
        flux,
        make_proxy=ProxyQwenTransformer,
        rule=rule,
        skip_first_n_steps=skip_first_n_steps,
        skip_last_n_steps=skip_last_n_steps,
    )
