"""Before/after equivalence oracle for refactors that must not change behaviour.

`collect()` drives only seams that a refactor of the integration layer keeps: `apply_teacache`, the installed
`flux.transformer` / `flux._predict`, the public handle and its stats, and the internal handle reached through
`handle._callback_instance._handle`. It returns one JSON-ready dict of exact values: floats as `float.hex`, arrays
as dtype, shape, the sha256 of their float32 bytes and their first 16 values. Two records made from two commits are
then compared key by key; any difference is a behaviour change.

Command line, from the repository root:

    python -m tests._equivalence_oracle record --out PATH [--device gpu|cpu]
    python -m tests._equivalence_oracle compare A B

`compare` exits 0 and prints `identical (N scenarios)`, or exits 1 and prints each differing key path.
`meta.module_file` (which library was imported) and `meta.recorded_at` are not compared; `meta.harness_sha256` is,
so both sides must run the same harness. It covers this file and the fakes it builds models from (`_variant_fakes.py`,
`_fakes.py`).

It imports only seams that 0.16.0 already had, so the same file records a 0.16.0 tree and a later one.

Test utility only: nothing in `src/` imports it. It needs the `[mflux]` extra and loads no weights.
"""

import argparse
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import os
import sys
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlx.core as mx

_HARNESS = Path(__file__)
# Everything a record depends on besides the library: the oracle and the fakes it builds models from.
_HARNESS_FILES: tuple[Path, ...] = (
    _HARNESS,
    _HARNESS.with_name("_variant_fakes.py"),
    _HARNESS.with_name("_fakes.py"),
)
_NOT_COMPARED = frozenset({"meta/module_file", "meta/recorded_at"})
_CALLBACK_LISTS = ("before_loop", "in_loop", "after_loop", "interrupt")

# Threshold matrix inputs: every registry id x these thresholds x these coefficients.
_MATRIX_THRESHOLDS: tuple[float | None, ...] = (None, 0.0, 0.33)
_MATRIX_COEFFICIENTS: dict[str, tuple[float, float, float, float, float] | None] = {
    "builtin": None,
    "caller": (0.001, 0.0, 0.0, 0.5, 0.0),
}


# ---------- encoding ----------


def _digest(array: mx.array) -> dict[str, Any]:
    as_f32 = array.astype(mx.float32)
    mx.eval(as_f32)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256_f32": hashlib.sha256(bytes(memoryview(as_f32))).hexdigest(),
        "first16": [float(v).hex() for v in as_f32.reshape(-1)[:16].tolist()],
    }


def _enc(value: Any) -> Any:
    """Exact JSON form: bool/int/str/None as is, floats as float.hex, sequences as lists, arrays as digests."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, mx.array):
        return _digest(value)
    if isinstance(value, tuple | list):
        return [_enc(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _enc(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _enc(dataclasses.asdict(value))
    raise TypeError(f"no exact encoding for {type(value).__name__}")


def _warning_row(w: warnings.WarningMessage) -> list[Any]:
    return [w.category.__name__, str(w.message), os.path.basename(w.filename), w.lineno]


@contextmanager
def _recording() -> Iterator[list[warnings.WarningMessage]]:
    """Record every warning, once-per-process ones included, and re-arm the untested-mflux warning so each
    scenario sees the same warnings whatever ran before it."""
    from mlx_teacache.testing import reset_untested_mflux_warning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reset_untested_mflux_warning()
        yield caught


# ---------- what a handle and a teardown look like ----------


def _internal(handle: Any) -> Any:
    return handle._callback_instance._handle


def _handle_record(handle: Any) -> dict[str, Any]:
    internal = _internal(handle)
    return {
        "variant_id": handle.variant_id,
        "rel_l1_thresh": _enc(handle.rel_l1_thresh),
        "coefficients": _enc(handle.coefficients),
        "provenance": _enc(handle.provenance),
        "skip_first_n_steps": handle.skip_first_n_steps,
        "skip_last_n_steps": handle.skip_last_n_steps,
        "decision_rule_revision": handle.decision_rule_revision,
        "mflux_version": handle.mflux_version,
        "internal": {
            "rel_l1_thresh": _enc(internal.rel_l1_thresh),
            "coefficients": _enc(internal.coefficients),
            "calibrated_range": _enc(internal.calibrated_range),
            "skip_first_n_steps": internal.skip_first_n_steps,
            "skip_last_n_steps": internal.skip_last_n_steps,
        },
    }


def _teardown_record(flux: Any, original_transformer: Any, internal: Any) -> dict[str, Any]:
    cache = internal._state.cache
    return {
        "vars": sorted(vars(flux)),
        "callbacks": {name: len(getattr(flux.callbacks, name)) for name in _CALLBACK_LISTS},
        "transformer_is_original": flux.transformer is original_transformer,
        "cache_arrays_none": [
            cache.previous_mod_input is None,
            cache.cached_residual is None,
            cache.cached_residual_neg is None,
        ],
    }


# ---------- threshold matrix ----------


def _matrix_cell(variant_id: str, thresh: float | None, coefficients: Any) -> dict[str, Any]:
    from mlx_teacache import apply_teacache
    from tests._variant_fakes import fake_for_variant

    flux = fake_for_variant(variant_id)
    original_transformer = flux.transformer
    with _recording() as caught:
        handle = apply_teacache(flux, rel_l1_thresh=thresh, coefficients=coefficients)
    cell = {"handle": _handle_record(handle), "warnings": [_warning_row(w) for w in caught]}
    internal = _internal(handle)
    with _recording() as caught:
        handle.restore()
    cell["teardown"] = _teardown_record(flux, original_transformer, internal)
    cell["teardown_warnings"] = [_warning_row(w) for w in caught]
    return cell


def _threshold_matrix() -> dict[str, Any]:
    from mlx_teacache.variants import _REGISTRY

    matrix: dict[str, Any] = {}
    for variant_id in sorted(_REGISTRY):
        matrix[variant_id] = {
            f"thresh={thresh},coeffs={name}": _matrix_cell(variant_id, thresh, coefficients)
            for thresh in _MATRIX_THRESHOLDS
            for name, coefficients in _MATRIX_COEFFICIENTS.items()
        }
    return matrix


def _qwen_uncalibrated_checkpoint() -> dict[str, Any]:
    """Qwen-Image on a checkpoint its coefficients were not fitted on: the warning must name the caller's line."""
    from types import SimpleNamespace

    from mlx_teacache import apply_teacache
    from tests._variant_fakes import fake_for_variant

    flux = fake_for_variant("qwen-image")
    flux.model_config = SimpleNamespace(
        model_name="someone/qwen-image-finetune", base_model=None, aliases=["qwen-image"]
    )
    with _recording() as caught:
        handle = apply_teacache(flux)
    record = {"handle": _handle_record(handle), "warnings": [_warning_row(w) for w in caught]}
    handle.restore()
    return record


# ---------- generation pipelines ----------
#
# Each pipeline follows mflux's generate_image order: before-loop callbacks, one transformer or predict call per
# step (two transformer calls for Qwen-Image), an Euler-style `latents - 0.05 * noise` evaluated every step, then
# after-loop callbacks. The pipeline classes live outside mflux, so the detectors match them by alias.

_STEPS = 10
_STEP_SIZE = 0.05
_CFG_GUIDANCE = 4.0
# Every gated step costs a constant 0.1, whatever the signal: at threshold 0.25 the skip pattern is fixed
# (forced, seed, skip, skip, compute, skip, skip, compute, skip, forced over 10 steps).
_CONST_COEFFICIENTS = (0.0, 0.0, 0.0, 0.0, 0.1)


@dataclasses.dataclass(frozen=True)
class _Request:
    seed: int
    num_inference_steps: int = _STEPS
    init_time_step: int = 0
    raise_at_step: int | None = None
    shape_change_at_step: int | None = None
    skip_before_loop: bool = False


def _sigmas(num_inference_steps: int) -> mx.array:
    return mx.linspace(1.0, 0.0, num_inference_steps + 1)


class _Pipeline:
    latent_shape: tuple[int, ...] = ()
    changed_latent_shape: tuple[int, ...] = ()
    dtype: Any = mx.bfloat16

    def __init__(self, model_config: Any, transformer: Any, *, cfg: bool) -> None:
        from tests._fakes import FaithfulCallbackRegistry

        self.model_config = model_config
        self.transformer = transformer
        self.callbacks = FaithfulCallbackRegistry()
        self.cfg = cfg
        self.trace: list[list[dict[str, Any]]] = []

    def _config(self, request: _Request) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            num_inference_steps=request.num_inference_steps,
            init_time_step=request.init_time_step,
            sigmas=_sigmas(request.num_inference_steps),
        )

    def _latents(self, shape: tuple[int, ...]) -> mx.array:
        return mx.random.normal(shape).astype(self.dtype)

    def _step_function(self) -> Callable[[int, Any, mx.array], tuple[mx.array, list[mx.array]]]:
        raise NotImplementedError

    def generate_image(self, request: _Request) -> mx.array:
        config = self._config(request)
        mx.random.seed(request.seed)
        latents = self._latents(self.latent_shape)
        if not request.skip_before_loop:
            for cb in self.callbacks.before_loop:
                cb.call_before_loop(seed=request.seed, prompt="oracle", latents=latents, config=config)
        step = self._step_function()
        for t in range(request.init_time_step, request.num_inference_steps):
            if t == request.shape_change_at_step:
                latents = self._latents(self.changed_latent_shape)
            noise, outputs = step(t, config, latents)
            self.trace.append([_digest(o) for o in outputs])
            if t == request.raise_at_step:
                raise RuntimeError(f"generation failed on purpose at step {t}")
            latents = latents - _STEP_SIZE * noise
            mx.eval(latents)
        for cb in self.callbacks.after_loop:
            cb.call_after_loop(seed=request.seed, prompt="oracle", latents=latents, config=config)
        return latents


def _controlnet_sample(idx: int, blocks: list[Any], samples: list[mx.array] | None) -> mx.array | None:
    """mflux Transformer._get_controlnet_sample: block idx reads sample idx // ceil(blocks / samples)."""
    if not samples:
        return None
    interval = -(-len(blocks) // len(samples))
    return samples[idx // interval]


class _SurrogateFlux1JointBlock:
    """A joint block with its own weights; the rotary shifts both streams, so a lost or swapped rotary shows."""

    def __init__(self, dim: int) -> None:
        self.w_hidden = mx.random.normal((dim, dim)) * 0.3
        self.w_context = mx.random.normal((dim, dim)) * 0.3

    def norm1(self, hidden_states: mx.array, text_embeddings: mx.array) -> tuple[Any, ...]:
        return (mx.tanh(hidden_states * (1.0 + text_embeddings[:, None, :])), None, None, None, None)

    def __call__(
        self,
        *,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
    ) -> tuple[mx.array, mx.array]:
        temb = text_embeddings[:, None, :]
        hidden = hidden_states + mx.tanh((hidden_states + rotary_embeddings) @ self.w_hidden + temb)
        encoder = encoder_hidden_states + mx.tanh(
            (encoder_hidden_states - rotary_embeddings) @ self.w_context + temb
        )
        return encoder, hidden


class _SurrogateFlux1SingleBlock:
    def __init__(self, dim: int) -> None:
        self.w = mx.random.normal((dim, dim)) * 0.3

    def __call__(
        self, *, hidden_states: mx.array, text_embeddings: mx.array, rotary_embeddings: mx.array
    ) -> mx.array:
        return hidden_states + mx.tanh(
            (hidden_states * (1.0 + rotary_embeddings)) @ self.w + text_embeddings[:, None, :]
        )


class _SurrogateFlux1Transformer:
    """The surface of mflux's FLUX.1 Transformer that the gated forward touches, with seeded non-zero weights.
    mflux's own class fixes the width at 3072, so a weight-free copy of it is not tiny. Two joint and two single
    blocks, each with its own weights; a rotary that depends on the prompt length, `pos_embed` and the kontext ids;
    and block helpers that take exactly mflux's keywords and add the controlnet samples the way mflux does. So a
    forward that drops or misroutes one of those inputs, or runs only block 0, changes the output."""

    dim = 16

    def __init__(self) -> None:
        d = self.dim
        self.w_x = mx.random.normal((8, d)) * 0.5
        self.w_ctx = mx.random.normal((12, d)) * 0.5
        self.w_pool = mx.random.normal((12, d)) * 0.5
        self.w_time = mx.random.normal((d,))
        self.w_out = mx.random.normal((d, 8)) * 0.5
        self.transformer_blocks: list[Any] = [_SurrogateFlux1JointBlock(d), _SurrogateFlux1JointBlock(d)]
        self.single_transformer_blocks: list[Any] = [
            _SurrogateFlux1SingleBlock(d),
            _SurrogateFlux1SingleBlock(d),
        ]
        self.pos_embed = 0.5 + mx.random.uniform(shape=(d,))
        self.time_text_embed = object()

    def x_embedder(self, hidden_states: mx.array) -> mx.array:
        return hidden_states @ self.w_x

    def context_embedder(self, prompt_embeds: mx.array) -> mx.array:
        return prompt_embeds @ self.w_ctx

    def compute_text_embeddings(
        self, t: int, pooled: mx.array, time_text_embed: Any, config: Any
    ) -> mx.array:
        return pooled @ self.w_pool + config.sigmas[t] * self.w_time

    def compute_rotary_embeddings(
        self, prompt_embeds: mx.array, pos_embed: Any, config: Any, kontext_image_ids: Any
    ) -> mx.array:
        position = mx.array(float(prompt_embeds.shape[1]))
        if kontext_image_ids is not None:
            position = position + 0.25 * mx.sum(kontext_image_ids)
        freqs = 0.1 + 0.3 * mx.arange(self.dim)
        return (0.5 * mx.cos(freqs * position) * pos_embed).reshape(1, 1, self.dim)

    def _apply_joint_transformer_block(
        self,
        *,
        idx: int,
        block: Any,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: mx.array,
        controlnet_block_samples: list[mx.array] | None,
    ) -> tuple[mx.array, mx.array]:
        encoder, hidden = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=image_rotary_embeddings,
        )
        sample = _controlnet_sample(idx, self.transformer_blocks, controlnet_block_samples)
        return encoder, hidden if sample is None else hidden + sample

    def _apply_single_transformer_block(
        self,
        *,
        idx: int,
        block: Any,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: mx.array,
        controlnet_single_block_samples: list[mx.array] | None,
    ) -> mx.array:
        hidden = block(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=image_rotary_embeddings,
        )
        sample = _controlnet_sample(idx, self.single_transformer_blocks, controlnet_single_block_samples)
        if sample is None:
            return hidden
        n_text = encoder_hidden_states.shape[1]
        return mx.concatenate([hidden[:, :n_text, ...], hidden[:, n_text:, ...] + sample], axis=1)

    def norm_out(self, x: mx.array, text_embeddings: mx.array) -> mx.array:
        return x * (1.0 + 0.1 * text_embeddings[:, None, :])

    def proj_out(self, x: mx.array) -> mx.array:
        return x @ self.w_out

    def __call__(
        self,
        *,
        t: int,
        config: Any,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        controlnet_block_samples: list[mx.array] | None = None,
        controlnet_single_block_samples: list[mx.array] | None = None,
        kontext_image_ids: mx.array | None = None,
    ) -> mx.array:
        """mflux Transformer.__call__, in mflux's order."""
        hidden = self.x_embedder(hidden_states)
        encoder = self.context_embedder(prompt_embeds)
        temb = self.compute_text_embeddings(t, pooled_prompt_embeds, self.time_text_embed, config)
        rotary = self.compute_rotary_embeddings(prompt_embeds, self.pos_embed, config, kontext_image_ids)
        for idx, block in enumerate(self.transformer_blocks):
            encoder, hidden = self._apply_joint_transformer_block(
                idx=idx,
                block=block,
                hidden_states=hidden,
                encoder_hidden_states=encoder,
                text_embeddings=temb,
                image_rotary_embeddings=rotary,
                controlnet_block_samples=controlnet_block_samples,
            )
        hidden = mx.concatenate([encoder, hidden], axis=1)
        for idx, block in enumerate(self.single_transformer_blocks):
            hidden = self._apply_single_transformer_block(
                idx=idx,
                block=block,
                hidden_states=hidden,
                encoder_hidden_states=encoder,
                text_embeddings=temb,
                image_rotary_embeddings=rotary,
                controlnet_single_block_samples=controlnet_single_block_samples,
            )
        return self.proj_out(self.norm_out(hidden[:, encoder.shape[1] :, ...], temb))


class _Flux1Pipeline(_Pipeline):
    latent_shape = (1, 16, 8)
    changed_latent_shape = (1, 24, 8)
    dtype = mx.float32

    def __init__(self, model_config: Any, *, cfg: bool) -> None:
        super().__init__(model_config, _SurrogateFlux1Transformer(), cfg=cfg)
        d = _SurrogateFlux1Transformer.dim
        self.prompt = mx.random.normal((1, 4, 12))
        self.pooled = mx.random.normal((1, 12))
        # Every FLUX.1 call carries the inputs mflux passes for Kontext and ControlNet, one distinct sample per block,
        # so the record sees a forward that drops, swaps or reorders them.
        self.extra_inputs: dict[str, Any] = {
            "controlnet_block_samples": [mx.random.normal((1, 1, d)) * 0.2 for _ in range(2)],
            "controlnet_single_block_samples": [mx.random.normal((1, 1, d)) * 0.2 for _ in range(2)],
            "kontext_image_ids": mx.random.uniform(shape=(1, 3, 3)),
        }

    def _step_function(self) -> Callable[[int, Any, mx.array], tuple[mx.array, list[mx.array]]]:
        def step(t: int, config: Any, latents: mx.array) -> tuple[mx.array, list[mx.array]]:
            noise = self.transformer(
                t=t,
                config=config,
                hidden_states=latents,
                prompt_embeds=self.prompt,
                pooled_prompt_embeds=self.pooled,
                **self.extra_inputs,
            )
            return noise, [noise]

        return step


def _flux2_ids(tokens: int) -> mx.array:
    return mx.stack([mx.zeros(tokens), mx.arange(tokens), mx.zeros(tokens), mx.zeros(tokens)], axis=-1)


class _Flux2Pipeline(_Pipeline):
    latent_shape = (1, 6, 8)
    changed_latent_shape = (1, 8, 8)

    def __init__(self, model_config: Any, *, cfg: bool) -> None:
        from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer

        transformer = Flux2Transformer(
            in_channels=8,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=24,
            timestep_guidance_channels=32,
            axes_dims_rope=(4, 4, 4, 4),
        )
        transformer.set_dtype(mx.bfloat16)
        super().__init__(model_config, transformer, cfg=cfg)
        self.prompt = mx.random.normal((1, 4, 24)).astype(mx.bfloat16)
        self.negative = mx.random.normal((1, 4, 24)).astype(mx.bfloat16) if cfg else None

    @staticmethod
    def _predict(transformer: Any) -> Callable[..., mx.array]:
        """The unpatched predict, as mflux's FLUX.2 Klein builds it (without mx.compile)."""

        def predict(**kw: Any) -> mx.array:
            noise: mx.array = transformer(
                hidden_states=kw["latents"],
                encoder_hidden_states=kw["prompt_embeds"],
                timestep=kw["timestep"],
                img_ids=kw["latent_ids"],
                txt_ids=kw["text_ids"],
                guidance=None,
            )
            return noise

        return predict

    def _step_function(self) -> Callable[[int, Any, mx.array], tuple[mx.array, list[mx.array]]]:
        predict = self._predict(self.transformer)

        def step(t: int, config: Any, latents: mx.array) -> tuple[mx.array, list[mx.array]]:
            noise = predict(
                latents=latents,
                latent_ids=_flux2_ids(latents.shape[1]),
                prompt_embeds=self.prompt,
                text_ids=_flux2_ids(self.prompt.shape[1]),
                negative_prompt_embeds=self.negative,
                negative_text_ids=None if self.negative is None else _flux2_ids(self.negative.shape[1]),
                guidance=_CFG_GUIDANCE if self.cfg else 1.0,
                timestep=config.sigmas[t],
            )
            return noise, [noise]

        return step


class _ZImagePipeline(_Pipeline):
    latent_shape = (16, 1, 8, 8)
    changed_latent_shape = (16, 1, 16, 16)  # the patch stream pads to 32 tokens: 8 x 16 would not change it

    def __init__(self, model_config: Any, *, cfg: bool) -> None:
        from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer

        transformer = ZImageTransformer(
            dim=64,
            n_layers=2,
            n_refiner_layers=1,
            n_heads=2,
            cap_feat_dim=32,
            axes_dims=[8, 12, 12],
            axes_lens=[64, 32, 32],
        )
        transformer.set_dtype(mx.bfloat16)
        super().__init__(model_config, transformer, cfg=cfg)
        self.prompt = mx.random.normal((7, 32)).astype(mx.bfloat16)
        self.negative = mx.random.normal((7, 32)).astype(mx.bfloat16) if cfg else None

    @staticmethod
    def _predict(transformer: Any) -> Callable[..., mx.array]:
        """The unpatched predict, as mflux's Z-Image builds it (without mx.compile)."""

        def predict(**kw: Any) -> mx.array:
            noise: mx.array = transformer(
                timestep=kw["timestep"], x=kw["latents"], cap_feats=kw["text_encodings"], sigmas=kw["sigmas"]
            )
            return noise

        return predict

    def _step_function(self) -> Callable[[int, Any, mx.array], tuple[mx.array, list[mx.array]]]:
        predict = self._predict(self.transformer)

        def step(t: int, config: Any, latents: mx.array) -> tuple[mx.array, list[mx.array]]:
            sigma_t = config.sigmas[t].reshape((1,))
            noise = predict(
                latents=latents,
                timestep=mx.ones_like(sigma_t) - sigma_t,
                sigmas=config.sigmas,
                text_encodings=self.prompt,
                negative_encodings=self.negative,
                guidance=_CFG_GUIDANCE if self.cfg else 1.0,
            )
            return noise, [noise]

        return step


class _QwenPipeline(_Pipeline):
    latent_shape = (1, 16, 64)  # 64 x 64 pixels, 4 x 4 latent patches
    changed_latent_shape = (1, 32, 64)

    def __init__(self, model_config: Any, *, cfg: bool) -> None:
        from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer

        transformer = QwenTransformer(
            num_layers=2, attention_head_dim=128, num_attention_heads=1, joint_attention_dim=32
        )
        transformer.set_dtype(mx.bfloat16)
        super().__init__(model_config, transformer, cfg=cfg)
        self.prompt = mx.random.normal((1, 5, 32)).astype(mx.bfloat16)
        self.negative = mx.random.normal((1, 5, 32)).astype(mx.bfloat16)
        self.mask = mx.ones((1, 5))

    def _config(self, request: _Request) -> Any:
        from types import SimpleNamespace

        sigmas = _sigmas(request.num_inference_steps)
        return SimpleNamespace(
            height=64,
            width=64,
            num_inference_steps=request.num_inference_steps,
            init_time_step=request.init_time_step,
            scheduler=SimpleNamespace(sigmas=sigmas, timesteps=sigmas * 1000),
        )

    def _step_function(self) -> Callable[[int, Any, mx.array], tuple[mx.array, list[mx.array]]]:
        def step(t: int, config: Any, latents: mx.array) -> tuple[mx.array, list[mx.array]]:
            # mflux calls the transformer for the positive, then the negative prompt.
            positive = self.transformer(
                t=t,
                config=config,
                hidden_states=latents,
                encoder_hidden_states=self.prompt,
                encoder_hidden_states_mask=self.mask,
            )
            negative = self.transformer(
                t=t,
                config=config,
                hidden_states=latents,
                encoder_hidden_states=self.negative,
                encoder_hidden_states_mask=self.mask,
            )
            return negative + _CFG_GUIDANCE * (positive - negative), [positive, negative]

        return step


_PIPELINES: dict[str, type[_Pipeline]] = {
    "flux1": _Flux1Pipeline,
    "flux2": _Flux2Pipeline,
    "z-image": _ZImagePipeline,
    "qwen": _QwenPipeline,
}
_STRATEGY = {
    "flux1": "proxy-flux1",
    "flux2": "predict-flux2",
    "z-image": "predict-z-image",
    "qwen": "proxy-qwen",
}
# variant id -> (family, aliases, model_name). Qwen-Image names the checkpoint its fit was measured on.
_MODELS: dict[str, tuple[str, list[str], str | None]] = {
    "flux1-dev": ("flux1", ["dev"], "black-forest-labs/FLUX.1-dev"),
    "flux1-schnell": ("flux1", ["schnell"], "black-forest-labs/FLUX.1-schnell"),
    "flux1-krea-dev": ("flux1", ["krea-dev"], "black-forest-labs/FLUX.1-Krea-dev"),
    "flux2-klein-4b": ("flux2", ["flux2-klein-4b"], None),
    "flux2-klein-9b": ("flux2", ["flux2-klein-9b"], None),
    "flux2-klein-base-4b": ("flux2", ["flux2-klein-base-4b"], None),
    "flux2-klein-base-9b": ("flux2", ["flux2-klein-base-9b"], None),
    "z-image-base": ("z-image", ["z-image"], None),
    "qwen-image": ("qwen", ["qwen-image"], "Qwen/Qwen-Image"),
    "qwen-image-2512": (
        "qwen",
        ["qwen-image", "qwen", "qwen-image-2512", "qwen-2512"],
        "Qwen/Qwen-Image-2512",
    ),
}
_CFG_VARIANTS = ("flux2-klein-base-4b", "flux2-klein-base-9b", "z-image-base")
_FAMILY_REPRESENTATIVES = ("flux1-dev", "flux2-klein-base-4b", "z-image-base", "qwen-image")


def _build(variant_id: str, *, cfg: bool) -> _Pipeline:
    from types import SimpleNamespace

    family, aliases, model_name = _MODELS[variant_id]
    mx.random.seed(0)
    model_config = SimpleNamespace(aliases=aliases, model_name=model_name, base_model=None)
    return _PIPELINES[family](model_config, cfg=cfg)


def _scenarios() -> Iterator[tuple[str, str, bool, dict[str, Any], list[_Request]]]:
    """(name, variant id, CFG on, apply_teacache kwargs, one request per generation)."""
    const = {"rel_l1_thresh": 0.25, "coefficients": _CONST_COEFFICIENTS}
    for variant_id in sorted(_MODELS):
        for cfg in (False, True) if variant_id in _CFG_VARIANTS else (False,):
            prefix = f"{variant_id}/cfg" if cfg else variant_id
            yield f"{prefix}/thresh=0", variant_id, cfg, {"rel_l1_thresh": 0.0}, [_Request(seed=1)]
            yield f"{prefix}/default", variant_id, cfg, {}, [_Request(seed=1)]
            yield (
                f"{prefix}/const-coeffs-two-runs",
                variant_id,
                cfg,
                const,
                [_Request(seed=1), _Request(seed=2)],
            )
            # Built-in coefficients (and their calibrated range) at a high threshold; 12 steps leave room for a
            # run of skips as long as the consecutive-skip cap.
            high = [_Request(seed=1, num_inference_steps=12)]
            yield f"{prefix}/builtin-thresh=0.9", variant_id, cfg, {"rel_l1_thresh": 0.9}, high
    for variant_id in _FAMILY_REPRESENTATIVES:
        yield f"{variant_id}/img2img/default", variant_id, False, {}, [_Request(seed=1, init_time_step=4)]
        yield (
            f"{variant_id}/img2img/const-coeffs",
            variant_id,
            False,
            const,
            [_Request(seed=1, init_time_step=4)],
        )
    for variant_id in _FAMILY_REPRESENTATIVES:
        # A non-default window: the first three steps are forced, then the constant cost skips.
        yield (
            f"{variant_id}/skip-first-3/const-coeffs",
            variant_id,
            False,
            {**const, "skip_first_n_steps": 3},
            [_Request(seed=1)],
        )
    for variant_id in _FAMILY_REPRESENTATIVES:
        yield (
            f"{variant_id}/error/window-refusal",
            variant_id,
            False,
            {},
            [_Request(seed=1, num_inference_steps=2)],
        )
        yield (
            f"{variant_id}/error/shape-change",
            variant_id,
            False,
            {},
            [_Request(seed=1, shape_change_at_step=5)],
        )
        yield (
            f"{variant_id}/error/raise-at-step-5-then-clean",
            variant_id,
            False,
            const,
            [_Request(seed=1, raise_at_step=5), _Request(seed=2)],
        )
    for variant_id in ("flux2-klein-base-4b", "z-image-base"):
        yield (
            f"{variant_id}/error/missing-context",
            variant_id,
            False,
            {},
            [_Request(seed=1, skip_before_loop=True)],
        )


def _generation_record(stats: Any) -> dict[str, Any]:
    last = stats.last_generation
    return {
        "last_generation": None
        if last is None
        else {
            "num_steps": last.num_steps,
            "cfg_was_active": last.cfg_was_active,
            "decisions": [
                [d.step_idx, _enc(d.timestep), _enc(d.rel_l1), _enc(d.accumulated_distance), d.decision]
                for d in last.decisions
            ],
        },
        "counters": {
            "generations": stats.generations,
            "computed_count": stats.computed_count,
            "forced_count": stats.forced_count,
            "skipped_count": stats.skipped_count,
            "numerical_miss_count": stats.numerical_miss_count,
        },
    }


def _run(flux: _Pipeline, handle: Any, request: _Request) -> dict[str, Any]:
    flux.trace = []
    latents: mx.array | None = None
    error: list[str] | None = None
    with _recording() as caught:
        try:
            latents = flux.generate_image(request=request)
        except Exception as exc:
            error = [type(exc).__name__, str(exc)]
    return {
        "error": error,
        "steps": flux.trace,
        "final_latents": None if latents is None else _digest(latents),
        "warnings": [_warning_row(w) for w in caught],
        **_generation_record(handle.stats),
    }


def _generation(
    variant_id: str, cfg: bool, kwargs: dict[str, Any], requests: list[_Request]
) -> dict[str, Any]:
    from mlx_teacache import apply_teacache

    flux = _build(variant_id, cfg=cfg)
    original_transformer = flux.transformer
    with _recording() as caught:
        handle = apply_teacache(flux, **kwargs)
    scenario: dict[str, Any] = {
        "strategy": _STRATEGY[_MODELS[variant_id][0]],
        "handle": _handle_record(handle),
        "apply_warnings": [_warning_row(w) for w in caught],
        "runs": [_run(flux, handle, request) for request in requests],
    }
    internal = _internal(handle)
    with _recording() as caught:
        handle.restore()
    scenario["teardown"] = _teardown_record(flux, original_transformer, internal)
    scenario["teardown_warnings"] = [_warning_row(w) for w in caught]
    return scenario


def _generations() -> dict[str, Any]:
    return {
        name: _generation(variant_id, cfg, kwargs, requests)
        for name, variant_id, cfg, kwargs, requests in _scenarios()
    }


# ---------- the record ----------


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _meta(device: str) -> dict[str, Any]:
    import mlx_teacache
    from mlx_teacache._kernel.decision_rule import DECISION_RULE_REVISION
    from mlx_teacache._kernel.gate import MAX_CONSECUTIVE_SKIPS

    return {
        "decision_rule_revision": DECISION_RULE_REVISION,
        "max_consecutive_skips": MAX_CONSECUTIVE_SKIPS,
        "module_file": mlx_teacache.__file__,
        "mlx_version": _version("mlx"),
        "mflux_version": _version("mflux"),
        "device": device,
        "harness_sha256": _harness_sha256(_HARNESS_FILES),
    }


def _harness_sha256(files: tuple[Path, ...] | list[Path]) -> str:
    """One sha256 over each file's name and bytes, in the given order."""
    digest = hashlib.sha256()
    for path in files:
        data = path.read_bytes()
        digest.update(f"{path.name}\0{len(data)}\0".encode())
        digest.update(data)
    return digest.hexdigest()


@contextmanager
def _default_device(device: str) -> Iterator[None]:
    if device not in ("gpu", "cpu"):
        raise ValueError(f"device must be 'gpu' or 'cpu', got {device!r}")
    previous = mx.default_device()
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def collect(*, device: str = "gpu") -> dict[str, Any]:
    """Run every scenario on `device` and return the record (no timestamp; `record` adds one)."""
    with _default_device(device):
        return {
            "meta": _meta(device),
            "threshold_matrix": _threshold_matrix(),
            "warning_attribution": {"qwen-image/uncalibrated-checkpoint": _qwen_uncalibrated_checkpoint()},
            "generations": _generations(),
        }


def count_scenarios(record: dict[str, Any]) -> int:
    matrix = sum(len(cells) for cells in record.get("threshold_matrix", {}).values())
    return matrix + len(record.get("warning_attribution", {})) + len(record.get("generations", {}))


# ---------- compare ----------


def _diff(a: Any, b: Any, path: str, out: list[str]) -> None:
    if path in _NOT_COMPARED:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            sub = f"{path}/{key}" if path else str(key)
            if key not in a or key not in b:
                if sub not in _NOT_COMPARED:
                    out.append(sub)
                continue
            _diff(a[key], b[key], sub, out)
        return
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _diff(x, y, f"{path}[{i}]", out)
        return
    if type(a) is not type(b) or a != b:
        out.append(path)


def compare(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Key paths where the two records differ; empty means identical."""
    out: list[str] = []
    _diff(a, b, "", out)
    return out


# ---------- command line ----------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests._equivalence_oracle")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--out", required=True, type=Path)
    rec.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a", type=Path)
    cmp_.add_argument("b", type=Path)
    args = parser.parse_args(argv)

    if args.command == "record":
        record = collect(device=args.device)
        record["meta"]["recorded_at"] = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"recorded {count_scenarios(record)} scenarios to {args.out}")
        return 0

    a = json.loads(args.a.read_text())
    b = json.loads(args.b.read_text())
    differences = compare(a, b)
    if not differences:
        print(f"identical ({count_scenarios(a)} scenarios)")
        return 0
    for path in differences:
        print(path)
    print(f"{len(differences)} difference(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
