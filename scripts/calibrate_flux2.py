"""Calibrate FLUX.2 polynomial coefficients for one variant.

Run as: `uv run python scripts/calibrate_flux2.py --variant klein-4b`
        `uv run python scripts/calibrate_flux2.py --variant klein-9b`
        `uv run python scripts/calibrate_flux2.py --variant klein-base-4b --fit-mode origin`

klein-base-9b is declared but raises NotImplementedError (wired in v0.5.0).

For each calibration prompt:
- Patch `flux._predict` with a capturing wrapper that runs the full vanilla
  forward (no skipping) and records `mod_in` and `body_out_concat` per step.
- Run `flux.generate_image(...)` at the target inference budget.
- Compute, for every consecutive step pair (t-1, t), the relative-L1 deltas:
    x_t = ||mod_in_t   - mod_in_{t-1}||_1   / ||mod_in_{t-1}||_1
    y_t = ||body_out_t - body_out_{t-1}||_1 / ||body_out_{t-1}||_1
- Aggregate (x_t, y_t) pairs across all prompts and fit a degree-4 polynomial
  with `numpy.polyfit` (returns coefficients high-to-low, matching the
  `poly_eval` convention used at runtime in `mlx_teacache.gate`).

Output: a JSON report at `scripts/_calibration_flux2_<variant>.json`."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_teacache.integrations.mflux.forward import (
    _flux2_extract_mod_input,
    _flux2_run_body,
)

# 10 prompts mixing photographic, illustrative, abstract, and text content —
# covers the diversity TeaCache's polynomial fit needs.
CALIBRATION_PROMPTS = (
    "a red apple on a wooden table",
    "mountain landscape at sunset",
    "portrait of a woman",
    "abstract pattern with circles",
    "text saying HELLO",
    "a futuristic cityscape at night",
    "a watercolor painting of a cat",
    "a steampunk airship in the clouds",
    "macro photograph of a butterfly wing",
    "neon signs in a rainy street",
)

HEIGHT = 512
WIDTH = 512
GUIDANCE = 1.0
SEED = 42


def _model_config_klein_4b() -> Any:
    from mflux.models.common.config.model_config import ModelConfig

    return ModelConfig.flux2_klein_4b()


def _model_config_klein_9b() -> Any:
    from mflux.models.common.config.model_config import ModelConfig

    return ModelConfig.flux2_klein_9b()


def _model_config_klein_base_4b() -> Any:
    from mflux.models.common.config.model_config import ModelConfig

    return ModelConfig.flux2_klein_base_4b()


def _not_wired(release: str) -> Any:
    def _raise() -> Any:
        raise NotImplementedError(
            f"This variant will be wired in {release}; currently out of scope. "
            f"Use --variant klein-4b or --variant klein-9b."
        )

    return _raise


_VARIANTS: dict[str, dict[str, Any]] = {
    "klein-4b": {
        "variant_id": "flux2-klein-4b",
        "model_config_factory": _model_config_klein_4b,
        "num_inference_steps": 8,
        "output_json": "_calibration_flux2_klein_4b.json",
    },
    "klein-9b": {
        "variant_id": "flux2-klein-9b",
        "model_config_factory": _model_config_klein_9b,
        "num_inference_steps": 8,
        "output_json": "_calibration_flux2_klein_9b.json",
    },
    "klein-base-4b": {
        "variant_id": "flux2-klein-base-4b",
        "model_config_factory": _model_config_klein_base_4b,
        "num_inference_steps": 25,
        "output_json": "_calibration_flux2_klein_base_4b.json",
    },
    "klein-base-9b": {
        "variant_id": "flux2-klein-base-9b",
        "model_config_factory": _not_wired("v0.5.0"),
        "num_inference_steps": None,
        "output_json": None,
    },
}


def _rel_l1(curr: mx.array, prev: mx.array) -> float:
    """Relative-L1 distance, matching mflux's runtime gate signal."""
    curr_f = curr.astype(mx.float32)
    prev_f = prev.astype(mx.float32)
    num = mx.sum(mx.abs(curr_f - prev_f))
    den = mx.sum(mx.abs(prev_f)) + 1e-12
    return float(num / den)


def _build_capturing_predict_factory(captures: list[dict[str, Any]]) -> Any:
    """Return a function assignable to `flux._predict`. mflux calls it as
    `predict = self._predict(self.transformer)` once per generation, then
    invokes the returned closure per step. The closure mirrors
    `flux2_forward_with_gate`'s slow path math (no gating, no caching) and
    appends per-step `(mod_in, body_out_concat)` arrays to `captures`."""
    from mflux.models.common.config.model_config import ModelConfig

    def factory(transformer: Any) -> Any:
        inner = transformer
        return _make_capturing_closure(inner, captures, ModelConfig)

    return factory


def _make_capturing_closure(inner: Any, captures: list[dict[str, Any]], ModelConfig: Any) -> Any:
    def predict(
        latents: mx.array,
        latent_ids: mx.array,
        prompt_embeds: mx.array,
        text_ids: mx.array,
        negative_prompt_embeds: mx.array | None,
        negative_text_ids: mx.array | None,
        guidance: float,
        timestep: mx.array,
    ) -> mx.array:
        del guidance, negative_prompt_embeds, negative_text_ids
        # Mirror Flux2Transformer.__call__ exactly so the capture is the same
        # tensor flux2_forward_with_gate observes at runtime.
        ts = timestep
        if not isinstance(ts, mx.array):
            ts = mx.array(ts, dtype=latents.dtype)
        if ts.ndim == 0:
            ts = mx.full((latents.shape[0],), ts, dtype=latents.dtype)
        ts = ts.astype(latents.dtype)
        ts_scale = mx.where(mx.max(ts) <= 1.0, 1000.0, 1.0).astype(latents.dtype)
        ts = ts * ts_scale
        temb = inner.time_guidance_embed(ts, None)
        temb = temb.astype(ModelConfig.precision)

        body_in = inner.x_embedder(latents)
        encoder_hs = inner.context_embedder(prompt_embeds)
        img_ids = latent_ids[0] if latent_ids.ndim == 3 else latent_ids
        txt_ids = text_ids[0] if text_ids.ndim == 3 else text_ids
        image_rotary_emb = inner.pos_embed(img_ids)
        text_rotary_emb = inner.pos_embed(txt_ids)
        concat_rotary_emb = (
            mx.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
            mx.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
        )
        temb_mod_params_img = inner.double_stream_modulation_img(temb)
        temb_mod_params_txt = inner.double_stream_modulation_txt(temb)

        mod_in = _flux2_extract_mod_input(inner, body_in, temb_mod_params_img)
        body_out_concat = _flux2_run_body(
            inner,
            body_in,
            encoder_hs,
            temb,
            temb_mod_params_img,
            temb_mod_params_txt,
            concat_rotary_emb,
        )
        mx.eval(mod_in, body_out_concat)
        captures.append({"mod_in": mod_in, "body_out": body_out_concat})

        out = body_out_concat[:, encoder_hs.shape[1] :, ...]
        out = inner.norm_out(out, temb)
        out = inner.proj_out(out)
        return out

    return predict


def _make_cfg_capturing_closure(inner: Any, captures: list[dict[str, Any]], ModelConfig: Any) -> Any:
    """CFG-aware capture (v0.4.1). Runs BOTH branches per step, returns
    CFG-combined noise to the scheduler so the next latent follows the
    real g>1 trajectory, captures the shared mod_in plus per-branch
    body_out_concat."""

    def predict(
        latents: mx.array,
        latent_ids: mx.array,
        prompt_embeds: mx.array,
        text_ids: mx.array,
        negative_prompt_embeds: mx.array | None,
        negative_text_ids: mx.array | None,
        guidance: float,
        timestep: mx.array,
    ) -> mx.array:
        assert negative_prompt_embeds is not None, "CFG capture requires negative embeds"
        assert negative_text_ids is not None

        ts = timestep
        if not isinstance(ts, mx.array):
            ts = mx.array(ts, dtype=latents.dtype)
        if ts.ndim == 0:
            ts = mx.full((latents.shape[0],), ts, dtype=latents.dtype)
        ts = ts.astype(latents.dtype)
        ts_scale = mx.where(mx.max(ts) <= 1.0, 1000.0, 1.0).astype(latents.dtype)
        ts = ts * ts_scale
        temb = inner.time_guidance_embed(ts, None)
        temb = temb.astype(ModelConfig.precision)

        body_in = inner.x_embedder(latents)
        img_ids = latent_ids[0] if latent_ids.ndim == 3 else latent_ids
        image_rotary_emb = inner.pos_embed(img_ids)
        temb_mod_params_img = inner.double_stream_modulation_img(temb)
        temb_mod_params_txt = inner.double_stream_modulation_txt(temb)

        # Shared gate signal.
        mod_in = _flux2_extract_mod_input(inner, body_in, temb_mod_params_img)

        # Positive branch.
        enc_pos = inner.context_embedder(prompt_embeds)
        txt_ids_pos = text_ids[0] if text_ids.ndim == 3 else text_ids
        txt_rot_pos = inner.pos_embed(txt_ids_pos)
        concat_rot_pos = (
            mx.concatenate([txt_rot_pos[0], image_rotary_emb[0]], axis=0),
            mx.concatenate([txt_rot_pos[1], image_rotary_emb[1]], axis=0),
        )
        body_out_pos = _flux2_run_body(
            inner, body_in, enc_pos, temb, temb_mod_params_img, temb_mod_params_txt, concat_rot_pos
        )

        # Negative branch.
        enc_neg = inner.context_embedder(negative_prompt_embeds)
        txt_ids_neg = negative_text_ids[0] if negative_text_ids.ndim == 3 else negative_text_ids
        txt_rot_neg = inner.pos_embed(txt_ids_neg)
        concat_rot_neg = (
            mx.concatenate([txt_rot_neg[0], image_rotary_emb[0]], axis=0),
            mx.concatenate([txt_rot_neg[1], image_rotary_emb[1]], axis=0),
        )
        body_out_neg = _flux2_run_body(
            inner, body_in, enc_neg, temb, temb_mod_params_img, temb_mod_params_txt, concat_rot_neg
        )

        mx.eval(mod_in, body_out_pos, body_out_neg)
        captures.append({"mod_in": mod_in, "body_out_pos": body_out_pos, "body_out_neg": body_out_neg})

        # Tail + CFG combine for the scheduler.
        noise_pos = body_out_pos[:, enc_pos.shape[1] :, ...]
        noise_pos = inner.norm_out(noise_pos, temb)
        noise_pos = inner.proj_out(noise_pos)
        noise_neg = body_out_neg[:, enc_neg.shape[1] :, ...]
        noise_neg = inner.norm_out(noise_neg, temb)
        noise_neg = inner.proj_out(noise_neg)
        return noise_neg + guidance * (noise_pos - noise_neg)

    return predict


def _build_cfg_capturing_predict_factory(captures: list[dict[str, Any]]) -> Any:
    from mflux.models.common.config.model_config import ModelConfig

    def factory(transformer: Any) -> Any:
        return _make_cfg_capturing_closure(transformer, captures, ModelConfig)

    return factory


def _capture_one_prompt(
    flux: Any, prompt: str, *, num_inference_steps: int, guidance: float
) -> list[dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    had_instance_attr = "_predict" in vars(flux)
    original = flux._predict if had_instance_attr else None
    if guidance > 1.0:
        flux._predict = _build_cfg_capturing_predict_factory(captures)
    else:
        flux._predict = _build_capturing_predict_factory(captures)
    try:
        flux.generate_image(
            prompt=prompt,
            seed=SEED,
            num_inference_steps=num_inference_steps,
            height=HEIGHT,
            width=WIDTH,
            guidance=guidance,
        )
    finally:
        if had_instance_attr:
            flux._predict = original
        else:
            del flux._predict
    return captures


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        required=True,
        choices=sorted(_VARIANTS.keys()),
        help="Which variant to calibrate. v0.3.0 wires klein-4b and klein-9b.",
    )
    parser.add_argument(
        "--fit-mode",
        default="free",
        choices=["free", "origin"],
        help=(
            "Polynomial fit mode. 'free' = standard numpy.polyfit (c0 unconstrained); "
            "'origin' = forces the polynomial through (0, 0) so the predicted output "
            "rel_l1 is 0 when the input rel_l1 is 0. Use 'origin' when the free fit "
            "gives a non-zero intercept that prevents the gate from ever signaling 'skip'."
        ),
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=1.0,
        help="Guidance value for calibration (1.0 = no CFG / positive only; >1 enables CFG capture path).",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override the variant's hardcoded step count. Required when calibrating for a recipe that differs from the variant's default (e.g. base-4b CFG @ 50 steps, not the default 25).",
    )
    parser.add_argument(
        "--fit-branch-policy",
        default="worst",
        choices=["worst", "average", "positive", "negative"],
        help="Under CFG calibration, which per-step y target to fit: worst-branch (default), average, positive only, or negative only.",
    )
    args = parser.parse_args()
    cfg = _VARIANTS[args.variant]
    variant_id: str = cfg["variant_id"]
    num_inference_steps: int = (
        args.num_inference_steps if args.num_inference_steps is not None else cfg["num_inference_steps"]
    )
    output_json: str = cfg["output_json"]
    fit_mode: str = args.fit_mode

    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    flux = Flux2Klein(quantize=4, model_config=cfg["model_config_factory"]())
    flux.freeze()

    xs: list[float] = []
    ys: list[float] = []
    ys_pos: list[float] = []
    ys_neg: list[float] = []
    per_prompt: list[dict[str, Any]] = []
    t_start = time.time()
    for i, prompt in enumerate(CALIBRATION_PROMPTS, 1):
        print(f"[{i}/{len(CALIBRATION_PROMPTS)}] {prompt!r}")
        capture = _capture_one_prompt(
            flux, prompt, num_inference_steps=num_inference_steps, guidance=args.guidance
        )
        assert len(capture) == num_inference_steps, (
            f"expected {num_inference_steps} captures, got {len(capture)}"
        )
        prompt_pairs: list[tuple[float, float]] = []
        for t in range(1, len(capture)):
            x = _rel_l1(capture[t]["mod_in"], capture[t - 1]["mod_in"])
            if args.guidance > 1.0:
                y_pos = _rel_l1(capture[t]["body_out_pos"], capture[t - 1]["body_out_pos"])
                y_neg = _rel_l1(capture[t]["body_out_neg"], capture[t - 1]["body_out_neg"])
                if args.fit_branch_policy == "worst":
                    y = max(y_pos, y_neg)
                elif args.fit_branch_policy == "average":
                    y = 0.5 * (y_pos + y_neg)
                elif args.fit_branch_policy == "positive":
                    y = y_pos
                else:  # negative
                    y = y_neg
                ys_pos.append(y_pos)
                ys_neg.append(y_neg)
            else:
                y = _rel_l1(capture[t]["body_out"], capture[t - 1]["body_out"])
            xs.append(x)
            ys.append(y)
            prompt_pairs.append((x, y))
        per_prompt.append({"prompt": prompt, "pairs": prompt_pairs})

    elapsed = time.time() - t_start
    print(f"\nCaptured {len(xs)} (x, y) pairs in {elapsed:.1f}s")

    # Degree-4 polynomial. numpy.polyfit returns high-to-low which matches
    # the (c4, c3, c2, c1, c0) convention used by gate.poly_eval.
    xs_np = np.array(xs)
    ys_np = np.array(ys)
    if fit_mode == "free":
        coeffs = np.polyfit(xs_np, ys_np, 4)
    elif fit_mode == "origin":
        # Constrained least squares: fit y = a4*x^4 + a3*x^3 + a2*x^2 + a1*x
        # (no intercept), then pad c0 = 0 so the returned shape is still (5,).
        X = np.column_stack([xs_np**4, xs_np**3, xs_np**2, xs_np])
        a, *_ = np.linalg.lstsq(X, ys_np, rcond=None)
        coeffs = np.array([a[0], a[1], a[2], a[3], 0.0])
    else:
        raise ValueError(f"unknown fit_mode={fit_mode!r}")
    p = np.poly1d(coeffs)
    y_pred = p(xs_np)
    ss_res = float(np.sum((ys_np - y_pred) ** 2))
    ss_tot = float(np.sum((ys_np - np.mean(ys_np)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"fit_mode = {fit_mode}")
    print(f"R^2 = {r2:.6f}")
    print("Coefficients (c4, c3, c2, c1, c0):")
    for c in coeffs:
        print(f"  {c:.10g}")

    report: dict[str, Any] = {
        "variant": variant_id,
        "num_inference_steps": num_inference_steps,
        "height": HEIGHT,
        "width": WIDTH,
        "guidance": args.guidance,
        "seed": SEED,
        "num_prompts": len(CALIBRATION_PROMPTS),
        "num_pairs": len(xs),
        "elapsed_seconds": elapsed,
        "fit_mode": fit_mode,
        "coefficients_c4_to_c0": [float(c) for c in coeffs],
        "fit_r_squared": r2,
        "calibration_prompts": list(CALIBRATION_PROMPTS),
        "x_values": [float(x) for x in xs],  # raw x array for offline refit
        "y_values": [float(y) for y in ys],  # raw y array for offline refit
        "x_min": float(min(xs)),
        "x_max": float(max(xs)),
        "y_min": float(min(ys)),
        "y_max": float(max(ys)),
    }
    if args.guidance > 1.0:
        report["fit_branch_policy"] = args.fit_branch_policy
        report["y_values_pos"] = [float(y) for y in ys_pos]
        report["y_values_neg"] = [float(y) for y in ys_neg]
    out_path = Path(__file__).parent / output_json
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
