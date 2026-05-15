"""Calibrate Flux2 Klein 4b polynomial coefficients.

For each calibration prompt:
- Patch `flux._predict` with a capturing wrapper that runs the full vanilla
  forward (no skipping) and records `mod_in` and `body_out_concat` per step.
- Run `flux.generate_image(...)` at the target inference budget.
- Compute, for every consecutive step pair (t-1, t), the relative-L1 deltas
    x_t = ||mod_in_t       - mod_in_{t-1}||_1     / ||mod_in_{t-1}||_1
    y_t = ||body_out_t     - body_out_{t-1}||_1   / ||body_out_{t-1}||_1
- Aggregate (x_t, y_t) pairs across all prompts and fit a degree-4 polynomial
  with `numpy.polyfit` (returns coefficients high-to-low, matching the
  `poly_eval` convention used at runtime in `mlx_teacache.gate`).

Output: a JSON report at `scripts/_calibration_flux2_klein.json` with the
fitted coefficients, R^2, sample count, and a summary the user can inspect
before committing the new constants into `src/mlx_teacache/coefficients.py`.

Run as: `uv run python scripts/calibrate_flux2_klein.py`.
"""

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

NUM_INFERENCE_STEPS = 8
HEIGHT = 512
WIDTH = 512
GUIDANCE = 1.0  # no CFG so the captured path matches the gated path at runtime
SEED = 42


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


def _capture_one_prompt(flux: Any, prompt: str) -> list[dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    had_instance_attr = "_predict" in vars(flux)
    original = flux._predict if had_instance_attr else None
    flux._predict = _build_capturing_predict_factory(captures)
    try:
        flux.generate_image(
            prompt=prompt,
            seed=SEED,
            num_inference_steps=NUM_INFERENCE_STEPS,
            height=HEIGHT,
            width=WIDTH,
            guidance=GUIDANCE,
        )
    finally:
        if had_instance_attr:
            flux._predict = original
        else:
            del flux._predict
    return captures


def main() -> None:
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    print("Loading Flux2Klein 4b...")
    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())
    flux.freeze()

    xs: list[float] = []
    ys: list[float] = []
    per_prompt: list[dict[str, Any]] = []
    t_start = time.time()
    for i, prompt in enumerate(CALIBRATION_PROMPTS, 1):
        print(f"[{i}/{len(CALIBRATION_PROMPTS)}] {prompt!r}")
        capture = _capture_one_prompt(flux, prompt)
        assert len(capture) == NUM_INFERENCE_STEPS, (
            f"expected {NUM_INFERENCE_STEPS} captures, got {len(capture)}"
        )
        prompt_pairs: list[tuple[float, float]] = []
        for t in range(1, len(capture)):
            x = _rel_l1(capture[t]["mod_in"], capture[t - 1]["mod_in"])
            y = _rel_l1(capture[t]["body_out"], capture[t - 1]["body_out"])
            xs.append(x)
            ys.append(y)
            prompt_pairs.append((x, y))
        per_prompt.append({"prompt": prompt, "pairs": prompt_pairs})

    elapsed = time.time() - t_start
    print(f"\nCaptured {len(xs)} (x, y) pairs in {elapsed:.1f}s")

    # Degree-4 polynomial. numpy.polyfit returns high-to-low which matches
    # the (c4, c3, c2, c1, c0) convention used by gate.poly_eval.
    coeffs = np.polyfit(xs, ys, 4)
    p = np.poly1d(coeffs)
    y_pred = p(xs)
    ss_res = float(np.sum((np.array(ys) - y_pred) ** 2))
    ss_tot = float(np.sum((np.array(ys) - np.mean(ys)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"R^2 = {r2:.6f}")
    print("Coefficients (c4, c3, c2, c1, c0):")
    for c in coeffs:
        print(f"  {c:.10g}")

    report = {
        "variant": "flux2-klein-4b",
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "height": HEIGHT,
        "width": WIDTH,
        "guidance": GUIDANCE,
        "seed": SEED,
        "num_prompts": len(CALIBRATION_PROMPTS),
        "num_pairs": len(xs),
        "elapsed_seconds": elapsed,
        "coefficients_c4_to_c0": [float(c) for c in coeffs],
        "fit_r_squared": r2,
        "calibration_prompts": list(CALIBRATION_PROMPTS),
        "x_min": float(min(xs)),
        "x_max": float(max(xs)),
        "y_min": float(min(ys)),
        "y_max": float(max(ys)),
    }
    out_path = Path(__file__).parent / "_calibration_flux2_klein.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
