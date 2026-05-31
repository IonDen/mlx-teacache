"""Calibrate Z-Image base polynomial coefficients — captures BOTH candidate gate signals.

Z-Image is a single-stream DiT whose adaLN modulation is timestep-only
(content-independent), so the gate signal must be latent-dependent and tapped
inside the transformer. This script captures two candidates per step and fits a
degree-4 polynomial for each, so the winner can be chosen later by the held-out
skip-vs-SSIM knee (the sweep, run after the variant integration exists):

  Signal A — noise-refiner output rel-L1 (image-only, caption-INDEPENDENT ⇒ one
             shared CFG gate decision is exact). Computed in the prelude every
             step regardless, so zero extra runtime cost.
  Signal B — first-main-layer output residual rel-L1 (caption-dependent; costs
             one of the 30 main layers on a skip step).

Target predicted: per-step rel-L1 of `main_out` (the 30 main layers' output).
The runtime cache stores the residual `main_out - unified_in`; the script also
records the residual's rel-L1 so we can compare which target is better-
conditioned on this single-stream model.

The capturing closure RE-WALKS `ZImageTransformer.__call__` (mflux 0.17.5,
transformer.py:57-139) with taps, running the full vanilla forward (no skips),
and returns the real CFG-combined noise so the scheduler trajectory is correct.
A first-step self-check asserts the re-walk's noise matches the transformer's
own forward (cosine >= 0.999) — a faithful-port guard.

Run (AFTER the model is downloaded; HEAVY — one full vanilla forward per prompt,
~221s each at the pinned recipe ⇒ ~37 min for 10 prompts):

    uv run python scripts/calibrate_z_image.py --fit-mode origin

Output: scripts/_calibration_z_image.json (both signals' fits + R^2 + curve
range + held-out split + raw arrays for offline refit).
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_teacache._kernel.gate import mean_abs_rel_l1  # reuse the RUNTIME signal fn

# --- Pinned recipe (findings 2026-05-31). Calibrate + sweep + bench all share it. ---
SEED = 42
HEIGHT = WIDTH = 512
NUM_INFERENCE_STEPS = 50
GUIDANCE = 4.0  # CFG path (two transformer passes)
QUANTIZE = 8

# 10 prompts, mirroring calibrate_flux2.py's diversity. Held-out split below.
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
# Last 3 prompts are held out from the FIT; reported separately so signal
# selection (later, via the sweep) does not select on the fitted prompts.
N_HELDOUT = 3

OUTPUT_JSON = "_calibration_z_image.json"


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_forward_z_image.py — no weights).
# ---------------------------------------------------------------------------


def fit_signal(xs: list[float], ys: list[float], *, fit_mode: str) -> dict[str, Any]:
    """Fit a degree-4 polynomial mapping signal rel-L1 (x) -> body rel-L1 (y).

    fit_mode 'free' = numpy.polyfit (c0 unconstrained); 'origin' = forced
    through (0,0) so predicted output rel-L1 is 0 when input rel-L1 is 0
    (matches calibrate_flux2.py + the FLUX.2-family convention). Returns
    coefficients high-to-low (c4..c0), R^2, and the curve range. Pure / no I/O.
    """
    xs_np = np.asarray(xs, dtype=np.float64)
    ys_np = np.asarray(ys, dtype=np.float64)
    if fit_mode == "free":
        coeffs = np.polyfit(xs_np, ys_np, 4)
    elif fit_mode == "origin":
        X = np.column_stack([xs_np**4, xs_np**3, xs_np**2, xs_np])
        a, *_ = np.linalg.lstsq(X, ys_np, rcond=None)
        coeffs = np.array([a[0], a[1], a[2], a[3], 0.0])
    else:
        raise ValueError(f"unknown fit_mode={fit_mode!r}")
    y_pred = np.poly1d(coeffs)(xs_np)
    ss_res = float(np.sum((ys_np - y_pred) ** 2))
    ss_tot = float(np.sum((ys_np - np.mean(ys_np)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "coefficients_c4_to_c0": [float(c) for c in coeffs],
        "fit_r_squared": r2,
        "x_min": float(np.min(xs_np)),
        "x_max": float(np.max(xs_np)),
        "y_min": float(np.min(ys_np)),
        "y_max": float(np.max(ys_np)),
        "n_pairs": int(xs_np.size),
    }


def _rel_l1(curr: mx.array, prev: mx.array) -> float:
    """Consecutive-step relative-L1 using the SAME function the runtime gate uses."""
    return float(mean_abs_rel_l1(curr, prev))


# ---------------------------------------------------------------------------
# Capturing forward re-walk of ZImageTransformer.__call__ (transformer.py:57-139).
# ---------------------------------------------------------------------------


def _zimage_capture_forward(
    transformer: Any, latents: mx.array, t_emb: mx.array, cap_feats: mx.array
) -> dict[str, Any]:
    """Re-walk the Z-Image transformer for ONE branch, tapping the gate signals.

    Returns dict(signal_A, signal_B, unified_in, main_out, noise). `t_emb` is
    passed in (timestep-only, shared across branches/steps for a given timestep).
    Mirrors transformer.py:77-139 verbatim; only adds taps. `noise` is the final
    negated output — must equal `transformer(...)` for the same inputs.
    """
    ZImageTransformer = type(transformer)
    key = f"{transformer.patch_size}-{transformer.f_patch_size}"

    # Patchify (transformer.py:78-83)
    x_emb, cap_emb, x_size, x_pos_ids, cap_pos_ids, x_pad_mask, cap_pad_mask = ZImageTransformer._patchify(
        image=latents,
        cap_feats=cap_feats,
        patch_size=transformer.patch_size,
        f_patch_size=transformer.f_patch_size,
    )
    # Image embedding (85-90)
    x_emb = transformer.all_x_embedder[key](x_emb)
    x_emb = mx.where(x_pad_mask[:, None], transformer.x_pad_token, x_emb)
    x_freqs_cis = transformer.rope_embedder(x_pos_ids)
    x_attn_mask = mx.ones((1, x_emb.shape[0]), dtype=mx.bool_)
    x_emb = mx.expand_dims(x_emb, axis=0)
    # Noise refiner (92-99)  -> Signal A tap (image-only, caption-independent)
    for layer in transformer.noise_refiner:
        x_emb = layer(x=x_emb, attn_mask=x_attn_mask, freqs_cis=x_freqs_cis, t_emb=t_emb)
    signal_A = x_emb
    # Caption embedding + context refiner (101-114)
    cap_emb = transformer.cap_embedder[1](transformer.cap_embedder[0](cap_emb))
    cap_emb = mx.where(cap_pad_mask[:, None], transformer.cap_pad_token, cap_emb)
    cap_freqs_cis = transformer.rope_embedder(cap_pos_ids)
    cap_attn_mask = mx.ones((1, cap_emb.shape[0]), dtype=mx.bool_)
    cap_emb = mx.expand_dims(cap_emb, axis=0)
    for layer in transformer.context_refiner:
        cap_emb = layer(x=cap_emb, attn_mask=cap_attn_mask, freqs_cis=cap_freqs_cis)
    # Unify + main layers (116-128)  -> Signal B tap (first main layer residual)
    x_len = x_emb.shape[1]
    unified_in = mx.concatenate([x_emb, cap_emb], axis=1)
    unified_freqs_cis = mx.concatenate([x_freqs_cis, cap_freqs_cis], axis=0)
    unified_attn_mask = mx.ones((1, unified_in.shape[1]), dtype=mx.bool_)
    unified = unified_in
    signal_B = None
    for i, layer in enumerate(transformer.layers):
        unified = layer(x=unified, attn_mask=unified_attn_mask, freqs_cis=unified_freqs_cis, t_emb=t_emb)
        if i == 0:
            signal_B = unified - unified_in  # first-main-layer residual
    main_out = unified
    # Final layer + unpatchify + negation (130-139)
    final = transformer.all_final_layer[key](main_out, t_emb)
    output = ZImageTransformer._unpatchify(
        x=final[0, :x_len],
        size=x_size,
        patch_size=transformer.patch_size,
        f_patch_size=transformer.f_patch_size,
        out_channels=transformer.out_channels,
    )
    return {
        "signal_A": signal_A,
        "signal_B": signal_B,
        "unified_in": unified_in,
        "main_out": main_out,
        "noise": -output,
    }


def _t_emb(transformer: Any, timestep: Any, sigmas: mx.array) -> mx.array:
    """Replicate the timestep -> t_emb path (transformer.py:66-75)."""
    if not isinstance(timestep, mx.array):
        if isinstance(timestep, int):
            sigma_t = sigmas[timestep].reshape((1,))
            timestep = mx.ones_like(sigma_t) - sigma_t
        else:
            timestep = mx.array(timestep, dtype=mx.float32)
    if timestep.ndim == 0:
        timestep = timestep.reshape((1,))
    return transformer.t_embedder(timestep.astype(mx.float32) * transformer.t_scale)


def _make_capturing_factory(captures: list[dict[str, Any]], self_check: dict[str, bool]) -> Any:
    def factory(transformer: Any) -> Any:
        def predict(latents, timestep, sigmas, text_encodings, negative_encodings, guidance):  # noqa: ANN001
            t_emb = _t_emb(transformer, timestep, sigmas)
            pos = _zimage_capture_forward(transformer, latents, t_emb, text_encodings)
            rec: dict[str, Any] = {
                "signal_A": pos["signal_A"],
                "signal_B_pos": pos["signal_B"],
                "main_out_pos": pos["main_out"],
                "unified_in_pos": pos["unified_in"],
            }
            noise = pos["noise"]
            if negative_encodings is not None:
                neg = _zimage_capture_forward(transformer, latents, t_emb, negative_encodings)
                rec |= {
                    "signal_B_neg": neg["signal_B"],
                    "main_out_neg": neg["main_out"],
                    "unified_in_neg": neg["unified_in"],
                }
                noise = pos["noise"] + guidance * (pos["noise"] - neg["noise"])  # z_image.py:209
            # Faithful-port self-check on the first captured step.
            if not self_check["done"]:
                ref = transformer(timestep=timestep, x=latents, cap_feats=text_encodings, sigmas=sigmas)
                cos = float(mx.sum(ref * pos["noise"]) / (mx.linalg.norm(ref) * mx.linalg.norm(pos["noise"])))
                assert cos >= 0.999, f"re-walk diverges from transformer forward: cos={cos:.6f} (port bug)"
                self_check["done"] = True
            mx.eval(*[v for v in rec.values() if isinstance(v, mx.array)])
            captures.append(rec)
            return noise

        return predict

    return factory


def _capture_one_prompt(flux: Any, prompt: str) -> list[dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    self_check = {"done": False}
    had = "_predict" in vars(flux)
    original = flux._predict if had else None
    flux._predict = _make_capturing_factory(captures, self_check)
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
        if had:
            flux._predict = original
        else:
            del flux._predict
    return captures


def _pairs_for(
    captures: list[dict[str, Any]], signal_key: str, *, branch: str
) -> tuple[list[float], list[float]]:
    """Build (x=signal rel-L1, y=worst-branch main_out rel-L1) pairs across steps."""
    xs: list[float] = []
    ys: list[float] = []
    for t in range(1, len(captures)):
        if signal_key == "A":
            x = _rel_l1(captures[t]["signal_A"], captures[t - 1]["signal_A"])
        else:  # B (caption-dependent) — use the requested branch
            x = _rel_l1(captures[t][f"signal_B_{branch}"], captures[t - 1][f"signal_B_{branch}"])
        y_pos = _rel_l1(captures[t]["main_out_pos"], captures[t - 1]["main_out_pos"])
        y_neg = _rel_l1(
            captures[t].get("main_out_neg", captures[t]["main_out_pos"]),
            captures[t - 1].get("main_out_neg", captures[t - 1]["main_out_pos"]),
        )
        xs.append(x)
        ys.append(max(y_pos, y_neg))  # worst-branch target (matches flux2 'worst' policy)
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-mode", default="origin", choices=["free", "origin"])
    args = parser.parse_args()

    # Memory guardrail — before any model load (32 GB M1 Max). Set here, not at
    # import, so the module is importable for unit tests without mutating MLX state.
    mx.set_wired_limit(int(20 * 1024**3))
    mx.set_memory_limit(int(22 * 1024**3))

    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.z_image.variants.z_image import ZImage

    print(f"Loading Z-Image base (quantize={QUANTIZE})...", flush=True)
    flux = ZImage(quantize=QUANTIZE, model_config=ModelConfig.z_image())
    flux.freeze()

    all_caps: list[list[dict[str, Any]]] = []
    t0 = time.time()
    for i, prompt in enumerate(CALIBRATION_PROMPTS, 1):
        print(f"[{i}/{len(CALIBRATION_PROMPTS)}] {prompt!r}", flush=True)
        caps = _capture_one_prompt(flux, prompt)
        assert len(caps) == NUM_INFERENCE_STEPS, f"expected {NUM_INFERENCE_STEPS} captures, got {len(caps)}"
        all_caps.append(caps)
    elapsed = time.time() - t0

    n_fit = len(CALIBRATION_PROMPTS) - N_HELDOUT
    report: dict[str, Any] = {
        "variant": "z-image-base",
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance": GUIDANCE,
        "height": HEIGHT,
        "width": WIDTH,
        "seed": SEED,
        "quantize": QUANTIZE,
        "num_prompts": len(CALIBRATION_PROMPTS),
        "n_fit_prompts": n_fit,
        "n_heldout_prompts": N_HELDOUT,
        "elapsed_seconds": elapsed,
        "fit_mode": args.fit_mode,
        "signals": {},
        "calibration_prompts": list(CALIBRATION_PROMPTS),
    }
    for sig in ("A", "B"):
        branch = "pos"  # signal B branch for the fit; A is branch-independent
        fit_x, fit_y, held_x, held_y = [], [], [], []
        for pi, caps in enumerate(all_caps):
            xs, ys = _pairs_for(caps, sig, branch=branch)
            if pi < n_fit:
                fit_x += xs
                fit_y += ys
            else:
                held_x += xs
                held_y += ys
        fit = fit_signal(fit_x, fit_y, fit_mode=args.fit_mode)
        # Held-out R^2 against the fitted polynomial.
        p = np.poly1d(fit["coefficients_c4_to_c0"])
        hy = np.asarray(held_y)
        hp = p(np.asarray(held_x))
        ss_res = float(np.sum((hy - hp) ** 2))
        ss_tot = float(np.sum((hy - np.mean(hy)) ** 2))
        fit["heldout_r_squared"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        fit["x_values"] = [float(x) for x in fit_x]
        fit["y_values"] = [float(y) for y in fit_y]
        report["signals"][sig] = fit
        print(
            f"  signal {sig}: R^2={fit['fit_r_squared']:.4f} held-out R^2={fit['heldout_r_squared']:.4f} "
            f"range x[{fit['x_min']:.3f},{fit['x_max']:.3f}] y[{fit['y_min']:.3f},{fit['y_max']:.3f}]",
            flush=True,
        )

    out = Path(__file__).parent / OUTPUT_JSON
    out.write_text(json.dumps(report, indent=2))
    print(f"\nCaptured both signals in {elapsed:.1f}s. Wrote {out}")
    print(
        "Signal SELECTION (A vs B) happens after Phase 3 via sweep_threshold_z_image.py "
        "(usable-curve screen + held-out skip-vs-SSIM knee)."
    )


if __name__ == "__main__":
    main()
