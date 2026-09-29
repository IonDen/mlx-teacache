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

The capturing closure runs the full vanilla forward (no skips) through the
variant integration's own helpers (`_zimage_t_emb`, `_zimage_prelude`,
`_run_main_layers`, `_zimage_tail`), so the calibration fits the forward the
runtime executes, and returns the real CFG-combined noise so the scheduler
trajectory is correct. A first-step self-check asserts the capture's noise
matches the transformer's own forward (cosine >= 0.999) — a faithful-port guard.

Each step's taps go to an online reducer that keeps only step t-1's arrays and
appends the (t, t-1) rel-L1 floats with the runtime's `mean_abs_rel_l1`, so a
generation never holds more than one previous step of activations.

Run (AFTER the model is downloaded; HEAVY — one full vanilla forward per prompt,
~221s each at the pinned recipe ⇒ ~37 min for 10 prompts):

    uv run python scripts/calibrate_z_image.py --fit-mode origin

Output: scripts/_calibration_z_image.json (both signals' fits + R^2 + curve
range + held-out split + raw arrays for offline refit + the run's
peak_memory_gb).
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_teacache._kernel.gate import mean_abs_rel_l1  # reuse the RUNTIME signal fn
from mlx_teacache.variants.z_image_base.integration import (
    _run_main_layers,
    _zimage_prelude,
    _zimage_t_emb,
    _zimage_tail,
)

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
# Capturing forward: the integration's own prelude / main-layer / tail helpers
# (the runtime forward of ZImageTransformer.__call__), with taps.
# ---------------------------------------------------------------------------


def _zimage_capture_forward(
    transformer: Any, latents: mx.array, t_emb: mx.array, cap_feats: mx.array
) -> dict[str, Any]:
    """Run the Z-Image transformer for ONE branch, tapping the gate signals.

    Returns dict(signal_A, signal_B, main_out, noise). `t_emb` is passed in
    (timestep-only, shared across branches for a given timestep). Signal A is
    the noise-refiner output, i.e. the image part of the unified stream; Signal
    B is layer 0's residual, computed exactly as the runtime gate does. `noise`
    is the negated output — must equal `transformer(...)` for the same inputs.
    """
    pre = _zimage_prelude(transformer, latents, t_emb, cap_feats)
    signal_A = pre.unified_in[:, : pre.x_len]  # noise-refiner output (image-only)
    h1 = transformer.layers[0](
        x=pre.unified_in, attn_mask=pre.attn_mask, freqs_cis=pre.freqs_cis, t_emb=t_emb
    )
    signal_B = h1 - pre.unified_in  # first-main-layer residual (the runtime gate signal)
    main_out = _run_main_layers(transformer, h1, pre, t_emb, start=1)
    return {
        "signal_A": signal_A,
        "signal_B": signal_B,
        "main_out": main_out,
        "noise": _zimage_tail(transformer, main_out, t_emb, pre),
    }


class _ZImagePairReducer:
    """Online consecutive-step reducer: keeps only step t-1's arrays.

    Each `push` reduces (t, t-1) with the runtime's `mean_abs_rel_l1`: x per
    signal (A, and B on the positive branch), y = the worst branch's main_out
    rel-L1 (the positive branch stands in when a step has no negative one), then
    drops step t-1. The same function on the same array pairs as reducing a
    whole-generation capture afterwards, without holding every step."""

    def __init__(self) -> None:
        self._prev: dict[str, mx.array] | None = None
        self._xs: dict[str, list[float]] = {"A": [], "B": []}
        self._ys: list[float] = []
        self.steps = 0

    def push(
        self,
        *,
        signal_A: mx.array,
        signal_B: mx.array,
        main_out_pos: mx.array,
        main_out_neg: mx.array | None = None,
    ) -> None:
        step = {
            "signal_A": signal_A,
            "signal_B": signal_B,
            "main_out_pos": main_out_pos,
            "main_out_neg": main_out_pos if main_out_neg is None else main_out_neg,
        }
        mx.eval(*step.values())
        prev = self._prev
        if prev is not None:
            self._xs["A"].append(_rel_l1(step["signal_A"], prev["signal_A"]))
            self._xs["B"].append(_rel_l1(step["signal_B"], prev["signal_B"]))
            y_pos = _rel_l1(step["main_out_pos"], prev["main_out_pos"])
            y_neg = _rel_l1(step["main_out_neg"], prev["main_out_neg"])
            self._ys.append(max(y_pos, y_neg))  # worst-branch target (matches flux2 'worst' policy)
        self._prev = step
        self.steps += 1

    def series(self, signal_key: str) -> tuple[list[float], list[float]]:
        """(x = signal rel-L1, y = worst-branch main_out rel-L1) pairs across steps."""
        return list(self._xs[signal_key]), list(self._ys)


def _make_capturing_factory(reducer: _ZImagePairReducer, self_check: dict[str, bool]) -> Any:
    def factory(transformer: Any) -> Any:
        def predict(latents, timestep, sigmas, text_encodings, negative_encodings, guidance):  # noqa: ANN001
            t_emb = _zimage_t_emb(transformer, timestep, sigmas)
            pos = _zimage_capture_forward(transformer, latents, t_emb, text_encodings)
            noise = pos["noise"]
            main_out_neg = None
            if negative_encodings is not None:
                neg = _zimage_capture_forward(transformer, latents, t_emb, negative_encodings)
                main_out_neg = neg["main_out"]
                noise = pos["noise"] + guidance * (pos["noise"] - neg["noise"])  # z_image.py:209
            # Faithful-port self-check on the first captured step.
            if not self_check["done"]:
                ref = transformer(timestep=timestep, x=latents, cap_feats=text_encodings, sigmas=sigmas)
                cos = float(mx.sum(ref * pos["noise"]) / (mx.linalg.norm(ref) * mx.linalg.norm(pos["noise"])))
                assert cos >= 0.999, f"re-walk diverges from transformer forward: cos={cos:.6f} (port bug)"
                self_check["done"] = True
            reducer.push(
                signal_A=pos["signal_A"],
                signal_B=pos["signal_B"],
                main_out_pos=pos["main_out"],
                main_out_neg=main_out_neg,
            )
            return noise

        return predict

    return factory


def _capture_one_prompt(flux: Any, prompt: str) -> _ZImagePairReducer:
    reducer = _ZImagePairReducer()
    self_check = {"done": False}
    had = "_predict" in vars(flux)
    original = flux._predict if had else None
    flux._predict = _make_capturing_factory(reducer, self_check)
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
    return reducer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-mode", default="origin", choices=["free", "origin"])
    args = parser.parse_args()

    # Memory guardrail — before any model load (32 GB M1 Max). Set here, not at
    # import, so the module is importable for unit tests without mutating MLX state.
    from _mlx_caps import install_caps

    install_caps(wired_gb=20, soft_gb=22)  # device-clamped wired cap + bounded cache pool
    from _mlx_watchdog import abort_handler, arm_mlx_watchdog

    arm_mlx_watchdog(on_abort=abort_handler("calibrate_z_image"))

    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.z_image.variants.z_image import ZImage

    print(f"Loading Z-Image base (quantize={QUANTIZE})...", flush=True)
    flux = ZImage(quantize=QUANTIZE, model_config=ModelConfig.z_image())
    flux.freeze()

    reducers: list[_ZImagePairReducer] = []
    t0 = time.time()
    for i, prompt in enumerate(CALIBRATION_PROMPTS, 1):
        print(f"[{i}/{len(CALIBRATION_PROMPTS)}] {prompt!r}", flush=True)
        reducer = _capture_one_prompt(flux, prompt)
        assert reducer.steps == NUM_INFERENCE_STEPS, (
            f"expected {NUM_INFERENCE_STEPS} captures, got {reducer.steps}"
        )
        reducers.append(reducer)
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
        "peak_memory_gb": mx.get_peak_memory() / 1024**3,
    }
    for sig in ("A", "B"):
        # Signal B is taken on the positive branch; A is branch-independent.
        fit_x, fit_y, held_x, held_y = [], [], [], []
        for pi, reducer in enumerate(reducers):
            xs, ys = reducer.series(sig)
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
