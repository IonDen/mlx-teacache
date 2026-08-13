"""Calibrate FLUX.2 polynomial coefficients for one variant.

Run as: `uv run python scripts/calibrate_flux2.py --variant klein-4b`
        `uv run python scripts/calibrate_flux2.py --variant klein-9b`
        `uv run python scripts/calibrate_flux2.py --variant klein-base-4b --fit-mode origin`
        `uv run python scripts/calibrate_flux2.py --variant klein-base-9b --fit-mode origin`

klein-base-9b ships in v0.5.0 reusing klein-base-4b's polynomial verbatim.
Run this script for klein-base-9b only if you want to override the reused
coefficients with a fresh fit — see docs/calibration.md for when that's
warranted.

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

Output: a JSON report at `scripts/_calibration_flux2_<variant>.json`.

CHUNKED + RESUMABLE. The orchestrator spawns one worker SUBPROCESS per
calibration prompt (fresh MLX memory each, no cross-prompt accumulation),
each writing scripts/_calibration_chunks/<variant>_prompt<NN>.json the
instant it finishes. An interrupted run (throttle, sleep, crash, an approved
kill) RESUMES by re-running only the prompts whose chunk is missing —
completed prompts are never recomputed. Under CFG (`--guidance > 1.0`) each
chunk stores the per-branch (positive/negative) rel-L1 series; the
`--fit-branch-policy` selection (worst/average/positive/negative) is applied
at AGGREGATION time, so re-aggregating under a different policy does not
require re-running the capture.

Validate the chunk/resume/aggregate plumbing with NO model load (seconds, no
GPU):

    uv run python scripts/calibrate_flux2.py --variant klein-4b --dry-run \
        --chunk-dir /tmp/calib_flux2_dry

Memory safety
-------------

Each worker subprocess sets a hard wired-memory cap (`mx.set_wired_limit`)
and a soft cap (`mx.set_memory_limit`) BEFORE constructing the model, taken
from the variant's `META["memory_cap_hint_gb"]` in the mlx-teacache variant
registry when present, else a 20 GB wired / 22 GB soft fallback. An
unconstrained wired peak kernel-panics rather than cleanly OOMing on a 32 GB
M1 Max — see CLAUDE.md "Memory guardrails".
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_teacache.variants.flux2_klein_base_4b.integration import (
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

# Default soft memory cap (GB) when the variant's registry META has no
# memory_cap_hint_gb. Worker derives the hard wired cap as (soft_cap - 2) GB —
# mirrors scripts/bench_speedup.py's worker cap convention.
_DEFAULT_CAP_GB = 22

CHUNK_DIR_DEFAULT = Path(__file__).parent / "_calibration_chunks"


def _model_config_klein_4b() -> Any:
    from mflux.models.common.config.model_config import ModelConfig

    return ModelConfig.flux2_klein_4b()


def _model_config_klein_9b() -> Any:
    from mflux.models.common.config.model_config import ModelConfig

    return ModelConfig.flux2_klein_9b()


def _model_config_klein_base_4b() -> Any:
    from mflux.models.common.config.model_config import ModelConfig

    return ModelConfig.flux2_klein_base_4b()


def _model_config_klein_base_9b() -> Any:
    from mflux.models.common.config.model_config import ModelConfig

    return ModelConfig.flux2_klein_base_9b()


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
        "model_config_factory": _model_config_klein_base_9b,
        # v0.5.0 ships klein-base-9b reusing klein-base-4b's polynomial verbatim
        # (see src/mlx_teacache/coefficients.py). Running this script is only
        # needed if you want to override the reused coefficients with a fresh
        # fit — see docs/calibration.md for when that's warranted.
        "num_inference_steps": 25,
        "output_json": "_calibration_flux2_klein_base_9b.json",
    },
}


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without weights — see
# tests/test_calibrate_flux2_chunking.py).
# ---------------------------------------------------------------------------


def _chunk_filename(variant: str, idx: int) -> str:
    """Per-prompt chunk filename. Zero-padded so lexical order == numeric order."""
    return f"{variant}_prompt{idx:02d}.json"


def _pending_prompt_indices(chunk_dir: Path, variant: str, n_prompts: int) -> list[int]:
    """Prompt indices in [0, n_prompts) whose chunk file does NOT yet exist.

    The resume contract: a finished prompt has written its chunk and is skipped
    on a rerun; an interrupted prompt left no chunk, so it (and only it) reruns.
    """
    return [i for i in range(n_prompts) if not (chunk_dir / _chunk_filename(variant, i)).exists()]


def _select_y(y_pos: float, y_neg: float, *, policy: str) -> float:
    """Apply the CFG fit-branch policy to one (positive, negative) pair.

    Mirrors the per-step branch selection the monolithic script used to make
    inline during capture; extracted so it can run at AGGREGATION time (a
    re-aggregate under a different --fit-branch-policy no longer requires
    re-running the heavy capture)."""
    if policy == "worst":
        return max(y_pos, y_neg)
    if policy == "average":
        return 0.5 * (y_pos + y_neg)
    if policy == "positive":
        return y_pos
    if policy == "negative":
        return y_neg
    raise ValueError(f"unknown fit_branch_policy={policy!r}")


def _accumulate_chunks(
    chunks: list[dict[str, Any]], *, cfg: bool, fit_branch_policy: str = "worst"
) -> dict[str, list[float]]:
    """Merge per-prompt chunk dicts (sorted by idx) into flat xs/ys(/ys_pos/ys_neg)
    lists, preserving prompt order then within-prompt step order — matching the
    order the original monolithic capture loop produced. Pure — no MLX, no I/O —
    so the resume/aggregation logic is unit-testable without weights."""
    acc: dict[str, list[float]] = {"xs": [], "ys": []}
    if cfg:
        acc["ys_pos"] = []
        acc["ys_neg"] = []
    for chunk in sorted(chunks, key=lambda c: int(c["idx"])):
        acc["xs"] += [float(x) for x in chunk["xs"]]
        if cfg:
            pos = [float(y) for y in chunk["ys_pos"]]
            neg = [float(y) for y in chunk["ys_neg"]]
            acc["ys_pos"] += pos
            acc["ys_neg"] += neg
            acc["ys"] += [_select_y(p, n, policy=fit_branch_policy) for p, n in zip(pos, neg, strict=True)]
        else:
            acc["ys"] += [float(y) for y in chunk["ys"]]
    return acc


def _fit_polynomial(xs: list[float], ys: list[float], *, fit_mode: str) -> tuple[list[float], float]:
    """Degree-4 polynomial fit. 'free' = standard numpy.polyfit (c0
    unconstrained); 'origin' = constrained least squares through (0, 0) (c0
    padded to 0.0). Returns (coefficients_c4_to_c0, r_squared) — same math as
    the pre-chunking inline fit, so a full run reproduces the prior output
    byte-for-byte given the same captures."""
    xs_np = np.array(xs)
    ys_np = np.array(ys)
    if fit_mode == "free":
        coeffs = np.polyfit(xs_np, ys_np, 4)
    elif fit_mode == "origin":
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
    return [float(c) for c in coeffs], r2


def _aggregate_path(chunk_dir: Path, variant: str, *, dry_run: bool) -> Path:
    """Where the final aggregated calibration JSON lands. The committed
    scripts/_calibration_flux2_<variant>.json is written ONLY by a real run
    into the default chunk dir; a dry-run or a custom chunk dir writes beside
    its chunks so a smoke never clobbers a committed artifact."""
    output_json: str = _VARIANTS[variant]["output_json"]
    if not dry_run and chunk_dir.resolve() == CHUNK_DIR_DEFAULT.resolve():
        return Path(__file__).parent / output_json
    return chunk_dir / output_json


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


# ---------------------------------------------------------------------------
# WORKER side — captures ONE prompt in a subprocess, writes its chunk file.
# ---------------------------------------------------------------------------


def _run_worker(
    *, variant: str, idx: int, guidance: float, num_inference_steps: int, chunk_dir: Path, dry_run: bool
) -> None:
    """Capture ONE prompt and write its chunk file, then exit. A fresh
    subprocess per prompt = fresh MLX memory (no cross-prompt accumulation) AND
    a durable checkpoint: an interrupted run resumes from the last written
    chunk instead of from zero."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out = chunk_dir / _chunk_filename(variant, idx)
    prompt = CALIBRATION_PROMPTS[idx]
    cfg_capture = guidance > 1.0

    if dry_run:
        # Plumbing smoke: synthetic monotonic pairs, NO model load — exercises the
        # worker -> chunk -> resume -> aggregate -> fit path end-to-end without weights.
        m = max(1, num_inference_steps - 1)
        xs = [round(0.01 * (k + 1), 5) for k in range(m)]
        ys = [round(0.02 * (k + 1), 5) for k in range(m)]
        chunk: dict[str, Any] = {
            "idx": idx,
            "prompt": prompt,
            "num_captures": num_inference_steps,
            "dry_run": True,
            "xs": xs,
            "ys": ys,
        }
        if cfg_capture:
            chunk["ys_pos"] = ys
            chunk["ys_neg"] = ys
        out.write_text(json.dumps(chunk, indent=2))
        print(f"[worker {idx}] dry-run chunk -> {out}", flush=True)
        return

    # --- Memory guardrail (MUST come before the model load). ---
    variant_cfg = _VARIANTS[variant]
    variant_id: str = variant_cfg["variant_id"]
    from mlx_teacache.variants import _REGISTRY

    registry_entry = _REGISTRY.get(variant_id)
    hint = registry_entry["META"].get("memory_cap_hint_gb") if registry_entry is not None else None
    cap_gb = hint if hint is not None else _DEFAULT_CAP_GB
    wired_gb = max(1, cap_gb - 2)
    mx.set_wired_limit(int(wired_gb * 1024**3))
    mx.set_memory_limit(int(cap_gb * 1024**3))
    print(
        f"[worker {idx}] memory caps: wired={wired_gb} GB (hard), memory={cap_gb} GB (soft)",
        flush=True,
    )

    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    print(f"[worker {idx}] loading {variant_id} (quantize=4) for {prompt!r} ...", flush=True)
    flux = Flux2Klein(quantize=4, model_config=variant_cfg["model_config_factory"]())
    flux.freeze()

    capture = _capture_one_prompt(flux, prompt, num_inference_steps=num_inference_steps, guidance=guidance)
    assert len(capture) == num_inference_steps, f"expected {num_inference_steps} captures, got {len(capture)}"

    xs: list[float] = []
    ys: list[float] = []
    ys_pos: list[float] = []
    ys_neg: list[float] = []
    for t in range(1, len(capture)):
        x = _rel_l1(capture[t]["mod_in"], capture[t - 1]["mod_in"])
        if cfg_capture:
            y_pos = _rel_l1(capture[t]["body_out_pos"], capture[t - 1]["body_out_pos"])
            y_neg = _rel_l1(capture[t]["body_out_neg"], capture[t - 1]["body_out_neg"])
            ys_pos.append(y_pos)
            ys_neg.append(y_neg)
        else:
            ys.append(_rel_l1(capture[t]["body_out"], capture[t - 1]["body_out"]))
        xs.append(x)

    chunk: dict[str, Any] = {"idx": idx, "prompt": prompt, "num_captures": len(capture), "xs": xs}
    if cfg_capture:
        # Per-branch series only; the fit-branch policy is applied at
        # aggregation time (see _accumulate_chunks / _select_y).
        chunk["ys_pos"] = ys_pos
        chunk["ys_neg"] = ys_neg
    else:
        chunk["ys"] = ys
    out.write_text(json.dumps(chunk, indent=2))
    print(f"[worker {idx}] wrote {out} (peak {mx.get_peak_memory() / 1024**3:.2f} GB)", flush=True)


# ---------------------------------------------------------------------------
# ORCHESTRATOR side — spawns one worker subprocess per pending prompt, then
# aggregates + fits once every chunk is present.
# ---------------------------------------------------------------------------


def _run_orchestrator(
    *,
    variant: str,
    fit_mode: str,
    guidance: float,
    num_inference_steps: int,
    fit_branch_policy: str,
    chunk_dir: Path,
    dry_run: bool,
) -> None:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_prompts = len(CALIBRATION_PROMPTS)
    pending = _pending_prompt_indices(chunk_dir, variant, n_prompts)
    done = n_prompts - len(pending)
    print(
        f"[orchestrator] variant={variant} {n_prompts} prompts, {done} already done, "
        f"{len(pending)} pending: {pending}",
        flush=True,
    )
    t0 = time.time()
    for idx in pending:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--variant",
            variant,
            "--single-prompt-index",
            str(idx),
            "--guidance",
            str(guidance),
            "--num-inference-steps",
            str(num_inference_steps),
            "--chunk-dir",
            str(chunk_dir),
        ]
        if dry_run:
            cmd.append("--dry-run")
        print(f"[orchestrator] -> prompt {idx} ({CALIBRATION_PROMPTS[idx]!r})", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0 or not (chunk_dir / _chunk_filename(variant, idx)).exists():
            raise SystemExit(
                f"[orchestrator] worker for prompt {idx} failed (rc={result.returncode}); chunk not "
                f"written. Fix the cause and rerun — completed chunks in {chunk_dir} are reused."
            )
    elapsed = time.time() - t0
    print(f"\nCaptured {n_prompts} prompts in {elapsed:.1f}s ({len(pending)} run this pass)")

    chunks = [json.loads((chunk_dir / _chunk_filename(variant, i)).read_text()) for i in range(n_prompts)]
    cfg_capture = guidance > 1.0
    acc = _accumulate_chunks(chunks, cfg=cfg_capture, fit_branch_policy=fit_branch_policy)
    xs, ys = acc["xs"], acc["ys"]

    coeffs, r2 = _fit_polynomial(xs, ys, fit_mode=fit_mode)
    print(f"fit_mode = {fit_mode}")
    print(f"R^2 = {r2:.6f}")
    print("Coefficients (c4, c3, c2, c1, c0):")
    for c in coeffs:
        print(f"  {c:.10g}")

    variant_id: str = _VARIANTS[variant]["variant_id"]
    report: dict[str, Any] = {
        "variant": variant_id,
        "num_inference_steps": num_inference_steps,
        "height": HEIGHT,
        "width": WIDTH,
        "guidance": guidance,
        "seed": SEED,
        "num_prompts": n_prompts,
        "num_pairs": len(xs),
        "elapsed_seconds": elapsed,
        "fit_mode": fit_mode,
        "coefficients_c4_to_c0": coeffs,
        "fit_r_squared": r2,
        "calibration_prompts": list(CALIBRATION_PROMPTS),
        "x_values": xs,  # raw x array for offline refit
        "y_values": ys,  # raw y array for offline refit
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
    }
    if cfg_capture:
        report["fit_branch_policy"] = fit_branch_policy
        report["y_values_pos"] = acc["ys_pos"]
        report["y_values_neg"] = acc["ys_neg"]

    out_path = _aggregate_path(chunk_dir, variant, dry_run=dry_run)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        required=True,
        choices=sorted(_VARIANTS.keys()),
        help="Which variant to calibrate. Wired variants: klein-4b, klein-9b, klein-base-4b, klein-base-9b.",
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
        help="Under CFG calibration, which per-step y target to fit: worst-branch (default), average, positive only, or negative only. Applied at aggregation time.",
    )
    parser.add_argument(
        "--single-prompt-index",
        type=int,
        default=None,
        dest="single_prompt_index",
        help="(internal) worker mode: capture ONE prompt index and write its chunk file, then exit.",
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=None,
        dest="chunk_dir",
        help="Override the per-prompt chunk directory (default scripts/_calibration_chunks/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Worker: write a synthetic chunk with no model load — validates the chunk/resume/aggregate plumbing.",
    )
    args = parser.parse_args()

    variant_cfg = _VARIANTS[args.variant]
    num_inference_steps: int = (
        args.num_inference_steps
        if args.num_inference_steps is not None
        else variant_cfg["num_inference_steps"]
    )
    chunk_dir: Path = args.chunk_dir if args.chunk_dir is not None else CHUNK_DIR_DEFAULT

    if args.single_prompt_index is not None:
        _run_worker(
            variant=args.variant,
            idx=args.single_prompt_index,
            guidance=args.guidance,
            num_inference_steps=num_inference_steps,
            chunk_dir=chunk_dir,
            dry_run=args.dry_run,
        )
        return

    _run_orchestrator(
        variant=args.variant,
        fit_mode=args.fit_mode,
        guidance=args.guidance,
        num_inference_steps=num_inference_steps,
        fit_branch_policy=args.fit_branch_policy,
        chunk_dir=chunk_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
