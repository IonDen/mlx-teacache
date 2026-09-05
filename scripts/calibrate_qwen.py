"""Calibrate Qwen-Image polynomial coefficients — captures BOTH candidate gate signals.

Qwen-Image is a proxy-transformer variant (no `_predict`, no `mx.compile`): the
TeaCache integration replaces `flux.transformer` and gates between the 60 dual-
stream blocks and the norm/proj tail. mflux's `QwenImage.generate_image`
(qwen_image.py:105-119) calls the transformer TWICE per denoising step — once
with the positive prompt, once with the negative prompt — and combines the two
noises OUTSIDE the transformer via `QwenImage.compute_guided_noise`. So the
capturing transformer here fires once per branch; we record per branch and let
mflux do the CFG combine itself (faithful by construction; see the self-check).

This script captures two candidate gate signals per step per branch and fits a
degree-4 polynomial for each, so the winner can be chosen later by the held-out
skip-vs-SSIM knee (the sweep, run after the variant integration is wired):

  Signal A — modulated block-0 image input rel-L1 (the INTEGRATION's gate signal,
             `_qwen_signal_a`: FLUX-canonical modulated block-0 input). This is
             caption-INDEPENDENT — block-0 modulation comes from
             `time_text_embed(timestep, ...)`, which ignores the caption, so
             signal_A is identical pos/neg every step. (Still recorded per branch
             for a uniform record shape; the fit uses the positive branch.)
  Signal B — first-block image-stream residual rel-L1 (the Z-Image-style fallback:
             `block0_output_image - h_in`). Costs one of the 60 blocks on a skip
             step. Genuinely caption-DEPENDENT (block 0 mixes the encoder stream
             via attention), so the per-branch values differ.

Target predicted: per-step rel-L1 of `body_out` (the full 60-block image-stream
output, `_qwen_run_body`). The runtime cache stores the residual
`body_out - h_in`; we fit against `body_out`'s rel-L1 (the runtime gate signal
fn `mean_abs_rel_l1`), matching calibrate_z_image.py.

The capturing transformer REUSES the integration seams (`_qwen_prelude`,
`_qwen_signal_a`, `_qwen_run_body`, `_qwen_tail`) so the gate validates the real
code path. A first-call self-check asserts the re-walk's per-branch noise matches
the UNWRAPPED `QwenTransformer.__call__` (cosine >= 0.999) — a faithful-port
guard. The CFG combine itself is mflux's own `compute_guided_noise` (not
re-walked), so it is faithful by construction; the per-branch cosine gate is the
port guarantee.

Pinned recipe: q4 / 768x768 / 50 steps / guidance 4.0 / seed 42. 50 steps is the
official Qwen-Image recipe (mflux's 20-step default is a fast-preview value); 768x768
peaks ~28.5 GB on a 32 GB M1 Max, which fits (the wired cap, not the 24.96 GB Metal
working-set guideline, is the panic guard).

CHUNKED + RESUMABLE (HEAVY — one full vanilla forward per prompt, 20B model, two
transformer passes per step). Run only on the MAIN THREAD. The orchestrator spawns
one worker SUBPROCESS per prompt (fresh memory each, no cross-prompt accumulation),
each writing scripts/_calib_qwen_chunks/prompt_NN.json the instant it finishes. An
interrupted run (throttle, sleep, crash, an approved kill) RESUMES by re-running
only the prompts whose chunk is missing — completed prompts are never recomputed:

    uv run python scripts/calibrate_qwen.py --fit-mode origin     # run / resume

Validate the chunk/resume/aggregate plumbing with NO model load (seconds, no GPU):

    uv run python scripts/calibrate_qwen.py --dry-run --max-prompts 3 --steps 4 \
        --chunk-dir /tmp/calib_dry

Pre-flight memory (loads the model, runs ONE generation, prints peak vs ceiling,
writes NO JSON):

    uv run python scripts/calibrate_qwen.py --memory-probe

Output: scripts/_calibration_qwen.json (both signals' fits + R^2 + curve range +
held-out split + raw arrays for offline refit + recipe metadata), aggregated from
the per-prompt chunks once all are present.
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

from mlx_teacache._kernel.gate import mean_abs_rel_l1  # reuse the RUNTIME signal fn
from mlx_teacache.variants.qwen_image.integration import (
    _Prelude,
    _qwen_prelude,
    _qwen_run_body,
    _qwen_signal_a,
    _qwen_tail,
)

# Reuse the origin-constrained fit helper + the prompt set verbatim from the
# Z-Image script (the proven template). If the scripts dir is not importable as a
# package, this falls back to a verbatim copy below (kept identical on purpose).
try:
    from calibrate_z_image import fit_signal
except ImportError:  # pragma: no cover - import-path fallback for the run phase
    fit_signal = None  # type: ignore[assignment]

# --- Pinned recipe. Calibrate + sweep + bench all share it. ---
SEED = 42
# 768²/50-step is the official Qwen-Image recipe. 768² peaks ~28.5 GB on the 32 GB
# M1 Max — it fits (the wired cap bounds non-pageable memory; the excess above it is
# pageable). The earlier 512²/20-step recipe under-resolved detail (512² < native
# 1328²) and under-cooked (20 steps); 768²/50 is the in-scope quality recipe.
HEIGHT = WIDTH = 768
NUM_INFERENCE_STEPS = 50
GUIDANCE = 4.0  # CFG path (two transformer passes per step)
QUANTIZE = 4

# 10 prompts (verbatim from calibrate_z_image.py). Held-out split below.
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

OUTPUT_JSON = "_calibration_qwen.json"
CHUNK_DIR_DEFAULT = Path(__file__).parent / "_calib_qwen_chunks"


# ---------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------


def _chunk_filename(idx: int) -> str:
    """Per-prompt chunk filename. Zero-padded so lexical order == numeric order."""
    return f"prompt_{idx:02d}.json"


def _pending_prompt_indices(chunk_dir: Path, n_prompts: int) -> list[int]:
    """Prompt indices in [0, n_prompts) whose chunk file does NOT yet exist.

    The resume contract: a finished prompt has written its chunk and is skipped
    on a rerun; an interrupted prompt left no chunk, so it (and only it) reruns.
    """
    return [i for i in range(n_prompts) if not (chunk_dir / _chunk_filename(i)).exists()]


def _n_fit(n_prompts: int, n_heldout: int) -> int:
    """Count of FIT prompts (the remainder are held out). Always >= 1; the
    held-out split shrinks first when n_prompts is small (e.g. smoke runs)."""
    return max(1, n_prompts - n_heldout)


def _accumulate_chunks(chunks: list[dict[str, Any]], n_fit: int) -> dict[str, dict[str, list[float]]]:
    """Merge per-prompt chunk dicts into fit/held (x, y) lists per signal.

    Each chunk is {"idx": int, "signal_A": {"xs", "ys"}, "signal_B": {"xs", "ys"}}.
    A chunk is a FIT prompt iff its idx < n_fit. Pure — no MLX, no I/O — so the
    resume/aggregation logic is unit-testable without weights.
    """
    acc: dict[str, dict[str, list[float]]] = {
        sig: {"fit_x": [], "fit_y": [], "held_x": [], "held_y": []} for sig in ("A", "B")
    }
    for chunk in sorted(chunks, key=lambda c: int(c["idx"])):
        is_fit = int(chunk["idx"]) < n_fit
        for sig in ("A", "B"):
            pairs = chunk[f"signal_{sig}"]
            acc[sig]["fit_x" if is_fit else "held_x"] += [float(x) for x in pairs["xs"]]
            acc[sig]["fit_y" if is_fit else "held_y"] += [float(y) for y in pairs["ys"]]
    return acc


def _fit_signal_origin(xs: list[float], ys: list[float], *, fit_mode: str) -> dict[str, Any]:
    """Verbatim copy of calibrate_z_image.fit_signal (source of truth there).

    Used only when `from calibrate_z_image import fit_signal` is unavailable
    (scripts dir not on sys.path as a package). Kept byte-identical in behavior:
    fit_mode 'free' = numpy.polyfit (c0 unconstrained); 'origin' = forced through
    (0,0) via LSQ on the x**4..x columns + c0=0.0. Do NOT use np.polyfit for the
    origin fit. Returns coefficients high-to-low (c4..c0), R^2, and curve range.
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


def _fit(xs: list[float], ys: list[float], *, fit_mode: str) -> dict[str, Any]:
    """Dispatch to the Z-Image helper when importable, else the verbatim copy."""
    if fit_signal is not None:
        return dict(fit_signal(xs, ys, fit_mode=fit_mode))
    return _fit_signal_origin(xs, ys, fit_mode=fit_mode)


def _rel_l1(curr: mx.array, prev: mx.array) -> float:
    """Consecutive-step relative-L1 using the SAME function the runtime gate uses."""
    return float(mean_abs_rel_l1(curr, prev))


# ---------------------------------------------------------------------------
# Capturing re-walk of QwenTransformer.__call__ (qwen_transformer.py:37-72),
# reusing the integration seams so the gate signal validates the real code path.
# ---------------------------------------------------------------------------


def _block0_image_output(
    inner: Any,
    pre: _Prelude,
    *,
    config: Any,
    encoder_hidden_states: mx.array,
    encoder_hidden_states_mask: mx.array,
) -> mx.array:
    """Run ONLY transformer block 0 over the image stream; return its image output.

    Mirrors `_qwen_run_body`'s encoder/rope prep (qwen_transformer.py:51-59) so
    the single-block call sees the same inputs the body would, then invokes
    block 0 once. The block returns (encoder, image); we take the image stream.
    Signal B = block0_image_output - pre.h_in (first-block residual). Extra
    block-0 run is fine for calibration.
    """
    encoder = inner.txt_in(inner.txt_norm(encoder_hidden_states))
    image_rotary_embeddings = inner._compute_rotary_embeddings(
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        pos_embed=inner.pos_embed,
        config=config,
        cond_image_grid=None,
    )
    block0 = inner.transformer_blocks[0]
    _enc1, h1 = block0(
        hidden_states=pre.h_in,
        encoder_hidden_states=encoder,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        text_embeddings=pre.text_embeddings,
        image_rotary_emb=image_rotary_embeddings,
        block_idx=0,
    )
    out: mx.array = h1
    return out


def _qwen_capture_branch(
    inner: Any,
    *,
    t: int,
    config: Any,
    hidden_states: mx.array,
    encoder_hidden_states: mx.array,
    encoder_hidden_states_mask: mx.array,
) -> dict[str, Any]:
    """Re-walk the Qwen transformer for ONE branch, tapping the gate signals.

    Returns dict(signal_A, signal_B, h_in, body_out, noise). `noise` is the
    per-branch transformer output (pre-CFG-combine) — must equal the unwrapped
    `QwenTransformer.__call__` for the same inputs (asserted on the first call).
    """
    pre = _qwen_prelude(inner, t, config, hidden_states)
    signal_A = _qwen_signal_a(inner, pre)  # the integration's runtime gate signal
    block0_image_out = _block0_image_output(
        inner,
        pre,
        config=config,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
    )
    signal_B = block0_image_out - pre.h_in  # first-block residual (fallback signal)
    body_out = _qwen_run_body(
        inner,
        pre,
        config=config,
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        cond_image_grid=None,
    )
    noise = _qwen_tail(inner, body_out, pre)  # per-branch noise; keeps trajectory correct
    return {
        "signal_A": signal_A,
        "signal_B": signal_B,
        "h_in": pre.h_in,
        "body_out": body_out,
        "noise": noise,
    }


class _CapturingTransformer:
    """Drop-in replacement for `flux.transformer` during calibration.

    generate_image calls this once per branch per step (positive then negative;
    qwen_image.py:105-118). We track branch parity with a local counter (even =
    positive, odd = negative), mirroring CfgBranchPairer, and accumulate one
    record per step keyed by branch into `captures`. The CFG combine is done by
    mflux's compute_guided_noise on our per-branch outputs — faithful by
    construction; the per-branch cosine self-check is the port guarantee.
    """

    def __init__(self, inner: Any, captures: list[dict[str, Any]], self_check: dict[str, bool]) -> None:
        self._inner = inner
        self._captures = captures
        self._self_check = self_check
        self._call_idx = 0  # even = positive branch, odd = negative branch

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
    ) -> mx.array:
        positive = (self._call_idx % 2) == 0
        cap = _qwen_capture_branch(
            self._inner,
            t=t,
            config=config,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
        )
        # Faithful-port self-check on the FIRST captured branch, BEFORE continuing.
        # Compare the re-walk's per-branch noise against the UNWRAPPED real
        # QwenTransformer.__call__ for the same inputs (cosine >= 0.999). The CFG
        # combine (compute_guided_noise) is mflux's own — not re-walked — so it is
        # faithful by construction; this per-branch cosine is the port guarantee.
        if not self._self_check["done"]:
            ref = self._inner(
                t=t,
                config=config,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
            )
            noise = cap["noise"]
            cos = float(mx.sum(ref * noise) / (mx.linalg.norm(ref) * mx.linalg.norm(noise)))
            assert cos >= 0.999, f"re-walk diverges from QwenTransformer.__call__: cos={cos:.6f} (port bug)"
            self._self_check["done"] = True

        branch = "pos" if positive else "neg"
        rec: dict[str, Any] = {
            f"signal_A_{branch}": cap["signal_A"],
            f"signal_B_{branch}": cap["signal_B"],
            f"body_out_{branch}": cap["body_out"],
            f"h_in_{branch}": cap["h_in"],
        }
        if positive:
            self._captures.append(rec)  # start a new per-step record on the positive call
        else:
            self._captures[-1].update(rec)  # fold the negative branch into the current step
        mx.eval(*[v for v in rec.values() if isinstance(v, mx.array)])
        self._call_idx += 1
        return cap["noise"]

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else (freeze, parameters, attrs) to the real module.
        return getattr(self.__dict__["_inner"], name)


def _capture_one_prompt(flux: Any, prompt: str, *, steps: int = NUM_INFERENCE_STEPS) -> list[dict[str, Any]]:
    captures: list[dict[str, Any]] = []
    self_check = {"done": False}
    original_transformer = flux.transformer
    flux.transformer = _CapturingTransformer(original_transformer, captures, self_check)
    try:
        flux.generate_image(
            prompt=prompt,
            seed=SEED,
            num_inference_steps=steps,
            height=HEIGHT,
            width=WIDTH,
            guidance=GUIDANCE,
        )
    finally:
        flux.transformer = original_transformer
    return captures


def _pairs_for(
    captures: list[dict[str, Any]], signal_key: str, *, branch: str
) -> tuple[list[float], list[float]]:
    """Build (x=signal rel-L1, y=worst-branch body_out rel-L1) pairs across steps.

    x uses the requested branch. (Signal A is caption-independent, so the branch
    is moot for it; Signal B is caption-dependent, so the branch matters.) The
    target y is the worst-branch body_out rel-L1 (matches the flux2/z_image
    'worst' policy).
    """
    xs: list[float] = []
    ys: list[float] = []
    for t in range(1, len(captures)):
        x = _rel_l1(
            captures[t][f"signal_{signal_key}_{branch}"], captures[t - 1][f"signal_{signal_key}_{branch}"]
        )
        y_pos = _rel_l1(captures[t]["body_out_pos"], captures[t - 1]["body_out_pos"])
        y_neg = _rel_l1(
            captures[t].get("body_out_neg", captures[t]["body_out_pos"]),
            captures[t - 1].get("body_out_neg", captures[t - 1]["body_out_pos"]),
        )
        xs.append(x)
        ys.append(max(y_pos, y_neg))  # worst-branch target
    return xs, ys


def _run_memory_probe() -> None:
    """Load the model, run ONE generation at the pinned recipe (no capture),
    print the peak vs the device working-set ceiling, write NO JSON."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

    print(f"[memory-probe] Loading Qwen-Image (quantize={QUANTIZE})...", flush=True)
    flux = QwenImage(quantize=QUANTIZE, model_config=ModelConfig.qwen_image())
    flux.freeze()
    print(
        f"[memory-probe] One generation at {HEIGHT}x{WIDTH}, {NUM_INFERENCE_STEPS} steps, "
        f"guidance {GUIDANCE}, seed {SEED}...",
        flush=True,
    )
    flux.generate_image(
        prompt=CALIBRATION_PROMPTS[0],
        seed=SEED,
        num_inference_steps=NUM_INFERENCE_STEPS,
        height=HEIGHT,
        width=WIDTH,
        guidance=GUIDANCE,
    )
    peak = mx.get_peak_memory()
    max_set = int(mx.device_info()["max_recommended_working_set_size"])
    headroom = max_set - peak
    print(f"[memory-probe] peak_memory                = {peak / 1024**3:.2f} GB", flush=True)
    print(f"[memory-probe] max_recommended_working_set = {max_set / 1024**3:.2f} GB", flush=True)
    print(f"[memory-probe] headroom                    = {headroom / 1024**3:.2f} GB", flush=True)
    print("[memory-probe] done (no JSON written).", flush=True)


def _run_worker(prompt_idx: int, *, steps: int, chunk_dir: Path, dry_run: bool) -> None:
    """Capture ONE prompt and write its chunk file, then exit.

    A fresh subprocess per prompt = fresh MLX memory (no cross-prompt
    accumulation) AND a durable checkpoint: an interrupted run resumes from the
    last written chunk instead of from zero.
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out = chunk_dir / _chunk_filename(prompt_idx)
    prompt = CALIBRATION_PROMPTS[prompt_idx]

    if dry_run:
        # Plumbing smoke: synthetic monotonic pairs, NO model load — exercises the
        # worker -> chunk -> resume -> aggregate -> fit path end-to-end without weights.
        m = max(1, steps - 1)
        xs = [round(0.01 * (k + 1), 5) for k in range(m)]
        ys = [round(0.02 * (k + 1), 5) for k in range(m)]
        chunk: dict[str, Any] = {
            "idx": prompt_idx,
            "prompt": prompt,
            "num_captures": steps,
            "steps": steps,
            "dry_run": True,
            "signal_A": {"xs": xs, "ys": ys},
            "signal_B": {"xs": xs, "ys": ys},
        }
        out.write_text(json.dumps(chunk, indent=2))
        print(f"[worker {prompt_idx}] dry-run chunk -> {out}", flush=True)
        return

    # Memory guardrail — device-derived wired cap, strictly below the recommended
    # working set, BEFORE any model load (the kernel-panic guard).
    from _mlx_caps import install_caps

    install_caps(
        wired_gb=22, soft_gb=22
    )  # clamps to 0.85 x the recommended working set; bounds the cache pool
    from _mlx_watchdog import abort_handler, arm_mlx_watchdog

    # This recipe peaks ~26.2 GiB active, above the recommended working set; the
    # watchdog is the only thing that stops a paging storm.
    arm_mlx_watchdog(on_abort=abort_handler(f"calibrate_qwen-prompt{prompt_idx}", chunk_dir))
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

    print(f"[worker {prompt_idx}] loading Qwen-Image (q{QUANTIZE}) for {prompt!r} ...", flush=True)
    flux = QwenImage(quantize=QUANTIZE, model_config=ModelConfig.qwen_image())
    flux.freeze()
    caps = _capture_one_prompt(flux, prompt, steps=steps)
    assert len(caps) == steps, f"expected {steps} captures, got {len(caps)}"
    chunk = {"idx": prompt_idx, "prompt": prompt, "num_captures": len(caps), "steps": steps}
    for sig in ("A", "B"):
        xs, ys = _pairs_for(caps, sig, branch="pos")
        chunk[f"signal_{sig}"] = {"xs": [float(x) for x in xs], "ys": [float(y) for y in ys]}
    out.write_text(json.dumps(chunk, indent=2))
    print(f"[worker {prompt_idx}] wrote {out} (peak {mx.get_peak_memory() / 1024**3:.2f} GB)", flush=True)


def _aggregate_path(chunk_dir: Path, *, dry_run: bool) -> Path:
    """Where the final aggregated calibration JSON lands. The committed
    scripts/_calibration_qwen.json is written ONLY by a real run into the default
    chunk dir; a dry-run or a custom chunk dir writes beside its chunks so a smoke
    never clobbers the committed artifact."""
    if not dry_run and chunk_dir.resolve() == CHUNK_DIR_DEFAULT.resolve():
        return Path(__file__).parent / OUTPUT_JSON
    return chunk_dir / OUTPUT_JSON


def _run_orchestrator(*, steps: int, n_prompts: int, chunk_dir: Path, fit_mode: str, dry_run: bool) -> None:
    """Spawn one worker SUBPROCESS per pending prompt (sequential — never two 20B
    loads at once), each writing its chunk on completion; resume by skipping
    prompts whose chunk already exists; aggregate + fit once all are present."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pending = _pending_prompt_indices(chunk_dir, n_prompts)
    done = n_prompts - len(pending)
    print(
        f"[orchestrator] {n_prompts} prompts, {done} already done, {len(pending)} pending: {pending}",
        flush=True,
    )
    t0 = time.time()
    for idx in pending:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--prompt-idx",
            str(idx),
            "--steps",
            str(steps),
            "--chunk-dir",
            str(chunk_dir),
        ]
        if dry_run:
            cmd.append("--dry-run")
        print(f"[orchestrator] -> prompt {idx} ({CALIBRATION_PROMPTS[idx]!r})", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0 or not (chunk_dir / _chunk_filename(idx)).exists():
            raise SystemExit(
                f"[orchestrator] worker for prompt {idx} failed (rc={result.returncode}); chunk not "
                f"written. Fix the cause and rerun — completed chunks in {chunk_dir} are reused."
            )
    gen_seconds = time.time() - t0

    # All chunks present -> aggregate + fit. Same report schema as the monolith.
    chunks = [json.loads((chunk_dir / _chunk_filename(i)).read_text()) for i in range(n_prompts)]
    n_fit = _n_fit(n_prompts, N_HELDOUT)
    acc = _accumulate_chunks(chunks, n_fit)
    report: dict[str, Any] = {
        "variant": "qwen-image",
        "num_inference_steps": steps,
        "guidance": GUIDANCE,
        "height": HEIGHT,
        "width": WIDTH,
        "seed": SEED,
        "quantize": QUANTIZE,
        "num_prompts": n_prompts,
        "n_fit_prompts": n_fit,
        "n_heldout_prompts": n_prompts - n_fit,
        "elapsed_seconds": gen_seconds,
        "fit_mode": fit_mode,
        "signals": {},
        "calibration_prompts": list(CALIBRATION_PROMPTS[:n_prompts]),
        "chunked": True,
        "dry_run": dry_run,
    }
    for sig in ("A", "B"):
        tgt = acc[sig]
        fit = _fit(tgt["fit_x"], tgt["fit_y"], fit_mode=fit_mode)
        p = np.poly1d(fit["coefficients_c4_to_c0"])
        hy = np.asarray(tgt["held_y"])
        hp = p(np.asarray(tgt["held_x"]))
        ss_res = float(np.sum((hy - hp) ** 2))
        ss_tot = float(np.sum((hy - np.mean(hy)) ** 2))
        fit["heldout_r_squared"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        fit["x_values"] = [float(x) for x in tgt["fit_x"]]
        fit["y_values"] = [float(y) for y in tgt["fit_y"]]
        report["signals"][sig] = fit
        print(
            f"  signal {sig}: R^2={fit['fit_r_squared']:.4f} held-out R^2={fit['heldout_r_squared']:.4f} "
            f"range x[{fit['x_min']:.3f},{fit['x_max']:.3f}] y[{fit['y_min']:.3f},{fit['y_max']:.3f}]",
            flush=True,
        )

    out = _aggregate_path(chunk_dir, dry_run=dry_run)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[orchestrator] aggregated {n_prompts} chunks ({gen_seconds:.1f}s generation). Wrote {out}")
    print(
        "Signal SELECTION (A vs B) happens after the variant integration via the "
        "Qwen threshold sweep (usable-curve screen + held-out skip-vs-SSIM knee)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-mode", default="origin", choices=["free", "origin"])
    parser.add_argument(
        "--memory-probe",
        action="store_true",
        help="Load the model, run ONE generation, print peak vs ceiling, exit without writing JSON.",
    )
    parser.add_argument("--worker", action="store_true", help="internal: capture ONE prompt -> chunk file")
    parser.add_argument("--prompt-idx", type=int, default=None, help="worker: which prompt index to capture")
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS, help="override step count (smoke)")
    parser.add_argument(
        "--max-prompts", type=int, default=len(CALIBRATION_PROMPTS), help="limit number of prompts (smoke)"
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=CHUNK_DIR_DEFAULT,
        help="per-prompt chunk files live here; resume reads them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="synthetic chunks, no model load — validates the chunk/resume/aggregate plumbing",
    )
    args = parser.parse_args()

    if args.memory_probe:
        from _mlx_caps import install_caps

        install_caps(
            wired_gb=22, soft_gb=22
        )  # clamps to 0.85 x the recommended working set; bounds the cache pool
        from _mlx_watchdog import abort_handler, arm_mlx_watchdog

        arm_mlx_watchdog(on_abort=abort_handler("calibrate_qwen-memory-probe"))
        _run_memory_probe()
        return
    if args.worker:
        if args.prompt_idx is None:
            parser.error("--worker requires --prompt-idx")
        _run_worker(args.prompt_idx, steps=args.steps, chunk_dir=args.chunk_dir, dry_run=args.dry_run)
        return
    _run_orchestrator(
        steps=args.steps,
        n_prompts=args.max_prompts,
        chunk_dir=args.chunk_dir,
        fit_mode=args.fit_mode,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
