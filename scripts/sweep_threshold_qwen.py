"""Threshold sweep for qwen-image — picks DEFAULT_THRESH at the SSIM knee.

Generates a vanilla baseline at the pinned recipe, then sweeps the wrapper across
`rel_l1_thresh` values, recording skip count + wall-clock + SSIM-vs-vanilla at
each. DEFAULT_THRESH is set at the visible knee where SSIM holds the high bar
(>= ~0.97-0.99 measured, not the 0.85 test floor).

The committed qwen-image variant gates on Signal A (modulated block-0 input;
calibrated R^2 0.849 at 768x768/50 — chosen over Signal B for caption-independence
+ cheaper skips; see config.py). So this sweeps the one committed signal; there is
no A/B selector. This run also doubles as the skip-path quality validation at the
valid recipe: if a handful of skips crater SSIM, that signals a reconstruction bug.

SEQUENCING: runs AFTER the variant is registered — it calls `apply_teacache(flux,
rel_l1_thresh=...)`, which detects + wraps the QwenImage instance.

CHUNKED + RESUMABLE (HEAVY — vanilla baseline + one wrapped generation per
threshold). Main thread only. The orchestrator runs the vanilla baseline first
(the shared SSIM reference), then spawns one worker SUBPROCESS per threshold
(fresh memory each), writing tests/_artifacts/sweep_qwen/_chunks/chunk_<unit>.json
the instant each finishes. An interrupted run RESUMES by re-running only the units
whose chunk is missing:

    uv run python scripts/sweep_threshold_qwen.py            # run / resume

Validate the chunk/resume/aggregate plumbing with NO model load (seconds, no GPU):

    uv run python scripts/sweep_threshold_qwen.py --dry-run --chunk-dir /tmp/sweep_dry

MEMORY: 20B q4 + CFG peaks ~28.5 GB on a 32 GB M1 Max. The wired cap is
device-derived (NOT a hardcoded literal); subprocess-per-threshold keeps each
generation's peak isolated.

Produces tests/_artifacts/sweep_qwen/{vanilla,t<thresh>}.png + results_qwen.json
(aggregated from the per-unit chunks once all are present).
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
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Pinned recipe — matches the calibration so every artifact is comparable.
# 768x768 / 50 steps is the official Qwen-Image recipe and fits a 32 GB M1 Max.
PROMPT = "a red apple on a wooden table"
SEED = 42
HEIGHT = WIDTH = 768
STEPS = 50
GUIDANCE = 4.0
QUANTIZE = 4

# Coarse sweep; refine around the knee after the first pass.
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

OUT_DIR = Path(__file__).parent.parent / "tests" / "_artifacts" / "sweep_qwen"
CHUNK_DIR_DEFAULT = OUT_DIR / "_chunks"


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without weights).
# ---------------------------------------------------------------------------


def _threshold_name(t: float) -> str:
    """Stable unit/image name for a threshold (matches the t<thresh>.png naming)."""
    return f"t{t:.3f}"


def _units() -> list[str]:
    """Sweep units in run order: the vanilla reference FIRST (threshold units read
    its image + timing for SSIM and speedup), then one per threshold."""
    return ["vanilla"] + [_threshold_name(t) for t in THRESHOLDS]


def _chunk_filename(unit: str) -> str:
    return f"chunk_{unit}.json"


def _pending_units(chunk_dir: Path, units: list[str]) -> list[str]:
    """Units whose chunk file does not yet exist, in run order (vanilla first)."""
    return [u for u in units if not (chunk_dir / _chunk_filename(u)).exists()]


def _build_summary(threshold_chunks: list[dict[str, Any]], vanilla_seconds: float) -> dict[str, Any]:
    """Assemble results_qwen.json from the per-threshold chunks + the vanilla
    timing. Speedup is derived here from a single vanilla_seconds source. Pure."""
    results = sorted(
        (
            {
                "threshold": c["threshold"],
                "wrapper_seconds": c["wrapper_seconds"],
                "speedup_vs_vanilla_single_rep": vanilla_seconds / c["wrapper_seconds"],
                "skipped": c["skipped"],
                "computed": c["computed"],
                "ssim_vs_vanilla": c["ssim_vs_vanilla"],
            }
            for c in threshold_chunks
        ),
        key=lambda r: r["threshold"],
    )
    return {
        "variant": "qwen-image",
        "signal": "A",
        "num_inference_steps": STEPS,
        "guidance": GUIDANCE,
        "quantize": QUANTIZE,
        "prompt": PROMPT,
        "seed": SEED,
        "height": HEIGHT,
        "width": WIDTH,
        "vanilla_seconds": vanilla_seconds,
        "thresholds": results,
        "note": "Single-rep wall-clock (thermal noise; subprocess-per-threshold = cold each); "
        "SSIM is deterministic per threshold. Choose DEFAULT_THRESH at the knee where SSIM holds "
        "the high bar. skipped/computed are per denoising step (one shared CFG gate decision per "
        "step; each skip avoids both branches' 60-block bodies).",
    }


# ---------------------------------------------------------------------------
# Generation (imperative shell).
# ---------------------------------------------------------------------------


def _gen(flux: Any, *, save_path: Path) -> float:
    start = time.perf_counter()
    image = flux.generate_image(
        prompt=PROMPT, seed=SEED, num_inference_steps=STEPS, height=HEIGHT, width=WIDTH, guidance=GUIDANCE
    )
    mx.eval(mx.zeros(1))
    elapsed = time.perf_counter() - start
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # mflux's image.save() AUTO-APPENDS `_1` on a name collision instead of
    # overwriting — which silently left a stale image at save_path and made the
    # SSIM step read the wrong file. Unlink first so the exact name is written.
    if save_path.exists():
        save_path.unlink()
    image.save(path=str(save_path), export_json_metadata=False)
    return elapsed


def _load(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _run_worker(unit: str, *, chunk_dir: Path, dry_run: bool) -> None:
    """Run ONE sweep unit (vanilla or a single threshold) and write its chunk.
    A fresh subprocess per unit = isolated peak memory + a durable checkpoint."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    out = chunk_dir / _chunk_filename(unit)

    if dry_run:
        if unit == "vanilla":
            chunk: dict[str, Any] = {"unit": "vanilla", "vanilla_seconds": 10.0, "dry_run": True}
        else:
            t = float(unit[1:])
            chunk = {
                "unit": unit,
                "threshold": t,
                "dry_run": True,
                "wrapper_seconds": round(10.0 - 5.0 * t, 4),
                "skipped": int(40 * t),
                "computed": STEPS - int(40 * t),
                "ssim_vs_vanilla": round(0.999 - 0.25 * t, 4),
            }
        out.write_text(json.dumps(chunk, indent=2))
        print(f"[worker {unit}] dry-run chunk -> {out}", flush=True)
        return

    # Memory guardrail — device-derived wired cap, strictly below the recommended
    # working set, before any model load.
    _max_set = mx.device_info()["max_recommended_working_set_size"]
    mx.set_wired_limit(int(_max_set * 0.85))
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
    from qwen_mixed_precision import enable_qwen_mixed_precision

    from mlx_teacache import apply_teacache

    # Showcase quality: mixed-precision (q8 edge blocks + bf16 embeddings) clears the
    # uniform-q4 grain. mlx-teacache stays quant-agnostic; this is a construction-time
    # choice. The sweep on this model validates the stock-q4 coefficients still skip
    # well at high SSIM (coefficient transfer).
    enable_qwen_mixed_precision()
    print(f"[worker {unit}] loading qwen-image (mixed-precision q4) ...", flush=True)
    flux = QwenImage(quantize=QUANTIZE, model_config=ModelConfig.qwen_image())
    flux.freeze()

    if unit == "vanilla":
        van_t = _gen(flux, save_path=OUT_DIR / "vanilla.png")
        out.write_text(json.dumps({"unit": "vanilla", "vanilla_seconds": van_t}, indent=2))
        print(
            f"[worker vanilla] {van_t:.1f}s (peak {mx.get_peak_memory() / 1024**3:.2f} GB) -> {out}",
            flush=True,
        )
        return

    t = float(unit[1:])
    van_path = OUT_DIR / "vanilla.png"
    if not van_path.exists():
        raise SystemExit(f"[worker {unit}] vanilla.png missing — the vanilla unit must run first")
    with apply_teacache(flux, rel_l1_thresh=t) as h:
        wrap_t = _gen(flux, save_path=OUT_DIR / f"{unit}.png")
        skipped, computed = h.stats.skipped_count, h.stats.computed_count
    score = float(ssim(_load(van_path), _load(OUT_DIR / f"{unit}.png"), channel_axis=-1, data_range=255))
    chunk = {
        "unit": unit,
        "threshold": t,
        "wrapper_seconds": wrap_t,
        "skipped": skipped,
        "computed": computed,
        "ssim_vs_vanilla": score,
    }
    out.write_text(json.dumps(chunk, indent=2))
    print(
        f"[worker {unit}] skipped={skipped} computed={computed} {wrap_t:.1f}s SSIM={score:.4f} "
        f"(peak {mx.get_peak_memory() / 1024**3:.2f} GB) -> {out}",
        flush=True,
    )


def _run_orchestrator(*, chunk_dir: Path, dry_run: bool) -> None:
    """Run the vanilla reference, then one worker SUBPROCESS per threshold
    (sequential — never two 20B loads at once); resume by skipping units whose
    chunk already exists; aggregate into results_qwen.json once all are present."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    units = _units()
    pending = _pending_units(chunk_dir, units)
    print(
        f"[orchestrator] {len(units)} units, {len(units) - len(pending)} done, {len(pending)} pending: {pending}",
        flush=True,
    )
    for unit in pending:  # _pending_units preserves run order (vanilla first)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--unit",
            unit,
            "--chunk-dir",
            str(chunk_dir),
        ]
        if dry_run:
            cmd.append("--dry-run")
        print(f"[orchestrator] -> {unit}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0 or not (chunk_dir / _chunk_filename(unit)).exists():
            raise SystemExit(
                f"[orchestrator] worker for {unit} failed (rc={result.returncode}); chunk not written. "
                f"Fix the cause and rerun — completed chunks in {chunk_dir} are reused."
            )
        mx.clear_cache()

    vanilla_seconds = json.loads((chunk_dir / _chunk_filename("vanilla")).read_text())["vanilla_seconds"]
    threshold_chunks = [
        json.loads((chunk_dir / _chunk_filename(_threshold_name(t))).read_text()) for t in THRESHOLDS
    ]
    summary = _build_summary(threshold_chunks, vanilla_seconds)
    # Real default-chunk-dir run -> the canonical results file; a dry-run or custom
    # chunk dir writes beside its chunks so a smoke never touches the real artifact.
    out = (
        OUT_DIR / "results_qwen.json"
        if not dry_run and chunk_dir.resolve() == CHUNK_DIR_DEFAULT.resolve()
        else chunk_dir / "results_qwen.json"
    )
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[orchestrator] aggregated {len(THRESHOLDS)} thresholds. Wrote {out}")
    for r in summary["thresholds"]:
        print(
            f"  t={r['threshold']:.3f} skipped={r['skipped']} SSIM={r['ssim_vs_vanilla']:.4f} "
            f"speedup={r['speedup_vs_vanilla_single_rep']:.2f}x",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="internal: run ONE unit -> chunk file")
    parser.add_argument("--unit", default=None, help="worker: 'vanilla' or 't<thresh>'")
    parser.add_argument(
        "--chunk-dir", type=Path, default=CHUNK_DIR_DEFAULT, help="per-unit chunk files; resume reads these"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="synthetic chunks, no model load — validates the plumbing"
    )
    args = parser.parse_args()

    if args.worker:
        if args.unit is None:
            parser.error("--worker requires --unit")
        _run_worker(args.unit, chunk_dir=args.chunk_dir, dry_run=args.dry_run)
        return
    _run_orchestrator(chunk_dir=args.chunk_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
