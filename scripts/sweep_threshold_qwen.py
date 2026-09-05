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
from _bench_telemetry import streak_telemetry as _streak_telemetry
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


def _units(thresholds: list[float] | None = None) -> list[str]:
    """Sweep units in run order: the vanilla reference FIRST (threshold units read
    its image + timing for SSIM and speedup), then one per threshold."""
    return ["vanilla"] + [_threshold_name(t) for t in (THRESHOLDS if thresholds is None else thresholds)]


def _image_dir(plain_q4: bool) -> Path:
    """Per-build image directory so a plain-q4 sweep never overwrites the
    mixed-precision PNGs the June sweep produced."""
    return OUT_DIR / "plainq4" if plain_q4 else OUT_DIR


def _chunk_filename(unit: str) -> str:
    return f"chunk_{unit}.json"


def _pending_units(chunk_dir: Path, units: list[str]) -> list[str]:
    """Units whose chunk file does not yet exist, in run order (vanilla first)."""
    return [u for u in units if not (chunk_dir / _chunk_filename(u)).exists()]


def _build_summary(
    threshold_chunks: list[dict[str, Any]], vanilla_seconds: float, *, build: str = "mixed-precision"
) -> dict[str, Any]:
    """Assemble results_qwen.json from the per-threshold chunks + the vanilla
    timing. Speedup is derived here from a single vanilla_seconds source. Pure.
    Streak telemetry defaults to empty for chunks written before it existed."""
    results = sorted(
        (
            {
                "threshold": c["threshold"],
                "wrapper_seconds": c["wrapper_seconds"],
                "speedup_vs_vanilla_single_rep": vanilla_seconds / c["wrapper_seconds"],
                "skipped": c["skipped"],
                "computed": c["computed"],
                "ssim_vs_vanilla": c["ssim_vs_vanilla"],
                "skip_pattern": str(c.get("skip_pattern", "")),
                "max_consecutive_skips": int(c.get("max_consecutive_skips", 0)),
            }
            for c in threshold_chunks
        ),
        key=lambda r: r["threshold"],
    )
    return {
        "variant": "qwen-image",
        "signal": "A",
        "build": build,
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


def _run_worker(
    unit: str, *, chunk_dir: Path, dry_run: bool, plain_q4: bool = False, headroom_gib: float = 4.0
) -> None:
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

    # Memory guardrail — device-clamped wired cap, bounded cache pool (1 GiB), and
    # the active+cache watchdog, all before any model load. The plain-q4 recipe's
    # ~26.2 GiB active peak clears the default 28 GiB ceiling by under 2 GiB, so the
    # watchdog is a live gate here (--headroom-gib tunes it); the soft cap stays at
    # the bench recipe's 22 GB so the sweep and the bench share one memory setup.
    from _mlx_caps import install_caps
    from _mlx_watchdog import arm_mlx_watchdog

    install_caps(wired_gb=21, soft_gb=22, cache_gb=1.0)

    def _on_abort(payload: dict[str, int]) -> None:
        print(
            f"[worker {unit}] ABORTED by the memory watchdog: "
            f"{payload['resident_bytes'] / 1024**3:.2f} GB resident > "
            f"{payload['ceiling_bytes'] / 1024**3:.2f} GB ceiling",
            flush=True,
        )
        (chunk_dir / f"{unit}.aborted.json").write_text(json.dumps({"unit": unit, **payload}, indent=2))

    arm_mlx_watchdog(on_abort=_on_abort, headroom_gib=headroom_gib)
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

    from mlx_teacache import apply_teacache

    img_dir = _image_dir(plain_q4)
    img_dir.mkdir(parents=True, exist_ok=True)
    build = "plain-q4" if plain_q4 else "mixed-precision"
    if not plain_q4:
        # Showcase quality: mixed-precision (q8 edge blocks + bf16 embeddings) clears the
        # uniform-q4 grain. mlx-teacache stays quant-agnostic; this is a construction-time
        # choice. Peaks ~30.4 GB, above the watchdog ceiling on a 32 GB Mac; --plain-q4 is
        # the shipped recipe (the one bench_speedup measures) and the one the default
        # threshold is chosen on.
        from qwen_mixed_precision import enable_qwen_mixed_precision

        enable_qwen_mixed_precision()
    print(f"[worker {unit}] loading qwen-image ({build}) ...", flush=True)
    flux = QwenImage(quantize=QUANTIZE, model_config=ModelConfig.qwen_image())
    flux.freeze()

    if unit == "vanilla":
        van_t = _gen(flux, save_path=img_dir / "vanilla.png")
        out.write_text(json.dumps({"unit": "vanilla", "vanilla_seconds": van_t, "build": build}, indent=2))
        print(
            f"[worker vanilla] {van_t:.1f}s (peak {mx.get_peak_memory() / 1024**3:.2f} GB) -> {out}",
            flush=True,
        )
        return

    t = float(unit[1:])
    van_path = img_dir / "vanilla.png"
    if not van_path.exists():
        raise SystemExit(f"[worker {unit}] vanilla.png missing — the vanilla unit must run first")
    with apply_teacache(flux, rel_l1_thresh=t) as h:
        wrap_t = _gen(flux, save_path=img_dir / f"{unit}.png")
        skipped, computed = h.stats.skipped_count, h.stats.computed_count
        telemetry = _streak_telemetry(h.stats)
    score = float(ssim(_load(van_path), _load(img_dir / f"{unit}.png"), channel_axis=-1, data_range=255))
    chunk = {
        "unit": unit,
        "threshold": t,
        "build": build,
        "wrapper_seconds": wrap_t,
        "skipped": skipped,
        "computed": computed,
        "ssim_vs_vanilla": score,
        **telemetry,
    }
    out.write_text(json.dumps(chunk, indent=2))
    print(
        f"[worker {unit}] skipped={skipped} computed={computed} {wrap_t:.1f}s SSIM={score:.4f} "
        f"(peak {mx.get_peak_memory() / 1024**3:.2f} GB) -> {out}",
        flush=True,
    )


def _run_orchestrator(
    *,
    chunk_dir: Path,
    dry_run: bool,
    thresholds: list[float] | None = None,
    plain_q4: bool = False,
    max_units: int | None = None,
    headroom_gib: float = 4.0,
) -> None:
    """Run the vanilla reference, then one worker SUBPROCESS per threshold
    (sequential — never two 20B loads at once); resume by skipping units whose
    chunk already exists; aggregate into results_qwen.json once all are present.
    ``--max-units N`` bounds one invocation (exit 3 while units remain)."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    sweep = THRESHOLDS if thresholds is None else thresholds
    units = _units(sweep)
    pending = _pending_units(chunk_dir, units)
    print(
        f"[orchestrator] {len(units)} units, {len(units) - len(pending)} done, {len(pending)} pending: {pending}",
        flush=True,
    )
    to_run = pending if max_units is None else pending[:max_units]
    for unit in to_run:  # _pending_units preserves run order (vanilla first)
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
        if plain_q4:
            cmd.append("--plain-q4")
        cmd += ["--headroom-gib", str(headroom_gib)]
        print(f"[orchestrator] -> {unit}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0 or not (chunk_dir / _chunk_filename(unit)).exists():
            aborted = chunk_dir / f"{unit}.aborted.json"
            if aborted.exists():
                raise SystemExit(
                    f"[orchestrator] worker for {unit} was ABORTED by the memory watchdog; artifact "
                    f"{aborted}. Lower the recipe before rerunning — completed chunks are reused."
                )
            raise SystemExit(
                f"[orchestrator] worker for {unit} failed (rc={result.returncode}); chunk not written. "
                f"Fix the cause and rerun — completed chunks in {chunk_dir} are reused."
            )
    if _pending_units(chunk_dir, units):
        remaining = _pending_units(chunk_dir, units)
        print(f"[orchestrator] PARTIAL: {len(remaining)} unit(s) pending {remaining}; re-invoke to continue.")
        sys.exit(3)

    vanilla_seconds = json.loads((chunk_dir / _chunk_filename("vanilla")).read_text())["vanilla_seconds"]
    threshold_chunks = [
        json.loads((chunk_dir / _chunk_filename(_threshold_name(t))).read_text()) for t in sweep
    ]
    build = "plain-q4" if plain_q4 else "mixed-precision"
    summary = _build_summary(threshold_chunks, vanilla_seconds, build=build)
    # Real default-chunk-dir run -> the canonical results file for that build; a
    # dry-run or custom chunk dir writes beside its chunks so a smoke never touches
    # the real artifact.
    canonical_name = "results_qwen_plainq4.json" if plain_q4 else "results_qwen.json"
    out = (
        OUT_DIR / canonical_name
        if not dry_run and chunk_dir.resolve() == _default_chunk_dir(plain_q4).resolve()
        else chunk_dir / canonical_name
    )
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[orchestrator] aggregated {len(sweep)} thresholds ({build}). Wrote {out}")
    for r in summary["thresholds"]:
        print(
            f"  t={r['threshold']:.3f} skipped={r['skipped']} streak={r['max_consecutive_skips']} "
            f"SSIM={r['ssim_vs_vanilla']:.4f} speedup={r['speedup_vs_vanilla_single_rep']:.2f}x",
            flush=True,
        )


def _default_chunk_dir(plain_q4: bool) -> Path:
    return OUT_DIR / "_chunks_plainq4" if plain_q4 else CHUNK_DIR_DEFAULT


def _parse_thresholds(text: str | None) -> list[float] | None:
    """`--thresholds 0.15,0.20` -> [0.15, 0.2]; None keeps the module list."""
    if text is None:
        return None
    values = [float(x) for x in text.split(",") if x.strip()]
    if not values:
        raise SystemExit("--thresholds needs at least one value")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="internal: run ONE unit -> chunk file")
    parser.add_argument("--unit", default=None, help="worker: 'vanilla' or 't<thresh>'")
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=None,
        help="per-unit chunk files; resume reads these (default: the build's own chunk dir)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="synthetic chunks, no model load — validates the plumbing"
    )
    parser.add_argument(
        "--plain-q4",
        action="store_true",
        dest="plain_q4",
        help="sweep the shipped uniform-q4 build (the bench recipe) instead of the mixed-precision showcase build",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="comma-separated override of the threshold list, e.g. 0.15,0.20,0.25",
    )
    parser.add_argument(
        "--headroom-gib",
        type=float,
        default=4.0,
        dest="headroom_gib",
        help="memory the watchdog leaves to the OS (default 4 GiB → a 28 GiB ceiling on a 32 GB Mac)",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=None,
        dest="max_units",
        help="run at most this many pending units this invocation, then exit 3 if any remain",
    )
    args = parser.parse_args()
    chunk_dir: Path = args.chunk_dir if args.chunk_dir is not None else _default_chunk_dir(args.plain_q4)

    if args.worker:
        if args.unit is None:
            parser.error("--worker requires --unit")
        _run_worker(
            args.unit,
            chunk_dir=chunk_dir,
            dry_run=args.dry_run,
            plain_q4=args.plain_q4,
            headroom_gib=args.headroom_gib,
        )
        return
    _run_orchestrator(
        chunk_dir=chunk_dir,
        dry_run=args.dry_run,
        thresholds=_parse_thresholds(args.thresholds),
        plain_q4=args.plain_q4,
        max_units=args.max_units,
        headroom_gib=args.headroom_gib,
    )


if __name__ == "__main__":
    main()
