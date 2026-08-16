"""Threshold-sweep reproducer for flux2-klein-base-4b's per-variant default.

This is the script that was used to choose `Provenance.default_thresh=0.17`
for `flux2-klein-base-4b` in v0.4.0. It generates a vanilla baseline at the
calibrated 25-step schedule, then sweeps the wrapper across a list of
`rel_l1_thresh` values, measuring skip count + wall-clock + SSIM-vs-vanilla
at each. The 0.17 default sits on the visible knee of the curve.

CHUNKED + RESUMABLE. The orchestrator runs the vanilla reference first, then
spawns one worker SUBPROCESS per threshold (fresh MLX memory each), each
writing tests/_artifacts/sweep_klein_base_4b/_chunks/chunk_<unit>.json the
instant it finishes. An interrupted run (throttle, sleep, crash, an approved
kill) RESUMES by re-running only the units whose chunk is missing.

Run as:
    uv run python scripts/sweep_threshold_klein_base_4b.py

Validate the chunk/resume/aggregate plumbing with NO model load (seconds, no
GPU):
    uv run python scripts/sweep_threshold_klein_base_4b.py --dry-run --chunk-dir /tmp/sweep_dry

Produces:
    tests/_artifacts/sweep_klein_base_4b/vanilla.png
    tests/_artifacts/sweep_klein_base_4b/t<thresh>.png  (one per threshold)
    tests/_artifacts/sweep_klein_base_4b/results.json    (full summary)
    stdout: markdown table

The output directory is gitignored (`tests/_artifacts/`).

Measured on M1 Max 32GB, mflux 0.17.5, quantize=4, 512×512, seed=42,
guidance=1.0, red-apple prompt, 2026-05-17. Single-rep measurements — for
3-rep stable wall-clock numbers (the 1.41× README headline) run
`scripts/bench_speedup.py --variant klein-base-4b` instead. Since each unit
now runs in its own fresh subprocess (memory isolation + resumability), every
timing here is a "cold" first-generation measurement — matching the isolation
already used by scripts/bench_speedup.py — so wall-clock is even less
representative of a warm steady state than the original single-process sweep;
SSIM (the metric that actually picks the threshold knee) is unaffected.

Memory safety
-------------

Each worker subprocess sets a hard wired-memory cap (`mx.set_wired_limit`)
and a soft cap (`mx.set_memory_limit`) BEFORE constructing the model, taken
from flux2-klein-base-4b's `META["memory_cap_hint_gb"]` in the mlx-teacache
variant registry when present, else a 20 GB wired / 22 GB soft fallback.
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

PROMPT = "a red apple on a wooden table"
SEED = 42
HEIGHT = WIDTH = 512
STEPS = 25
VARIANT_ID = "flux2-klein-base-4b"

# Coarse + fine, in one pass. The knee is between 0.165 and 0.175.
THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18]

# Default soft memory cap (GB) when the variant's registry META has no
# memory_cap_hint_gb. Worker derives the hard wired cap as (soft_cap - 2) GB —
# mirrors scripts/bench_speedup.py's worker cap convention.
_DEFAULT_CAP_GB = 22

OUT_DIR = Path(__file__).parent.parent / "tests" / "_artifacts" / "sweep_klein_base_4b"
CHUNK_DIR_DEFAULT = OUT_DIR / "_chunks"


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without weights — see
# tests/test_sweep_klein_base_4b_chunking.py).
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
    """Assemble results.json from the per-threshold chunks + the vanilla timing.
    Speedup is derived here from a single vanilla_seconds source. Pure."""
    results = sorted(
        (
            {
                "threshold": c["threshold"],
                "wrapper_seconds": c["wrapper_seconds"],
                "speedup_vs_vanilla_single_rep": vanilla_seconds / c["wrapper_seconds"],
                "skipped": c["skipped"],
                "computed": c["computed"],
                "ssim_vs_vanilla": c["ssim_vs_vanilla"],
                "image_path": c["image_path"],
            }
            for c in threshold_chunks
        ),
        key=lambda r: r["threshold"],
    )
    return {
        "variant": VARIANT_ID,
        "num_inference_steps": STEPS,
        "guidance": 1.0,
        "prompt": PROMPT,
        "seed": SEED,
        "height": HEIGHT,
        "width": WIDTH,
        "vanilla_seconds": vanilla_seconds,
        "thresholds": results,
        "note": (
            "Single-rep wall-clock measurements; subprocess-per-unit (memory isolation + "
            "resumability) means every timing is a cold first-generation measurement. For the "
            "stable 3-rep median wall-clock numbers reported in the README "
            "(77.5s vanilla / 55.1s wrapper / 1.41x at threshold 0.17) run "
            "`scripts/bench_speedup.py --variant klein-base-4b`. SSIM is stable "
            "across reps since the wrapper output is deterministic at a given threshold."
        ),
    }


def _aggregate_path(chunk_dir: Path, *, dry_run: bool) -> Path:
    """Where results.json lands. A real run into the default chunk dir writes the
    canonical tests/_artifacts/sweep_klein_base_4b/results.json; a dry-run or a
    custom chunk dir writes beside its chunks so a smoke never clobbers it."""
    if not dry_run and chunk_dir.resolve() == CHUNK_DIR_DEFAULT.resolve():
        return OUT_DIR / "results.json"
    return chunk_dir / "results.json"


# ---------------------------------------------------------------------------
# Generation (imperative shell).
# ---------------------------------------------------------------------------


def _gen(flux: Any, *, save_path: Path) -> float:
    start = time.perf_counter()
    image = flux.generate_image(
        prompt=PROMPT,
        seed=SEED,
        num_inference_steps=STEPS,
        height=HEIGHT,
        width=WIDTH,
        guidance=1.0,
    )
    mx.eval(mx.zeros(1))
    elapsed = time.perf_counter() - start
    save_path.parent.mkdir(parents=True, exist_ok=True)
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
                "skipped": int(STEPS * t / 2),
                "computed": STEPS - int(STEPS * t / 2),
                "ssim_vs_vanilla": round(0.999 - 0.25 * t, 4),
                "image_path": f"tests/_artifacts/sweep_klein_base_4b/{unit}.png",
            }
        out.write_text(json.dumps(chunk, indent=2))
        print(f"[worker {unit}] dry-run chunk -> {out}", flush=True)
        return

    # --- Memory guardrail (MUST come before the model load). ---
    from mlx_teacache.variants import _REGISTRY

    registry_entry = _REGISTRY.get(VARIANT_ID)
    hint = registry_entry["META"].get("memory_cap_hint_gb") if registry_entry is not None else None
    cap_gb = hint if hint is not None else _DEFAULT_CAP_GB
    wired_gb = max(1, cap_gb - 2)
    from _mlx_caps import install_caps

    install_caps(wired_gb=wired_gb, soft_gb=cap_gb)  # device-clamped: never above the system wired limit
    print(
        f"[worker {unit}] memory caps: wired={wired_gb} GB (hard), memory={cap_gb} GB (soft)",
        flush=True,
    )

    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    print(f"[worker {unit}] loading {VARIANT_ID} (quantize=4)...", flush=True)
    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
    flux.freeze()

    if unit == "vanilla":
        van_path = OUT_DIR / "vanilla.png"
        van_t = _gen(flux, save_path=van_path)
        out.write_text(json.dumps({"unit": "vanilla", "vanilla_seconds": van_t}, indent=2))
        print(
            f"[worker vanilla] {van_t:.2f}s (peak {mx.get_peak_memory() / 1024**3:.2f} GB) -> {out}",
            flush=True,
        )
        return

    from mlx_teacache import apply_teacache

    t = float(unit[1:])
    van_path = OUT_DIR / "vanilla.png"
    if not van_path.exists():
        raise SystemExit(f"[worker {unit}] vanilla.png missing — the vanilla unit must run first")
    van_arr = _load(van_path)
    wrap_path = OUT_DIR / f"{unit}.png"
    with apply_teacache(flux, rel_l1_thresh=t) as h:
        wrap_t = _gen(flux, save_path=wrap_path)
        skipped = h.stats.skipped_count
        computed = h.stats.computed_count
    wrap_arr = _load(wrap_path)
    score = float(ssim(van_arr, wrap_arr, channel_axis=-1, data_range=255))
    chunk = {
        "unit": unit,
        "threshold": t,
        "wrapper_seconds": wrap_t,
        "skipped": skipped,
        "computed": computed,
        "ssim_vs_vanilla": score,
        "image_path": str(wrap_path.relative_to(OUT_DIR.parent.parent.parent)),
    }
    out.write_text(json.dumps(chunk, indent=2))
    print(
        f"[worker {unit}] skipped={skipped}/{STEPS} time={wrap_t:.2f}s SSIM={score:.4f} "
        f"(peak {mx.get_peak_memory() / 1024**3:.2f} GB) -> {out}",
        flush=True,
    )


def _run_orchestrator(*, chunk_dir: Path, dry_run: bool) -> None:
    """Run the vanilla reference, then one worker SUBPROCESS per threshold
    (sequential — never two loads at once); resume by skipping units whose
    chunk already exists; aggregate into results.json once all are present."""
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

    vanilla_seconds = json.loads((chunk_dir / _chunk_filename("vanilla")).read_text())["vanilla_seconds"]
    threshold_chunks = [
        json.loads((chunk_dir / _chunk_filename(_threshold_name(t))).read_text()) for t in THRESHOLDS
    ]
    summary = _build_summary(threshold_chunks, vanilla_seconds)
    out = _aggregate_path(chunk_dir, dry_run=dry_run)
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n\n== Sweep summary (vanilla {vanilla_seconds:.2f}s) ==")
    print("\n| Threshold | Skipped | Wall-clock | Single-rep speedup | SSIM vs vanilla |")
    print("|---|---|---|---|---|")
    for r in summary["thresholds"]:
        print(
            f"| {r['threshold']:.3f} | {r['skipped']}/{STEPS} | {r['wrapper_seconds']:.2f}s | "
            f"{r['speedup_vs_vanilla_single_rep']:.2f}x | {r['ssim_vs_vanilla']:.4f} |"
        )
    print(f"\n  Results JSON: {out}")
    print(f"  PNGs: {OUT_DIR}/*.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="internal: run ONE unit -> chunk file")
    parser.add_argument("--unit", default=None, help="worker: 'vanilla' or 't<thresh>'")
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=CHUNK_DIR_DEFAULT,
        dest="chunk_dir",
        help="per-unit chunk files; resume reads these",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="synthetic chunks, no model load — validates the plumbing",
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
