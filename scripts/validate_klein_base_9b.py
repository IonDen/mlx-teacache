"""One-shot validation that flux2-klein-base-9b's reused base-4b coefficients
work at the canonical 50-step CFG recipe.

Generates one fixed prompt at seed 42, 1024x768, num_inference_steps=50,
guidance=4.0, both vanilla and wrapped via apply_teacache. Decodes through
the VAE, computes SSIM, writes _artifacts/validation_klein_base_9b.json.
Exits non-zero if SSIM < 0.95.

This is a release-gate run for v0.5.0 — not a generic benchmark. Run once
before tagging. Heavy generation; expect ~30-90 min on M1 Max.

## Memory guardrails (see CLAUDE.md "Memory guardrails for heavy generations on 32 GB")

A previous same-process vanilla-then-wrapper run of this script triggered
system-level OOM and crashed the machine on 2026-05-18. This version is
restructured to avoid that:

1. **Subprocess-per-condition.** Vanilla and wrapper each run in a fresh
   subprocess (`--worker --condition {vanilla,wrapper}`). MLX's lazy
   allocator releases everything on process exit; the second condition
   never has to share memory with the first.
2. **Explicit MLX memory cap.** Each worker sets `mx.metal.set_memory_limit`
   before loading the model. Configurable via `--mlx-memory-cap-gb` on the
   orchestrator (default 24 GB, leaves ~8 GB OS headroom on 32 GB Max).
   MLX raises a clean OOM error if the cap is exceeded instead of swap-
   thrashing the OS.
3. **Vanilla image is persisted to disk** between subprocesses (the
   SSIM-comparison loop in the orchestrator reloads both PNGs).

Usage:
  uv run python scripts/validate_klein_base_9b.py
  uv run python scripts/validate_klein_base_9b.py --mlx-memory-cap-gb 26
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Reuse the comparison prompt for continuity with COMPARISON.md.
PROMPT = (
    "Portrait of a young woman with auburn hair and green eyes, soft "
    "golden-hour window light, photorealistic, shallow depth of field, "
    "50mm prime lens, subtle freckles, neutral background, cinematic "
    "color grading."
)
SEED = 42
HEIGHT = 1024
WIDTH = 768
STEPS = 50
GUIDANCE = 4.0
SSIM_THRESHOLD = 0.95

WORKER_RESULT_SENTINEL = "::VALIDATE_RESULT::"


# ---------------------------------------------------------------------------
# Hardware detection (orchestrator side)
# ---------------------------------------------------------------------------


def _detect_hardware() -> dict[str, Any]:
    chip = (
        subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        or "Apple Silicon"
    )
    ram_str = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
    ram_bytes = int(ram_str) if ram_str else 0
    return {
        "chip": chip,
        "ram_gb": round(ram_bytes / (1024**3)),
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
    }


def _check_memory_pressure() -> None:
    """Warn (but don't abort) if system memory pressure is already elevated."""
    try:
        result = subprocess.run(["memory_pressure", "-l", "warn"], capture_output=True, text=True, timeout=5)
        out = (result.stdout + result.stderr).lower()
        if "warn" in out or "critical" in out:
            print(
                "WARNING: system memory pressure is elevated before launch. "
                "Close other apps / wait for it to settle before proceeding.",
                file=sys.stderr,
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


# ---------------------------------------------------------------------------
# Worker side — runs in a subprocess for one condition.
# ---------------------------------------------------------------------------


def _worker(condition: str, save_image_to: Path, mlx_memory_cap_gb: int) -> None:
    """Subprocess entrypoint. Loads the model, runs one generation, prints a
    single JSON line prefixed by WORKER_RESULT_SENTINEL on stdout."""
    import mlx.core as mx

    # Set the memory cap BEFORE the model load. See module docstring.
    mx.metal.set_memory_limit(int(mlx_memory_cap_gb * 1024**3))

    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_9b())
    flux.freeze()

    skipped = 0
    computed = 0
    thresh: float | None = None

    if condition == "vanilla":
        t0 = time.perf_counter()
        image = flux.generate_image(
            prompt=PROMPT,
            seed=SEED,
            num_inference_steps=STEPS,
            height=HEIGHT,
            width=WIDTH,
            guidance=GUIDANCE,
        )
        mx.eval(mx.zeros(1))
        elapsed = time.perf_counter() - t0
    elif condition == "wrapper":
        from mlx_teacache import apply_teacache

        with apply_teacache(flux) as handle:
            t0 = time.perf_counter()
            image = flux.generate_image(
                prompt=PROMPT,
                seed=SEED,
                num_inference_steps=STEPS,
                height=HEIGHT,
                width=WIDTH,
                guidance=GUIDANCE,
            )
            mx.eval(mx.zeros(1))
            elapsed = time.perf_counter() - t0
            skipped = handle.stats.skipped_count
            computed = handle.stats.computed_count
            thresh = handle.rel_l1_thresh
    else:
        raise ValueError(f"unknown condition {condition!r}")

    save_image_to.parent.mkdir(parents=True, exist_ok=True)
    image.image.save(save_image_to, format="PNG")

    peak_gb = float(mx.metal.get_peak_memory()) / (1024**3)

    result = {
        "condition": condition,
        "elapsed_seconds": elapsed,
        "wrapper_skipped": skipped,
        "wrapper_computed": computed,
        "rel_l1_thresh_used": thresh,
        "image_path": str(save_image_to),
        "peak_memory_gb": peak_gb,
    }
    print(f"{WORKER_RESULT_SENTINEL}{json.dumps(result)}", flush=True)


# ---------------------------------------------------------------------------
# Orchestrator side — spawns workers, aggregates SSIM.
# ---------------------------------------------------------------------------


def _spawn_worker(condition: str, save_image_to: Path, mlx_memory_cap_gb: int) -> dict[str, Any]:
    """Run one worker subprocess. Returns the parsed result JSON.

    Stdout is streamed to the parent's stdout so the user sees progress; the
    final `::VALIDATE_RESULT::` line is captured and parsed.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--condition",
        condition,
        "--save-image-to",
        str(save_image_to),
        "--mlx-memory-cap-gb",
        str(mlx_memory_cap_gb),
    ]
    print(f">> spawning worker: {condition} (MLX cap {mlx_memory_cap_gb} GB)")
    last_result_line: str | None = None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith(WORKER_RESULT_SENTINEL):
            last_result_line = line[len(WORKER_RESULT_SENTINEL) :].strip()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"worker {condition!r} exited with code {rc}")
    if last_result_line is None:
        raise SystemExit(f"worker {condition!r} did not emit {WORKER_RESULT_SENTINEL}")
    return json.loads(last_result_line)


def _compute_ssim(vanilla_path: Path, wrapper_path: Path) -> float:
    """Load both PNGs and compute SSIM. Imports are deferred so the
    orchestrator's startup is fast."""
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim_fn

    with Image.open(vanilla_path) as v_img, Image.open(wrapper_path) as w_img:
        v = np.array(v_img.convert("RGB")).astype(np.float32) / 255.0
        w = np.array(w_img.convert("RGB")).astype(np.float32) / 255.0
    return float(ssim_fn(v, w, data_range=1.0, channel_axis=-1))


def _orchestrator_main(args: argparse.Namespace) -> int:
    _check_memory_pressure()
    print(f"=== klein-base-9b validation: {STEPS} steps, guidance={GUIDANCE} ===")
    print(f"    MLX memory cap: {args.mlx_memory_cap_gb} GB (set in each worker)")
    print("    Subprocess-per-condition: vanilla then wrapper, each in a fresh process")

    with tempfile.TemporaryDirectory(prefix="validate-klein-base-9b-") as tmpdir:
        tmp = Path(tmpdir)
        vanilla_path = tmp / "vanilla.png"
        wrapper_path = tmp / "wrapper.png"

        vanilla_result = _spawn_worker("vanilla", vanilla_path, args.mlx_memory_cap_gb)
        wrapper_result = _spawn_worker("wrapper", wrapper_path, args.mlx_memory_cap_gb)

        # SSIM in a third pass so neither generation subprocess holds the
        # numpy + PIL intermediates while a model is also resident.
        print(">> computing SSIM (no model in memory at this point)")
        score = _compute_ssim(vanilla_path, wrapper_path)
        passed = score >= SSIM_THRESHOLD

        # Optionally save the images alongside the report.
        if args.save_images_dir is not None:
            args.save_images_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(vanilla_path, args.save_images_dir / "vanilla.png")
            shutil.copy2(wrapper_path, args.save_images_dir / "wrapper.png")

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
        "hardware": _detect_hardware(),
        "mlx_memory_cap_gb": args.mlx_memory_cap_gb,
        "isolation": "subprocess-per-condition",
        "prompt": PROMPT,
        "seed": SEED,
        "height": HEIGHT,
        "width": WIDTH,
        "num_inference_steps": STEPS,
        "guidance": GUIDANCE,
        "rel_l1_thresh_used": wrapper_result["rel_l1_thresh_used"],
        "vanilla_seconds": vanilla_result["elapsed_seconds"],
        "wrapper_seconds": wrapper_result["elapsed_seconds"],
        "vanilla_peak_memory_gb": vanilla_result["peak_memory_gb"],
        "wrapper_peak_memory_gb": wrapper_result["peak_memory_gb"],
        "wrapper_skipped": wrapper_result["wrapper_skipped"],
        "wrapper_computed": wrapper_result["wrapper_computed"],
        "ssim": score,
        "ssim_threshold": SSIM_THRESHOLD,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nReport written: {args.output}")
    print(f"SSIM: {score:.4f} (threshold {SSIM_THRESHOLD})")
    print(
        f"Vanilla peak: {vanilla_result['peak_memory_gb']:.2f} GB | "
        f"Wrapper peak: {wrapper_result['peak_memory_gb']:.2f} GB"
    )
    print("RESULT: PASS" if passed else "RESULT: FAIL")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="(internal) run a single-condition worker")
    parser.add_argument("--condition", choices=["vanilla", "wrapper"], help="(worker) which condition to run")
    parser.add_argument("--save-image-to", type=Path, help="(worker) where to write the generated image")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "_artifacts" / "validation_klein_base_9b.json",
    )
    parser.add_argument(
        "--save-images-dir",
        type=Path,
        default=Path(__file__).parent.parent / "_artifacts" / "validation_klein_base_9b_images",
        help="optional directory to keep the vanilla + wrapper PNGs after SSIM. Pass empty to skip.",
    )
    parser.add_argument(
        "--mlx-memory-cap-gb",
        type=int,
        default=24,
        help=(
            "MLX memory limit applied via mx.metal.set_memory_limit at the top of "
            "each worker. Default 24 GB leaves ~8 GB OS headroom on a 32 GB Max. "
            "Lower this if other apps are running; raise it on machines with more RAM."
        ),
    )
    args = parser.parse_args()

    if args.worker:
        if args.condition is None or args.save_image_to is None:
            parser.error("--worker requires --condition and --save-image-to")
        _worker(args.condition, args.save_image_to, args.mlx_memory_cap_gb)
        return 0

    return _orchestrator_main(args)


if __name__ == "__main__":
    sys.exit(main())
