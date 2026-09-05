"""Wall-clock benchmark: vanilla mflux vs. mlx-teacache wrapper.

Produces the speedup numbers cited in README.md's "Benchmarks" table. Pins
seed, prompt, image dimensions, step count, and dtype so the result is
reproducible across runs (within the bounds of Metal / thermal noise on the
measuring host).

For each --variant:
  1. Load the model inside a fresh subprocess.
  2. Run --reps generations per (condition, rep) — every combination spawns
     an isolated Python interpreter so each timing starts from a clean
     MLX allocator state ("cold" reps are genuinely cold).
  3. Report the median per condition + ratio + per-rep skip counts so the
     wall-clock improvement can be attributed to step-skipping or to
     other causes (e.g. avoiding mflux's mx.compile).

Architecture
------------

Each (variant, condition, rep) runs in a SEPARATE subprocess so timing is
not contaminated by warm MLX kernel state or Metal wired-memory from a prior
run. The same file is both orchestrator and worker — selected by --worker flag.
Workers print one JSON line (prefixed by WORKER_RESULT_SENTINEL) on stdout;
the orchestrator collects + aggregates into the final report.

Run as:
  uv run python scripts/bench_speedup.py --variant klein-9b
  uv run python scripts/bench_speedup.py --variant klein-4b
  uv run python scripts/bench_speedup.py --variant klein-base-4b   # 50-step, g=4.0
  uv run python scripts/bench_speedup.py --variant klein-base-9b   # 50-step, g=4.0
  uv run python scripts/bench_speedup.py --variant flux1-dev

Three-way mode (--three-way) additionally runs a wrapped-no-gate condition
(rel_l1_thresh=0.0) to separate the compile-avoidance effect from the gating
effect:
  A = vanilla                (no wrapper)
  B = wrapped, no gate       (rel_l1_thresh=0.0 — compile-avoidance only)
  C = wrapped, gated         (default threshold — full TeaCache)
  A/B = compile-avoidance contribution  [v0.4 effect]
  B/C = gating contribution             [v0.4.1 effect]

Output: prints summary to stdout, optionally writes JSON via --report.

The script also generates and saves a single vanilla + wrapper image pair
under tests/_artifacts/bench_images/<variant>/ for visual quality comparison.
The bench_images/ directory is git-ignored.

Chunking / resume
-----------------
Every (variant, condition, rep) worker result is persisted the moment the
worker returns, to ``--results-dir/<variant>/<condition>_rep<N>.json``
(default ``tests/_artifacts/bench_chunks/``, git-ignored). A re-invocation
skips chunks whose file exists and the report is written only once every
chunk is present, so one three-way bench can be split into several short,
finite jobs — ``--max-chunks 3`` with ``--reps 3`` runs one condition per
invocation — and an interruption loses at most the in-flight worker. Move the
variant's chunk subdir to the Trash for a fresh measurement.

Memory safety
-------------

Each worker installs three caps BEFORE the model loads (``_mlx_caps.install_caps``):
a device-clamped wired cap (the only hard ceiling; non-pageable Metal memory),
the advisory soft cap, and a bound on MLX's retained cache pool (2 GiB, 1 GiB
for qwen), then arms the active+cache watchdog (``_mlx_watchdog``), which
aborts the worker with exit 3 and a ``::BENCH_RESULT::{"aborted": ...}`` line
the moment resident memory exceeds ``memory_size - 4 GiB``. The orchestrator
persists that payload as ``<condition>_rep<N>.aborted.json`` (never as a chunk)
and exits 4. The soft cap is taken from the variant's META["memory_cap_hint_gb"]
(a cap request, not a peak prediction), 22 GB by default, or --cap-gb. Running
vanilla then wrapper in the same process panicked the kernel on 2026-05-19 and
2026-05-20 on a 32 GB M1 Max; subprocess isolation prevents wired-memory
accumulation, and the watchdog is what stops a pageable-memory paging storm.

Exit codes: 0 report written · 3 PARTIAL (chunks pending, re-invoke) · 4 ABORTED
by the memory watchdog (artifact written, nothing persisted as a result).

Compatibility note
------------------

The orchestrator output format (Summary section + JSON keys in --report) is
backward-compatible with the pre-v0.6.0 format where possible. The new JSON
adds "isolation": "subprocess-per-rep" and per-rep arrays for peak_memory_gb.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from _bench_telemetry import streak_telemetry as _streak_telemetry
from _mlx_watchdog import arm_mlx_watchdog

PROMPT = "a red apple on a wooden table"
SEED = 42
HEIGHT = 512
WIDTH = 512

WORKER_RESULT_SENTINEL = "::BENCH_RESULT::"

# Variants supported by this script. Maps the CLI --variant slug to the
# registry variant_id used in _REGISTRY.
_VARIANT_SLUG_TO_ID: dict[str, str] = {
    "klein-4b": "flux2-klein-4b",
    "klein-9b": "flux2-klein-9b",
    "klein-base-4b": "flux2-klein-base-4b",
    "klein-base-9b": "flux2-klein-base-9b",
    "flux1-dev": "flux1-dev",
    "flux1-schnell": "flux1-schnell",
    "z-image": "z-image-base",
    "qwen": "qwen-image",
}

# Default bench recipe per variant (num_inference_steps, guidance).
_VARIANT_RECIPE: dict[str, dict[str, Any]] = {
    "klein-4b": {"num_inference_steps": 8, "guidance": 1.0},
    "klein-9b": {"num_inference_steps": 8, "guidance": 1.0},
    "klein-base-4b": {"num_inference_steps": 50, "guidance": 4.0},
    "klein-base-9b": {"num_inference_steps": 50, "guidance": 4.0},
    "flux1-dev": {"num_inference_steps": 25, "guidance": 3.5},
    "flux1-schnell": {"num_inference_steps": 4, "guidance": 1.0},
    "z-image": {"num_inference_steps": 50, "guidance": 4.0},
    "qwen": {"num_inference_steps": 50, "guidance": 4.0},
}

# Quantization bits per variant. FLUX variants bench at q4; Z-Image at q8 (its
# pinned recipe — findings 2026-05-31). Reported in the bench JSON hardware block.
_VARIANT_QUANTIZE: dict[str, int] = {
    "klein-4b": 4,
    "klein-9b": 4,
    "klein-base-4b": 4,
    "klein-base-9b": 4,
    "flux1-dev": 4,
    "flux1-schnell": 4,
    "z-image": 8,
    "qwen": 4,
}

# Per-variant render size. Everything benches at the shared 512x512 recipe
# except qwen-image, whose DEFAULT_THRESH=0.30 was calibrated and swept at
# 768x768 (scripts/calibrate_qwen.py, scripts/sweep_threshold_qwen.py) — the
# same size its parity gate uses. Benching it at 512x512 would report a skip
# pattern for an operating point the threshold was never tuned against.
_VARIANT_RESOLUTION: dict[str, tuple[int, int]] = {
    "qwen": (768, 768),
}


def _resolution_for(variant: str) -> tuple[int, int]:
    """Return (height, width) for a CLI variant slug."""
    return _VARIANT_RESOLUTION.get(variant, (HEIGHT, WIDTH))


# Default soft memory cap (GB) per variant when _REGISTRY META is absent.
# Workers derive the hard wired cap as (soft_cap - 2) GB. This is a cap request
# (the wired cap is clamped to the device anyway), not a prediction of the peak.
_DEFAULT_CAP_GB = 22

# MLX cache-pool bound per variant. qwen's active peak at 768x768 q4 is ~26.2 GB
# on a 32 GB machine, so its pool gets 1 GiB to stay under the 28 GiB watchdog
# ceiling; everything else has room for the 2 GiB default.
_VARIANT_CACHE_GB: dict[str, float] = {"qwen": 1.0}
_DEFAULT_CACHE_GB = 2.0


# ---------------------------------------------------------------------------
# WORKER side — runs in a subprocess for one (variant, condition, rep).
# ---------------------------------------------------------------------------


def _load_flux(variant: str) -> Any:
    """Load and freeze the mflux model for the given CLI variant slug."""
    if variant in ("klein-4b", "klein-9b", "klein-base-4b", "klein-base-9b"):
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

        cfg_map = {
            "klein-4b": ModelConfig.flux2_klein_4b,
            "klein-9b": ModelConfig.flux2_klein_9b,
            "klein-base-4b": ModelConfig.flux2_klein_base_4b,
            "klein-base-9b": ModelConfig.flux2_klein_base_9b,
        }
        flux = Flux2Klein(quantize=4, model_config=cfg_map[variant]())
    elif variant in ("flux1-dev", "flux1-schnell"):
        from mflux.models.flux.variants.txt2img.flux import Flux1

        name = "dev" if variant == "flux1-dev" else "schnell"
        flux = Flux1.from_name(name, quantize=4)
    elif variant == "z-image":
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.z_image.variants.z_image import ZImage

        flux = ZImage(quantize=_VARIANT_QUANTIZE[variant], model_config=ModelConfig.z_image())
    elif variant == "qwen":
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

        flux = QwenImage(quantize=_VARIANT_QUANTIZE[variant], model_config=ModelConfig.qwen_image())
    else:
        raise ValueError(f"unsupported variant: {variant!r}")
    flux.freeze()
    return flux


def _generate(
    flux: Any,
    *,
    num_inference_steps: int,
    guidance: float,
    height: int,
    width: int,
    save_path: Path | None = None,
) -> tuple[float, Any]:
    """Time one generation. Flushes GPU before stopping the clock."""
    import mlx.core as mx

    start = time.perf_counter()
    image = flux.generate_image(
        prompt=PROMPT,
        seed=SEED,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        guidance=guidance,
    )
    mx.synchronize()  # drain submitted GPU work; generate_image already returned a host-side image
    elapsed = time.perf_counter() - start
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path=str(save_path), export_json_metadata=False)
    return elapsed, image


def _worker_main(args: argparse.Namespace) -> None:
    """Subprocess entrypoint. Runs ONE (variant, condition, rep) and prints
    a single JSON line prefixed by WORKER_RESULT_SENTINEL on stdout."""
    import mlx.core as mx

    # --- Memory guardrail (MUST come before model load) ---
    # cap_gb is passed by the orchestrator; fall back to _DEFAULT_CAP_GB.
    cap_gb: int = args.cap_gb if args.cap_gb is not None else _DEFAULT_CAP_GB
    # If the variant has a META hint, prefer that (unless orchestrator overrode).
    if args.cap_gb is None:
        variant_id = _VARIANT_SLUG_TO_ID.get(args.variant, "")
        if variant_id:
            from mlx_teacache.variants import _REGISTRY

            registry_entry = _REGISTRY.get(variant_id)
            if registry_entry is not None:
                hint = registry_entry["META"].get("memory_cap_hint_gb")
                if hint is not None:
                    cap_gb = hint

    wired_gb = max(1, cap_gb - 2)
    from _mlx_caps import install_caps

    variant = args.variant
    condition = args.condition
    rep = args.rep
    cache_gb = _VARIANT_CACHE_GB.get(variant, _DEFAULT_CACHE_GB)
    wired_b, soft_b, cache_b = install_caps(wired_gb=wired_gb, soft_gb=cap_gb, cache_gb=cache_gb)
    print(
        f"  [worker] memory caps: wired={wired_b / 1024**3:.2f} GB (hard), "
        f"memory={soft_b / 1024**3:.2f} GB (advisory), cache pool={cache_b / 1024**3:.2f} GB",
        flush=True,
    )

    def _on_abort(payload: dict[str, int]) -> None:
        abort = {"aborted": "active-memory watchdog", "variant": variant, "condition": condition, "rep": rep}
        print(f"{WORKER_RESULT_SENTINEL}{json.dumps({**abort, **payload})}", flush=True)

    arm_mlx_watchdog(on_abort=_on_abort)
    started_at = datetime.now(timezone.utc).isoformat()
    recipe = _VARIANT_RECIPE[variant]
    num_inference_steps: int = (
        args.num_inference_steps if args.num_inference_steps is not None else recipe["num_inference_steps"]
    )
    guidance: float = args.guidance if args.guidance is not None else recipe["guidance"]
    height, width = _resolution_for(variant)
    save_path: Path | None = Path(args.save_to) if args.save_to else None

    flux = _load_flux(variant)
    load_peak = int(mx.get_peak_memory())
    mx.reset_peak_memory()

    stats_summary: dict[str, Any] = {}
    elapsed: float

    if condition == "vanilla":
        elapsed, _ = _generate(
            flux,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            height=height,
            width=width,
            save_path=save_path,
        )
        print(f"  vanilla rep {rep + 1}: {elapsed:.2f}s", flush=True)
    elif condition == "wrapper_nogate":
        from mlx_teacache import apply_teacache

        with apply_teacache(flux, rel_l1_thresh=0.0) as handle:
            elapsed, _ = _generate(
                flux,
                num_inference_steps=num_inference_steps,
                guidance=guidance,
                height=height,
                width=width,
                save_path=save_path,
            )
            stats_summary = {
                "skipped_count": handle.stats.skipped_count,
                "computed_count": handle.stats.computed_count,
                "rel_l1_thresh_used": 0.0,
                **_streak_telemetry(handle.stats),
            }
        print(
            f"  wrapper_nogate rep {rep + 1}: {elapsed:.2f}s "
            f"(skipped {stats_summary['skipped_count']}/{num_inference_steps})",
            flush=True,
        )
    elif condition == "wrapper":
        from mlx_teacache import apply_teacache

        with apply_teacache(flux) as handle:
            elapsed, _ = _generate(
                flux,
                num_inference_steps=num_inference_steps,
                guidance=guidance,
                height=height,
                width=width,
                save_path=save_path,
            )
            stats_summary = {
                "skipped_count": handle.stats.skipped_count,
                "computed_count": handle.stats.computed_count,
                "rel_l1_thresh_used": handle.rel_l1_thresh,
                **_streak_telemetry(handle.stats),
            }
        print(
            f"  wrapper rep {rep + 1}: {elapsed:.2f}s "
            f"(skipped {stats_summary['skipped_count']}/{num_inference_steps}, "
            f"max streak {stats_summary['max_consecutive_skips']}, pattern {stats_summary['skip_pattern']})",
            flush=True,
        )
    else:
        raise ValueError(f"unknown --condition {condition!r}")

    loop_peak = int(mx.get_peak_memory())
    result: dict[str, Any] = {
        "variant": variant,
        "condition": condition,
        "rep": rep,
        "num_inference_steps": num_inference_steps,
        "guidance": guidance,
        "elapsed_s": elapsed,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        **_memory_fields(
            load_peak_bytes=load_peak, loop_peak_bytes=loop_peak, cache_bytes=int(mx.get_cache_memory())
        ),
        "stats_summary": stats_summary,
    }
    print(f"{WORKER_RESULT_SENTINEL}{json.dumps(result)}", flush=True)


def _memory_fields(*, load_peak_bytes: int, loop_peak_bytes: int, cache_bytes: int) -> dict[str, float]:
    """Load-time peak, denoising-loop peak (after reset_peak_memory), the cache pool at
    exit, and the legacy process-lifetime `peak_memory_gb` (max of the two peaks)."""
    gib = 1024**3
    return {
        "peak_memory_gb": max(load_peak_bytes, loop_peak_bytes) / gib,
        "load_peak_memory_gb": load_peak_bytes / gib,
        "loop_peak_memory_gb": loop_peak_bytes / gib,
        "cache_memory_gb": cache_bytes / gib,
    }


# ---------------------------------------------------------------------------
# ORCHESTRATOR side — spawns the workers and assembles the report.
# ---------------------------------------------------------------------------


def _mflux_version() -> str:
    try:
        from importlib.metadata import version

        return version("mflux")
    except Exception:
        return "unknown"


def _mlx_teacache_version() -> str:
    from mlx_teacache import __version__

    return __version__


def _macos_sysctl(key: str) -> str | None:
    """Read a macOS sysctl value as a string. Returns None on failure."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_revision(cwd: Path) -> dict[str, Any]:
    """Commit + dirty flag of the checkout the bench ran from (None outside a repo).
    The package `__version__` freezes at install time, so it cannot identify the code."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(cwd), "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": sha, "git_dirty": bool(status.strip())}


def _detect_hardware(*, quantize: int) -> dict[str, Any]:
    chip = _macos_sysctl("machdep.cpu.brand_string") or platform.processor() or "Apple Silicon"
    ram_bytes_str = _macos_sysctl("hw.memsize")
    ram_gb: int | None = None
    if ram_bytes_str is not None:
        try:
            ram_gb = round(int(ram_bytes_str) / (1024**3))
        except ValueError:
            ram_gb = None
    return {
        "chip": chip,
        "ram_gb": ram_gb,
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "mlx_teacache_version": _mlx_teacache_version(),
        **_git_revision(Path(__file__).resolve().parent.parent),
        "mflux_version": _mflux_version(),
        "quantize": quantize,
        "dtype": "bf16",
    }


def _run_one_worker(
    *,
    variant: str,
    condition: str,
    rep: int,
    cap_gb: int | None,
    num_inference_steps: int | None,
    guidance: float | None,
    save_to: Path | None,
) -> dict[str, Any]:
    """Spawn one worker subprocess and return its parsed result dict."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--variant",
        variant,
        "--condition",
        condition,
        "--rep",
        str(rep),
    ]
    if cap_gb is not None:
        cmd += ["--cap-gb", str(cap_gb)]
    if num_inference_steps is not None:
        cmd += ["--num-inference-steps", str(num_inference_steps)]
    if guidance is not None:
        cmd += ["--guidance", str(guidance)]
    if save_to is not None:
        cmd += ["--save-to", str(save_to)]

    label = f"{variant}/{condition}/rep{rep}"
    print(f"\n>> spawning worker: {label}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    payload = _parse_worker_line(proc.stdout)
    if payload is not None and "aborted" in payload:
        return payload  # the caller persists the abort artifact and stops
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed for {label}: exit {proc.returncode}")
    if payload is None:
        raise RuntimeError(f"worker for {label} did not emit a {WORKER_RESULT_SENTINEL} result line")
    return payload


def _parse_worker_line(stdout: str) -> dict[str, Any] | None:
    """The worker's single sentinel-prefixed JSON line, or None if it never printed one."""
    for line in stdout.splitlines():
        if line.startswith(WORKER_RESULT_SENTINEL):
            return cast(dict[str, Any], json.loads(line[len(WORKER_RESULT_SENTINEL) :]))
    return None


def _image_path_for(bench_dir: Path, condition: str, rep: int, *, three_way: bool) -> Path | None:
    """Image destination for a worker: only rep 0 of each condition saves an image."""
    if rep != 0:
        return None
    if condition == "vanilla":
        return bench_dir / "vanilla.png"
    if condition == "wrapper_nogate":
        return bench_dir / "wrapper_nogate.png"
    return bench_dir / ("wrapper_gated.png" if three_way else "wrapper.png")


def _wrapper_streak_arrays(results: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Per-rep skip patterns + max streaks for the report (empty/0 for pre-telemetry chunks)."""
    return {
        "skip_patterns": [str(r["stats_summary"].get("skip_pattern", "")) for r in results],
        "max_consecutive_skips": [int(r["stats_summary"].get("max_consecutive_skips", 0)) for r in results],
    }


# --- Per-chunk persistence + resume ---------------------------------------
# One (condition, rep) worker = one chunk. Each chunk's result is written to
# disk the instant the worker returns, and a re-invocation skips chunks whose
# file already exists, so a three-way bench can be split into several short
# finite jobs (e.g. one condition per invocation via --max-chunks) and an
# interruption loses at most the in-flight worker.


def _chunk_path(results_dir: Path, condition: str, rep: int) -> Path:
    return results_dir / f"{condition}_rep{rep}.json"


def _pending_chunks(conditions: list[str], reps: int, results_dir: Path) -> list[tuple[str, int]]:
    """(condition, rep) pairs with no persisted result yet, in run order.

    Rep-outer / condition-inner (A0, B0, C0, A1, B1, C1, ...) so slow host-state
    drift over a multi-hour run lands on every condition alike instead of on
    whichever condition happened to run last."""
    return [
        (condition, rep)
        for rep in range(reps)
        for condition in conditions
        if not _chunk_path(results_dir, condition, rep).exists()
    ]


def _abort_path(results_dir: Path, condition: str, rep: int) -> Path:
    return results_dir / f"{condition}_rep{rep}.aborted.json"


def _persist_abort(results_dir: Path, payload: dict[str, Any]) -> Path:
    """Write a watchdog-abort payload beside the chunk it failed to produce. It never
    counts as a result: `_pending_chunks` looks only for `<condition>_rep<N>.json`."""
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = _abort_path(results_dir, str(payload["condition"]), int(payload["rep"]))
    dest.write_text(json.dumps(payload, indent=2))
    return dest


def _memory_arrays(results: list[dict[str, Any]], key: str) -> list[float]:
    """Per-rep memory figures; pre-v0.10.1 chunks fall back to their single peak."""
    return [float(r.get(key, r["peak_memory_gb"])) for r in results]


def _persist_chunk(results_dir: Path, result: dict[str, Any]) -> Path:
    """Write one worker result to its chunk file (atomic replace); return the path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = _chunk_path(results_dir, str(result["condition"]), int(result["rep"]))
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(dest)
    return dest


def _load_chunks(
    conditions: list[str], reps: int, results_dir: Path
) -> dict[str, list[dict[str, Any]]] | None:
    """All persisted worker results keyed by condition (ordered by rep), or None if any is missing."""
    if _pending_chunks(conditions, reps, results_dir):
        return None
    return {
        condition: [
            cast(dict[str, Any], json.loads(_chunk_path(results_dir, condition, rep).read_text()))
            for rep in range(reps)
        ]
        for condition in conditions
    }


def _verify_chunk_recipes(
    conditions: list[str], reps: int, results_dir: Path, *, num_inference_steps: int, guidance: float
) -> None:
    """Refuse to reuse a persisted chunk measured under a different recipe.

    Chunks are keyed by (condition, rep) only, so without this check a re-run
    with another --num-inference-steps / --guidance would silently aggregate
    the old timings under the new report header."""
    expected = {"num_inference_steps": num_inference_steps, "guidance": guidance}
    for condition in conditions:
        for r in range(reps):
            path = _chunk_path(results_dir, condition, r)
            if not path.exists():
                continue
            chunk = json.loads(path.read_text())
            for key, want in expected.items():
                got = chunk.get(key)
                if got is None or float(got) != float(want):
                    raise SystemExit(
                        f"persisted chunk {path} was measured with {key}={got} but this invocation "
                        f"uses {key}={want}; move {results_dir} to the Trash for a fresh measurement"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    # --worker flag puts this invocation in worker mode.
    parser.add_argument(
        "--worker",
        action="store_true",
        help="(internal) run as a worker subprocess for one (variant, condition, rep).",
    )

    # Shared args (used by both orchestrator and worker).
    parser.add_argument(
        "--variant",
        choices=list(_VARIANT_SLUG_TO_ID.keys()),
        help="Variant slug.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="Override the variant's default guidance value.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        dest="num_inference_steps",
        help="Override the variant's default step count.",
    )
    parser.add_argument(
        "--cap-gb",
        type=int,
        default=None,
        dest="cap_gb",
        help=(
            "Advisory MLX memory cap in GB. The worker also sets a device-clamped hard "
            "wired cap at (cap - 2) GB, bounds the MLX cache pool, and arms the "
            "active+cache watchdog. Defaults to the variant META's memory_cap_hint_gb "
            f"(a cap request, not a peak prediction) or {_DEFAULT_CAP_GB} GB otherwise."
        ),
    )

    # Worker-only args.
    parser.add_argument("--condition", help="vanilla / wrapper / wrapper_nogate (worker mode).")
    parser.add_argument("--rep", type=int, default=0, help="Rep index 0-based (worker mode).")
    parser.add_argument(
        "--save-to", default=None, dest="save_to", help="Image destination path (worker mode)."
    )

    # Orchestrator-only args.
    parser.add_argument("--reps", type=int, default=3, help="Timed reps per condition (orchestrator mode).")
    parser.add_argument(
        "--three-way",
        action="store_true",
        default=None,
        dest="three_way",
        help=(
            "Run vanilla + wrapped-no-gate + wrapped-gated conditions. "
            "Three-way mode separates compile-avoidance (A/B) from gating (B/C)."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON output path for the summary.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(__file__).parent.parent / "tests" / "_artifacts" / "bench_images",
        dest="images_dir",
        help="Root directory for saved bench images (one subdir per variant).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "tests" / "_artifacts" / "bench_chunks",
        dest="results_dir",
        help=(
            "Root directory for per-chunk worker results (one subdir per variant; one JSON per "
            "(condition, rep)). Existing chunks are REUSED on re-invocation — move the variant's "
            "subdir to the Trash for a fresh measurement."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        dest="max_chunks",
        help=(
            "Run at most this many pending (condition, rep) chunks this invocation, then exit; "
            "the report is written only once every chunk exists (re-invoke to continue). "
            "With --reps 3, --max-chunks 3 = one condition per invocation. An invocation that "
            "leaves chunks pending exits with status 3."
        ),
    )

    # Legacy alias kept for backward-compat.
    parser.add_argument(
        "--mlx-memory-cap-gb",
        type=int,
        default=None,
        dest="mlx_memory_cap_gb",
        help=argparse.SUPPRESS,  # hidden; use --cap-gb instead
    )

    args = parser.parse_args()

    # --worker → subprocess worker mode.
    if args.worker:
        # Merge legacy alias into cap_gb.
        if args.cap_gb is None and args.mlx_memory_cap_gb is not None:
            args.cap_gb = args.mlx_memory_cap_gb
        if args.variant is None or args.condition is None:
            parser.error("--worker requires --variant and --condition")
        _worker_main(args)
        return

    # Orchestrator mode — --variant is required.
    if args.variant is None:
        parser.error("--variant is required")

    # Merge legacy alias.
    cap_gb = args.cap_gb if args.cap_gb is not None else args.mlx_memory_cap_gb

    variant = args.variant
    recipe = _VARIANT_RECIPE[variant]
    num_inference_steps: int = (
        args.num_inference_steps if args.num_inference_steps is not None else recipe["num_inference_steps"]
    )
    guidance: float = args.guidance if args.guidance is not None else recipe["guidance"]
    reps: int = args.reps
    # three_way defaults to None from argparse (store_true + default=None).
    three_way: bool = bool(args.three_way)

    bench_dir = args.images_dir / variant

    conditions = ["vanilla"]
    if three_way:
        conditions.append("wrapper_nogate")
    conditions.append("wrapper")

    results_dir: Path = args.results_dir / variant
    _verify_chunk_recipes(
        conditions, reps, results_dir, num_inference_steps=num_inference_steps, guidance=guidance
    )
    pending = _pending_chunks(conditions, reps, results_dir)
    total_chunks = len(conditions) * reps
    if len(pending) < total_chunks:
        reused = [f"{c}/rep{r}" for c in conditions for r in range(reps) if (c, r) not in pending]
        print(f"\n== RESUMING: reusing {len(reused)}/{total_chunks} persisted chunks from {results_dir} ==")
        for name in reused:
            print(f"   {name}")
    to_run = pending if args.max_chunks is None else pending[: args.max_chunks]
    print(f"\n== running {len(to_run)} of {len(pending)} pending chunks (reps={reps}) ==")

    for condition, rep in to_run:
        result = _run_one_worker(
            variant=variant,
            condition=condition,
            rep=rep,
            cap_gb=cap_gb,
            num_inference_steps=num_inference_steps if args.num_inference_steps is not None else None,
            guidance=guidance if args.guidance is not None else None,
            save_to=_image_path_for(bench_dir, condition, rep, three_way=three_way),
        )
        if "aborted" in result:
            written = _persist_abort(results_dir, result)
            print(
                f"\n== ABORTED by the memory watchdog: {result['resident_bytes'] / 1024**3:.2f} GB resident "
                f"> {result['ceiling_bytes'] / 1024**3:.2f} GB ceiling; artifact {written}. "
                "No chunk persisted; re-invoke only after lowering the recipe or the caps. =="
            )
            sys.exit(4)
        written = _persist_chunk(results_dir, result)
        print(f">> chunk persisted: {written}", flush=True)

    loaded = _load_chunks(conditions, reps, results_dir)
    if loaded is None:
        remaining = _pending_chunks(conditions, reps, results_dir)
        print(
            f"\n== PARTIAL: {total_chunks - len(remaining)}/{total_chunks} chunks persisted under "
            f"{results_dir}; {len(remaining)} pending — re-invoke to continue. No report written. =="
        )
        sys.exit(3)
    all_results: dict[str, list[dict[str, Any]]] = loaded

    # --- Aggregate ---
    def _times(cond: str) -> list[float]:
        return [r["elapsed_s"] for r in all_results[cond]]

    def _skipped(cond: str) -> list[int]:
        return [r["stats_summary"].get("skipped_count", 0) for r in all_results[cond]]

    def _computed(cond: str) -> list[int]:
        return [r["stats_summary"].get("computed_count", 0) for r in all_results[cond]]

    vanilla_times = _times("vanilla")
    wrapper_times = _times("wrapper")
    vanilla_med = statistics.median(vanilla_times)
    wrapper_med = statistics.median(wrapper_times)
    speedup = vanilla_med / wrapper_med
    skipped_counts = _skipped("wrapper")
    computed_counts = _computed("wrapper")

    print("\n== Summary ==")
    print(f"  variant:             {variant}")
    print("  isolation:           subprocess-per-rep")
    print(f"  num_inference_steps: {num_inference_steps}")
    print(f"  guidance:            {guidance}")
    print(f"  reps:                {reps}")
    print(f"  vanilla median:      {vanilla_med:.2f}s   (all: {[round(x, 2) for x in vanilla_times]})")
    print(f"  wrapper median:      {wrapper_med:.2f}s   (all: {[round(x, 2) for x in wrapper_times]})")
    print(f"  speedup (median):    {speedup:.2f}x")
    print(f"  skipped/computed:    {skipped_counts} / {computed_counts}")

    if three_way:
        nogate_times = _times("wrapper_nogate")
        nogate_med = statistics.median(nogate_times)
        compile_avoidance_ratio = vanilla_med / nogate_med
        gating_ratio = nogate_med / wrapper_med
        combined_ratio = vanilla_med / wrapper_med
        print(
            f"  three-way medians:   vanilla {vanilla_med:.2f}s | no-gate {nogate_med:.2f}s | gated {wrapper_med:.2f}s"
        )
        print(f"  compile-avoidance (vanilla / no-gate): {compile_avoidance_ratio:.2f}x  [v0.4 effect]")
        print(f"  gating          (no-gate / gated):     {gating_ratio:.2f}x  [v0.4.1 effect]")
        print(f"  combined        (vanilla / gated):     {combined_ratio:.2f}x")
        print(f"  bench images:        {bench_dir}/{{vanilla,wrapper_nogate,wrapper_gated}}.png")
    else:
        print(f"  bench images:        {bench_dir}/{{vanilla,wrapper}}.png")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report_data: dict[str, Any] = {
            "schema_version": 3,
            "isolation": "subprocess-per-rep",
            "chunk_order": "rep-outer",
            "variant": variant,
            "num_inference_steps": num_inference_steps,
            "guidance": guidance,
            "prompt": PROMPT,
            "seed": SEED,
            "height": _resolution_for(variant)[0],
            "width": _resolution_for(variant)[1],
            "reps": reps,
            "hardware": _detect_hardware(quantize=_VARIANT_QUANTIZE[variant]),
            "vanilla_seconds": vanilla_times,
            "wrapper_seconds": wrapper_times,
            "vanilla_median": vanilla_med,
            "wrapper_median": wrapper_med,
            "speedup_median": speedup,
            "skipped_counts": skipped_counts,
            "computed_counts": computed_counts,
            **_wrapper_streak_arrays(all_results["wrapper"]),
            "bench_images_dir": str(bench_dir),
            "vanilla_peak_memory_gb": [r["peak_memory_gb"] for r in all_results["vanilla"]],
            "wrapper_peak_memory_gb": [r["peak_memory_gb"] for r in all_results["wrapper"]],
            "vanilla_load_peak_memory_gb": _memory_arrays(all_results["vanilla"], "load_peak_memory_gb"),
            "vanilla_loop_peak_memory_gb": _memory_arrays(all_results["vanilla"], "loop_peak_memory_gb"),
            "wrapper_load_peak_memory_gb": _memory_arrays(all_results["wrapper"], "load_peak_memory_gb"),
            "wrapper_loop_peak_memory_gb": _memory_arrays(all_results["wrapper"], "loop_peak_memory_gb"),
            "chunk_timestamps": {
                cond: [[r.get("started_at"), r.get("finished_at")] for r in all_results[cond]]
                for cond in conditions
            },
        }
        if three_way:
            nogate_times = _times("wrapper_nogate")
            nogate_med = statistics.median(nogate_times)
            compile_avoidance_ratio = vanilla_med / nogate_med
            gating_ratio = nogate_med / wrapper_med
            combined_ratio = vanilla_med / wrapper_med
            report_data["nogate_seconds"] = nogate_times
            report_data["nogate_median"] = nogate_med
            report_data["compile_avoidance_ratio"] = compile_avoidance_ratio
            report_data["gating_ratio"] = gating_ratio
            report_data["combined_ratio"] = combined_ratio
            report_data["nogate_peak_memory_gb"] = [
                r["peak_memory_gb"] for r in all_results["wrapper_nogate"]
            ]
            report_data["nogate_load_peak_memory_gb"] = _memory_arrays(
                all_results["wrapper_nogate"], "load_peak_memory_gb"
            )
            report_data["nogate_loop_peak_memory_gb"] = _memory_arrays(
                all_results["wrapper_nogate"], "loop_peak_memory_gb"
            )
        args.report.write_text(json.dumps(report_data, indent=2))
        print(f"  report:              {args.report}")


if __name__ == "__main__":
    main()
