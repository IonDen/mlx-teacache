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

Memory safety
-------------

Each worker calls mx.set_wired_limit (hard cap on non-pageable Metal
allocations) AND mx.set_memory_limit (soft secondary) BEFORE the model loads.
The cap is taken from the variant's META["memory_cap_hint_gb"] in _REGISTRY, or
22 GB by default, or overridden via --cap-gb.  Running vanilla then wrapper in
the same process was confirmed to panic the kernel watchdog on 2026-05-19
22:17 and 2026-05-20 20:35 on a 32 GB M1 Max. Subprocess isolation prevents
wired-memory accumulation.

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
from pathlib import Path
from typing import Any, cast

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
}

# Default soft memory cap (GB) per variant when _REGISTRY META is absent.
# Workers derive the hard wired cap as (soft_cap - 2) GB.
_DEFAULT_CAP_GB = 22


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
    else:
        raise ValueError(f"unsupported variant: {variant!r}")
    flux.freeze()
    return flux


def _generate(
    flux: Any,
    *,
    num_inference_steps: int,
    guidance: float,
    save_path: Path | None = None,
) -> tuple[float, Any]:
    """Time one generation. Flushes GPU before stopping the clock."""
    import mlx.core as mx

    start = time.perf_counter()
    image = flux.generate_image(
        prompt=PROMPT,
        seed=SEED,
        num_inference_steps=num_inference_steps,
        height=HEIGHT,
        width=WIDTH,
        guidance=guidance,
    )
    mx.eval(mx.zeros(1))  # flush GPU work before stopping the clock
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
    mx.set_wired_limit(int(wired_gb * 1024**3))
    mx.set_memory_limit(int(cap_gb * 1024**3))
    print(
        f"  [worker] memory caps: wired={wired_gb} GB (hard), memory={cap_gb} GB (soft)",
        flush=True,
    )

    variant = args.variant
    condition = args.condition
    rep = args.rep
    recipe = _VARIANT_RECIPE[variant]
    num_inference_steps: int = (
        args.num_inference_steps if args.num_inference_steps is not None else recipe["num_inference_steps"]
    )
    guidance: float = args.guidance if args.guidance is not None else recipe["guidance"]
    save_path: Path | None = Path(args.save_to) if args.save_to else None

    flux = _load_flux(variant)

    stats_summary: dict[str, Any] = {}
    elapsed: float

    if condition == "vanilla":
        elapsed, _ = _generate(
            flux,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
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
                save_path=save_path,
            )
            stats_summary = {
                "skipped_count": handle.stats.skipped_count,
                "computed_count": handle.stats.computed_count,
                "rel_l1_thresh_used": 0.0,
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
                save_path=save_path,
            )
            stats_summary = {
                "skipped_count": handle.stats.skipped_count,
                "computed_count": handle.stats.computed_count,
                "rel_l1_thresh_used": handle.rel_l1_thresh,
            }
        print(
            f"  wrapper rep {rep + 1}: {elapsed:.2f}s "
            f"(skipped {stats_summary['skipped_count']}/{num_inference_steps})",
            flush=True,
        )
    else:
        raise ValueError(f"unknown --condition {condition!r}")

    peak_memory_gb = mx.get_peak_memory() / 1024**3

    result: dict[str, Any] = {
        "variant": variant,
        "condition": condition,
        "rep": rep,
        "elapsed_s": elapsed,
        "peak_memory_gb": peak_memory_gb,
        "stats_summary": stats_summary,
    }
    print(f"{WORKER_RESULT_SENTINEL}{json.dumps(result)}", flush=True)


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
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed for {label}: exit {proc.returncode}")
    for line in proc.stdout.splitlines():
        if line.startswith(WORKER_RESULT_SENTINEL):
            return cast(dict[str, Any], json.loads(line[len(WORKER_RESULT_SENTINEL) :]))
    raise RuntimeError(f"worker for {label} did not emit a {WORKER_RESULT_SENTINEL} result line")


def _run_condition(
    *,
    variant: str,
    condition: str,
    reps: int,
    cap_gb: int | None,
    num_inference_steps: int | None,
    guidance: float | None,
    bench_dir: Path,
    three_way: bool,
) -> list[dict[str, Any]]:
    """Run all reps for one (variant, condition). Returns list of worker result dicts."""
    results: list[dict[str, Any]] = []
    for rep in range(reps):
        # Save image only on rep 0.
        if rep == 0:
            if condition == "vanilla":
                save_to = bench_dir / "vanilla.png"
            elif condition == "wrapper_nogate":
                save_to = bench_dir / "wrapper_nogate.png"
            else:  # wrapper
                save_to = bench_dir / ("wrapper_gated.png" if three_way else "wrapper.png")
        else:
            save_to = None
        result = _run_one_worker(
            variant=variant,
            condition=condition,
            rep=rep,
            cap_gb=cap_gb,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            save_to=save_to,
        )
        results.append(result)
    return results


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
            "Soft MLX memory cap in GB (mx.set_memory_limit). The worker also sets a "
            "hard wired cap at (cap - 2) GB via mx.set_wired_limit. Defaults to the "
            "variant META's memory_cap_hint_gb (e.g. 24 GB on klein-base-9b) or "
            f"{_DEFAULT_CAP_GB} GB otherwise. See CLAUDE.md 'Memory guardrails'."
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

    all_results: dict[str, list[dict[str, Any]]] = {}

    for condition in conditions:
        print(f"\n== {condition} x{reps} ==")
        results = _run_condition(
            variant=variant,
            condition=condition,
            reps=reps,
            cap_gb=cap_gb,
            num_inference_steps=num_inference_steps if args.num_inference_steps is not None else None,
            guidance=guidance if args.guidance is not None else None,
            bench_dir=bench_dir,
            three_way=three_way,
        )
        all_results[condition] = results

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
            "schema_version": 2,
            "isolation": "subprocess-per-rep",
            "variant": variant,
            "num_inference_steps": num_inference_steps,
            "guidance": guidance,
            "prompt": PROMPT,
            "seed": SEED,
            "height": HEIGHT,
            "width": WIDTH,
            "reps": reps,
            "hardware": _detect_hardware(quantize=_VARIANT_QUANTIZE[variant]),
            "vanilla_seconds": vanilla_times,
            "wrapper_seconds": wrapper_times,
            "vanilla_median": vanilla_med,
            "wrapper_median": wrapper_med,
            "speedup_median": speedup,
            "skipped_counts": skipped_counts,
            "computed_counts": computed_counts,
            "bench_images_dir": str(bench_dir),
            "vanilla_peak_memory_gb": [r["peak_memory_gb"] for r in all_results["vanilla"]],
            "wrapper_peak_memory_gb": [r["peak_memory_gb"] for r in all_results["wrapper"]],
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
        args.report.write_text(json.dumps(report_data, indent=2))
        print(f"  report:              {args.report}")


if __name__ == "__main__":
    main()
