"""Generate COMPARISON.md content: vanilla mflux vs mlx-teacache on the
non-distilled FLUX variants. Produces per-variant webp images + a complete
recoverable JSON report under _artifacts/.

Run as:
  uv run python scripts/bench_comparison.py

Architecture
------------

Each (variant, condition) pair runs in a SEPARATE subprocess so the rep-1
timing is genuinely "cold" — no prior mflux generation in the same Python
process, no warm MLX kernel state. The main script orchestrates 4
subprocesses (2 variants x {vanilla, wrapper}), reads their stdout JSON,
and aggregates into _artifacts/comparison_report.json.

The same script file is the orchestrator AND the per-condition worker —
selected by --condition / --variant flags. Workers print one JSON line at
the end of stdout (their bench result); the orchestrator parses that line
to assemble the final report.

Two entries (non-distilled only, recommended-upstream settings):
  - flux1-dev at 25 steps, guidance=3.5
  - flux2-klein-base-4b at 50 steps, guidance=4.0 (canonical upstream CFG)

The g=1.0 row was dropped: klein-base-4b is NOT guidance-distilled, so
running it at guidance=1.0 produces washed-out output and the wrapper
skips zero steps (no CFG → no caching engagement). The CFG row is the
only meaningful klein-base-4b configuration for this comparison.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any, cast

# Shared portrait prompt — same across every variant. Single variable = recipe.
PROMPT = (
    "Portrait of a young woman with auburn hair and green eyes, soft "
    "golden-hour window light, photorealistic, shallow depth of field, "
    "50mm prime lens, subtle freckles, neutral background, cinematic "
    "color grading."
)
SEED = 42
HEIGHT = 1024
WIDTH = 768
REPS = 3  # rep 1 = cold (subprocess just started); reps 2-3 = warm

WEBP_QUALITY = 88
WEBP_METHOD = 6  # Pillow's slowest+best encoder; ~1-2s on 768x1024.

WORKER_RESULT_SENTINEL = "::BENCH_RESULT::"


@dataclass(frozen=True)
class VariantConfig:
    slug: str  # subdir name under _artifacts/comparison/
    variant_id: str  # registry id (used only for reporting clarity)
    num_inference_steps: int
    guidance: float
    loader: str  # "flux1-dev" / "klein-base-4b" / "z-image"
    # Per-variant overrides. Defaults preserve the shared portrait recipe used by
    # the q4 FLUX rows (768x1024 q4). Z-Image is q8, so it drops resolution to
    # stay under the 32 GB unified-memory ceiling — same PROMPT + SEED, only the
    # resolution changes (per the COMPARISON shared-prompt rule).
    height: int = 1024
    width: int = 768
    quantize: int = 4
    wired_cap_gb: int = (
        22  # mx.set_wired_limit; must stay < max_recommended_working_set_size (25 on M1 Max 32GB)
    )
    # Free the MLX buffer cache between reps. Off for the q4 FLUX rows (their warm
    # reps intentionally reuse the warm allocator). On for q8 Z-Image at 640x896,
    # where a single gen peaks ~18.7 GB but the cache accumulates across reps in
    # one process and OOMs the Metal command buffer on rep 2 without this.
    clear_cache_between_reps: bool = False


VARIANTS: tuple[VariantConfig, ...] = (
    VariantConfig(
        slug="flux1-dev",
        variant_id="flux1-dev",
        num_inference_steps=25,
        guidance=3.5,
        loader="flux1-dev",
    ),
    VariantConfig(
        slug="klein-base-4b-cfg",
        variant_id="flux2-klein-base-4b",
        num_inference_steps=50,
        guidance=4.0,
        loader="klein-base-4b",
    ),
    VariantConfig(
        slug="z-image",
        variant_id="z-image-base",
        num_inference_steps=50,
        guidance=4.0,
        loader="z-image",
        height=896,
        width=640,
        quantize=8,
        wired_cap_gb=24,  # 640x896 q8 peaks higher than the q4 rows; 24 < 25 recommended
        clear_cache_between_reps=True,  # single gen ~18.7 GB; cache accumulation OOMs rep 2 without this
    ),
)


# ---------------------------------------------------------------------------
# WORKER side — runs in a subprocess for one (variant, condition) pair.
# ---------------------------------------------------------------------------


def _load_flux(loader: str, quantize: int) -> Any:
    from mflux.models.common.config.model_config import ModelConfig

    if loader == "flux1-dev":
        from mflux.models.flux.variants.txt2img.flux import Flux1

        flux = Flux1.from_name("dev", quantize=quantize)
    elif loader == "klein-base-4b":
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

        flux = Flux2Klein(quantize=quantize, model_config=ModelConfig.flux2_klein_base_4b())
    elif loader == "z-image":
        from mflux.models.z_image.variants.z_image import ZImage

        flux = ZImage(quantize=quantize, model_config=ModelConfig.z_image())
    else:
        raise ValueError(f"unknown loader: {loader!r}")
    flux.freeze()
    return flux


def _generate(
    flux: Any, *, num_inference_steps: int, guidance: float, height: int, width: int
) -> tuple[float, Any]:
    """Time one flux.generate_image call. Flushes GPU before stopping clock."""
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
    mx.eval(mx.zeros(1))  # flush GPU work before stopping the clock
    elapsed = time.perf_counter() - start
    return elapsed, image


def _save_as_webp(image: Any, dest_webp: Path) -> None:
    """Save mflux's image as webp via PNG intermediate.

    Plan-audit Finding 1 fix: the intermediate file must have a real `.png`
    suffix so Pillow can infer the format. `<stem>.tmp.png` (NOT
    `<stem>.png.tmp`) keeps `.png` as the final suffix. After Pillow writes
    the PNG we re-open it, encode as webp, and unlink the PNG so only the
    webp survives in the repo.
    """
    from PIL import Image

    dest_webp.parent.mkdir(parents=True, exist_ok=True)
    png_tmp = dest_webp.with_name(dest_webp.stem + ".tmp.png")
    image.save(path=str(png_tmp), export_json_metadata=False)
    with Image.open(png_tmp) as pil_img:
        pil_img.save(dest_webp, format="WEBP", quality=WEBP_QUALITY, method=WEBP_METHOD)
    png_tmp.unlink()


def _run_worker_vanilla(cfg: VariantConfig, save_to: Path) -> dict[str, Any]:
    import mlx.core as mx

    flux = _load_flux(cfg.loader, cfg.quantize)
    times: list[float] = []
    for i in range(REPS):
        elapsed, image = _generate(
            flux,
            num_inference_steps=cfg.num_inference_steps,
            guidance=cfg.guidance,
            height=cfg.height,
            width=cfg.width,
        )
        times.append(elapsed)
        if i == 0:
            _save_as_webp(image, save_to)
        print(
            f"  vanilla rep {i + 1}: {elapsed:.2f}s (peak {mx.get_peak_memory() / 1024**3:.2f} GB)",
            flush=True,
        )
        del image
        if cfg.clear_cache_between_reps:
            mx.clear_cache()
    return {"condition": "vanilla", "rep_seconds": times, "peak_memory_gb": mx.get_peak_memory() / 1024**3}


def _run_worker_wrapper(cfg: VariantConfig, save_to: Path) -> dict[str, Any]:
    import mlx.core as mx

    from mlx_teacache import apply_teacache

    flux = _load_flux(cfg.loader, cfg.quantize)
    times: list[float] = []
    skipped: list[int] = []
    computed: list[int] = []
    thresh_used: float = 0.0
    for i in range(REPS):
        with apply_teacache(flux) as handle:
            if i == 0:
                thresh_used = handle.rel_l1_thresh
            elapsed, image = _generate(
                flux,
                num_inference_steps=cfg.num_inference_steps,
                guidance=cfg.guidance,
                height=cfg.height,
                width=cfg.width,
            )
            times.append(elapsed)
            skipped.append(handle.stats.skipped_count)
            computed.append(handle.stats.computed_count)
            if i == 0:
                _save_as_webp(image, save_to)
        print(
            f"  wrapper rep {i + 1}: {elapsed:.2f}s (skipped {skipped[-1]}/{cfg.num_inference_steps}, "
            f"peak {mx.get_peak_memory() / 1024**3:.2f} GB)",
            flush=True,
        )
        del image
        if cfg.clear_cache_between_reps:
            mx.clear_cache()
    return {
        "condition": "wrapper",
        "rep_seconds": times,
        "skipped_per_rep": skipped,
        "computed_per_rep": computed,
        "rel_l1_thresh_used": thresh_used,
        "peak_memory_gb": mx.get_peak_memory() / 1024**3,
    }


def _worker_main(args: argparse.Namespace) -> None:
    """Subprocess entrypoint. Runs one (variant, condition) pair and prints
    a single JSON line prefixed by WORKER_RESULT_SENTINEL on stdout."""
    import mlx.core as mx

    cfg = next(v for v in VARIANTS if v.slug == args.variant)
    # Memory guardrail — before any model load. wired must stay strictly below
    # max_recommended_working_set_size (25 GB on M1 Max 32GB) so the worst case
    # is a clean MLX OOM, never a kernel watchdog panic.
    wired = cfg.wired_cap_gb
    mx.set_wired_limit(int(wired * 1024**3))
    mx.set_memory_limit(int((wired + 1) * 1024**3))
    print(
        f"  [worker] {cfg.slug}/{args.condition}: caps wired={wired} GB soft={wired + 1} GB, "
        f"res={cfg.width}x{cfg.height} q{cfg.quantize}",
        flush=True,
    )
    save_to: Path = Path(args.save_to)
    if args.condition == "vanilla":
        result = _run_worker_vanilla(cfg, save_to)
    elif args.condition == "wrapper":
        result = _run_worker_wrapper(cfg, save_to)
    else:
        raise ValueError(f"unknown --condition {args.condition!r}")
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


def _detect_hardware(machine_label_override: str | None, ram_gb_override: int | None) -> dict[str, Any]:
    """Plan-audit Finding 4 fix: hardware provenance recorded in the JSON.

    Reads chip name + RAM via macOS sysctl. CLI flags override whatever
    sysctl reports if the marketing chip name is missing or wrong (e.g.
    a future macOS that doesn't expose `machdep.cpu.brand_string` cleanly)."""
    chip = (
        machine_label_override
        or _macos_sysctl("machdep.cpu.brand_string")
        or platform.processor()
        or "Apple Silicon"
    )
    ram_bytes_str = _macos_sysctl("hw.memsize")
    ram_gb: int | None = ram_gb_override
    if ram_gb is None and ram_bytes_str is not None:
        try:
            ram_gb = round(int(ram_bytes_str) / (1024**3))
        except ValueError:
            ram_gb = None
    return {
        "chip": chip,
        "ram_gb": ram_gb,  # may be None if neither sysctl nor override yielded a value
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "mlx_teacache_version": _mlx_teacache_version(),
        "mflux_version": _mflux_version(),
        "quantize": 4,
        "dtype": "bf16",
    }


def _run_one_worker(slug: str, condition: str, save_to: Path) -> dict[str, Any]:
    """Spawn the worker subprocess and capture its result line."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--variant",
        slug,
        "--condition",
        condition,
        "--save-to",
        str(save_to),
    ]
    print(f"\n>> spawning worker: {slug} / {condition}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    # Stream child stdout/stderr to the orchestrator's stdout so progress is visible.
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed for {slug}/{condition}: exit {proc.returncode}")
    # Find the result line in stdout.
    for line in proc.stdout.splitlines():
        if line.startswith(WORKER_RESULT_SENTINEL):
            return cast(dict[str, Any], json.loads(line[len(WORKER_RESULT_SENTINEL) :]))
    raise RuntimeError(f"worker for {slug}/{condition} did not emit a {WORKER_RESULT_SENTINEL} result line")


def _orchestrate(cfg: VariantConfig, base_dir: Path) -> dict[str, Any]:
    """Run vanilla + wrapper subprocesses for one variant; merge into a JSON entry."""
    variant_dir = base_dir / cfg.slug

    vanilla_path = variant_dir / "vanilla.webp"
    vanilla = _run_one_worker(cfg.slug, "vanilla", vanilla_path)

    wrapper_path = variant_dir / "wrapper.webp"
    wrapper = _run_one_worker(cfg.slug, "wrapper", wrapper_path)

    vanilla_times = vanilla["rep_seconds"]
    wrapper_times = wrapper["rep_seconds"]
    vanilla_cold = vanilla_times[0]
    vanilla_warm = statistics.median(vanilla_times[1:])
    wrapper_cold = wrapper_times[0]
    wrapper_warm = statistics.median(wrapper_times[1:])
    speedup_warm = vanilla_warm / wrapper_warm if wrapper_warm else 0.0
    speedup_cold = vanilla_cold / wrapper_cold if wrapper_cold else 0.0

    print(
        f"  cold: vanilla {vanilla_cold:.2f}s | wrapper {wrapper_cold:.2f}s "
        f"| speedup_cold {speedup_cold:.2f}x"
    )
    print(
        f"  warm: vanilla {vanilla_warm:.2f}s | wrapper {wrapper_warm:.2f}s "
        f"| speedup_warm {speedup_warm:.2f}x"
    )

    return {
        "variant_id": cfg.variant_id,
        "num_inference_steps": cfg.num_inference_steps,
        "guidance": cfg.guidance,
        "height": cfg.height,
        "width": cfg.width,
        "quantize": cfg.quantize,
        "vanilla": {
            "rep_seconds": vanilla_times,
            "cold_seconds": vanilla_cold,
            "warm_median_seconds": vanilla_warm,
            "peak_memory_gb": vanilla.get("peak_memory_gb"),
        },
        "wrapper": {
            "rep_seconds": wrapper_times,
            "cold_seconds": wrapper_cold,
            "warm_median_seconds": wrapper_warm,
            "skipped_per_rep": wrapper["skipped_per_rep"],
            "computed_per_rep": wrapper["computed_per_rep"],
            "rel_l1_thresh_used": wrapper["rel_l1_thresh_used"],
            "peak_memory_gb": wrapper.get("peak_memory_gb"),
        },
        "speedup_warm": speedup_warm,
        "speedup_cold": speedup_cold,
        "image_paths": {
            "vanilla": str(vanilla_path.relative_to(base_dir.parent.parent)),
            "wrapper": str(wrapper_path.relative_to(base_dir.parent.parent)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker",
        action="store_true",
        help="(internal) run as a worker subprocess for one (variant, condition) pair.",
    )
    parser.add_argument("--variant", help="Variant slug (worker mode only).")
    parser.add_argument("--condition", help="vanilla or wrapper (worker mode only).")
    parser.add_argument("--save-to", help="Image destination path (worker mode only).")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).parent.parent / "_artifacts",
        help="Root directory for outputs. Default: <repo>/_artifacts/",
    )
    parser.add_argument(
        "--machine-label",
        default=None,
        help="Override the chip name written into comparison_report.json's hardware section "
        "(e.g. 'Apple M1 Max'). Defaults to macOS sysctl machdep.cpu.brand_string.",
    )
    parser.add_argument(
        "--ram-gb",
        type=int,
        default=None,
        help="Override the RAM-GB field. Defaults to round(hw.memsize / 1 GiB) on macOS.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Restrict orchestration to a single variant slug (e.g. 'klein-base-4b-cfg'). "
        "Useful for resuming after a partial run.",
    )
    args = parser.parse_args()

    if args.worker:
        _worker_main(args)
        return

    base_dir: Path = args.output_root / "comparison"
    base_dir.mkdir(parents=True, exist_ok=True)
    report_path: Path = args.output_root / "comparison_report.json"

    from datetime import datetime

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
        "hardware": _detect_hardware(args.machine_label, args.ram_gb),
        "prompt": PROMPT,
        "seed": SEED,
        "height": HEIGHT,
        "width": WIDTH,
        "reps_per_condition": REPS,
        "isolation": "subprocess-per-condition",
        "variants": {},
    }

    variants_to_run = tuple(v for v in VARIANTS if v.slug == args.only) if args.only else VARIANTS
    if args.only and not variants_to_run:
        raise SystemExit(f"--only {args.only!r} did not match any variant slug")

    if args.only and report_path.exists():
        report = json.loads(report_path.read_text())
        print(f"Resuming from existing report at {report_path}")

    for cfg in variants_to_run:
        print(
            f"\n=== {cfg.variant_id} (slug={cfg.slug}) — "
            f"{cfg.num_inference_steps} steps, guidance={cfg.guidance} ==="
        )
        report["variants"][cfg.slug] = _orchestrate(cfg, base_dir)

    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written: {report_path}")
    for slug, entry in report["variants"].items():
        print(
            f"  {slug:24s} "
            f"vanilla_warm={entry['vanilla']['warm_median_seconds']:.2f}s "
            f"wrapper_warm={entry['wrapper']['warm_median_seconds']:.2f}s "
            f"speedup_warm={entry['speedup_warm']:.2f}x "
            f"skipped[0]={entry['wrapper']['skipped_per_rep'][0]}"
        )


if __name__ == "__main__":
    main()
