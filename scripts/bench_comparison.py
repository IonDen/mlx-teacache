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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from _bench_telemetry import streak_telemetry as _streak_telemetry
from _mlx_watchdog import arm_mlx_watchdog

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
HEADROOM_GIB = 4.0  # watchdog headroom; overridden by --headroom-gib in main()

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
    # Construction note recorded in the report so a reader can tell HOW the model
    # was built (e.g. mixed-precision), since `quantize` alone is misleading for a
    # variant that overrides it. "" means uniform q{quantize}.
    build: str = ""
    wired_cap_gb: int = (
        22  # mx.set_wired_limit; must stay < max_recommended_working_set_size (25 on M1 Max 32GB)
    )
    # Free the MLX buffer cache between reps. Off for the q4 FLUX rows (their warm
    # reps intentionally reuse the warm allocator). On for q8 Z-Image at 640x896,
    # where a single gen peaks ~18.7 GB but the cache accumulates across reps in
    # one process and OOMs the Metal command buffer on rep 2 without this.
    clear_cache_between_reps: bool = False
    # Soft memory limit (mx.set_memory_limit) in GB. 0 = use wired_cap_gb + 1 (the
    # default for variants whose peak fits under wired+1). The 20B Qwen row sets
    # this above its ~27.6 GB peak so the advisory soft limit doesn't force
    # mid-generation cache eviction that would inflate the timing; the wired cap
    # still bounds the panic-causing wired memory.
    soft_cap_gb: int = 0


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
    VariantConfig(
        slug="qwen-image",
        variant_id="qwen-image",
        num_inference_steps=50,
        guidance=4.0,
        loader="qwen-image",
        # 20B Qwen-Image at the mixed-precision build (q8 edge blocks + bf16 embeddings,
        # for showcase quality) peaks ~30.4 GB at 768x768 on a 32 GB M1 Max — it fits
        # (the wired cap bounds non-pageable memory; the excess is pageable). 768x768
        # / 50 steps is the official Qwen recipe; same PROMPT + SEED as every other
        # row, only the resolution (and incidentally the aspect) changes, per the
        # COMPARISON shared-prompt rule.
        height=768,
        width=768,
        quantize=4,
        build="mixed-precision: q8 first/last-6 transformer blocks + bf16 embeddings/projection (quantize=4 base)",
        wired_cap_gb=21,  # device-derived ~0.85*24.96; bounds wired memory (peak ~30.4 GB total is pageable)
        # Advisory only. The watchdog now bounds the run at memory_size - 4 GiB (28 GiB on
        # this machine), so the mixed-precision build's ~30.4 GB peak will trip it: this row
        # cannot run on a 32 GB Mac until its build or resolution is lowered.
        soft_cap_gb=31,
        clear_cache_between_reps=True,  # 20B near the 32 GB edge; cache accumulation OOMs reps without this
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
    elif loader == "qwen-image":
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
        from qwen_mixed_precision import enable_qwen_mixed_precision

        # Showcase quality: mixed-precision (q8 edge blocks + bf16 embeddings) clears
        # the uniform-q4 grain so the COMPARISON portraits look good. mlx-teacache
        # stays quant-agnostic — this is a construction-time choice in the bench only.
        enable_qwen_mixed_precision()
        flux = QwenImage(quantize=quantize, model_config=ModelConfig.qwen_image())
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
    mx.synchronize()  # drain submitted GPU work; generate_image already returned a host-side image
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
    skip_patterns: list[str] = []
    max_streaks: list[int] = []
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
            telemetry = _streak_telemetry(handle.stats)
            skip_patterns.append(telemetry["skip_pattern"])
            max_streaks.append(telemetry["max_consecutive_skips"])
            if i == 0:
                _save_as_webp(image, save_to)
        print(
            f"  wrapper rep {i + 1}: {elapsed:.2f}s (skipped {skipped[-1]}/{cfg.num_inference_steps}, "
            f"max streak {max_streaks[-1]}, peak {mx.get_peak_memory() / 1024**3:.2f} GB)",
            flush=True,
        )
        del image
        del handle  # release the rep's cached residuals before clearing MLX's cache pool
        if cfg.clear_cache_between_reps:
            mx.clear_cache()
    return {
        "condition": "wrapper",
        "rep_seconds": times,
        "skipped_per_rep": skipped,
        "computed_per_rep": computed,
        "skip_pattern_per_rep": skip_patterns,
        "max_consecutive_skips_per_rep": max_streaks,
        "rel_l1_thresh_used": thresh_used,
        "peak_memory_gb": mx.get_peak_memory() / 1024**3,
    }


def _worker_main(args: argparse.Namespace) -> None:
    """Subprocess entrypoint. Runs one (variant, condition) pair and prints
    a single JSON line prefixed by WORKER_RESULT_SENTINEL on stdout."""
    cfg = next(v for v in VARIANTS if v.slug == args.variant)
    # Memory guardrail — before any model load. wired must stay strictly below
    # max_recommended_working_set_size (25 GB on M1 Max 32GB) so the worst case
    # is a clean MLX OOM, never a kernel watchdog panic.
    wired = cfg.wired_cap_gb
    soft = cfg.soft_cap_gb or (wired + 1)
    from _mlx_caps import install_caps

    cache_gb = 1.0 if cfg.slug == "qwen-image" else 2.0  # qwen's active peak leaves ~1 GiB under the ceiling
    wired_b, soft_b, cache_b = install_caps(wired_gb=wired, soft_gb=soft, cache_gb=cache_gb)
    print(
        f"  [worker] {cfg.slug}/{args.condition}: caps wired={wired_b / 1024**3:.2f} GB "
        f"soft={soft_b / 1024**3:.2f} GB cache={cache_b / 1024**3:.2f} GB, "
        f"res={cfg.width}x{cfg.height} q{cfg.quantize}",
        flush=True,
    )

    def _on_abort(payload: dict[str, int]) -> None:
        abort = {"aborted": "active-memory watchdog", "slug": cfg.slug, "condition": args.condition}
        print(f"{WORKER_RESULT_SENTINEL}{json.dumps({**abort, **payload})}", flush=True)

    arm_mlx_watchdog(on_abort=_on_abort, headroom_gib=HEADROOM_GIB)
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


def _provenance() -> dict[str, str]:
    """Per-run provenance stamped into each variant entry and the top-level on
    every write. ``datetime.now`` is the only impure part; the merge that consumes
    this (``_merge_variant_into_report``) is pure and unit-tested."""
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "mlx_teacache_version": _mlx_teacache_version(),
        "mflux_version": _mflux_version(),
    }


def _merge_variant_into_report(
    report: dict[str, Any], slug: str, entry: dict[str, Any], provenance: dict[str, str]
) -> dict[str, Any]:
    """Insert/replace one variant entry (stamped with its own provenance) and
    refresh the report's top-level ``generated_at`` + hardware software-versions to
    THIS run.

    The report is assembled incrementally — one ``--only <slug>`` run per variant,
    often across different mlx-teacache versions — so no single top-level value can
    honestly describe every row. Per-variant ``provenance`` is authoritative for
    its row; the top-level reflects the most recent write. Without this, a
    ``--only`` resume reloaded a prior report and overwrote only the variant row,
    keeping the earlier run's stale ``generated_at`` / version at the top level.
    Pure: returns a new dict, never mutates the input."""
    out = dict(report)
    out["generated_at"] = provenance["generated_at"]
    if out.get("hardware"):
        out["hardware"] = {
            **out["hardware"],
            "mlx_teacache_version": provenance["mlx_teacache_version"],
            "mflux_version": provenance["mflux_version"],
        }
    out["variants"] = {**out.get("variants", {}), slug: {**entry, "provenance": dict(provenance)}}
    return out


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
        "--reps",
        str(REPS),  # orchestrator's REPS (possibly overridden by --reps) -> worker
        "--headroom-gib",
        str(HEADROOM_GIB),
    ]
    print(f"\n>> spawning worker: {slug} / {condition}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    # Stream child stdout/stderr to the orchestrator's stdout so progress is visible.
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    payload = _parse_worker_line(proc.stdout)
    if payload is not None and "aborted" in payload:
        return payload  # the caller persists the abort artifact and stops
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed for {slug}/{condition}: exit {proc.returncode}")
    if payload is None:
        raise RuntimeError(
            f"worker for {slug}/{condition} did not emit a {WORKER_RESULT_SENTINEL} result line"
        )
    return payload


def _parse_worker_line(stdout: str) -> dict[str, Any] | None:
    """The worker's sentinel-prefixed JSON payload, or None if it never printed one.
    An abort payload wins over an earlier result line: the watchdog can fire after
    the result was printed (during image.save), and that run must not count."""
    found: dict[str, Any] | None = None
    for line in stdout.splitlines():
        if line.startswith(WORKER_RESULT_SENTINEL):
            payload = cast(dict[str, Any], json.loads(line[len(WORKER_RESULT_SENTINEL) :]))
            if "aborted" in payload:
                return payload
            found = payload
    return found


def _condition_metrics(rep_seconds: list[float]) -> dict[str, float | None]:
    """Cold = rep 1 (the subprocess just started); warm = median of reps 2+.

    A one-rep images-only preview (``--reps 1``) has no warm measurement, so warm
    is ``None`` rather than crashing on ``statistics.median([])``. Pure."""
    return {
        "cold": rep_seconds[0],
        "warm": statistics.median(rep_seconds[1:]) if len(rep_seconds) > 1 else None,
    }


def _speedup(vanilla: float | None, wrapper: float | None) -> float | None:
    """vanilla/wrapper wall-clock ratio, or ``None`` when either side is missing
    (a one-rep preview has no warm timing) or the denominator is zero. Pure."""
    if vanilla is None or wrapper is None or wrapper == 0:
        return None
    return vanilla / wrapper


def _fmt_speedup(x: float | None) -> str:
    return f"{x:.2f}x" if x is not None else "n/a"


# --- Per-condition chunk persistence + resume --------------------------------
# One (variant, condition) worker = one chunk. The worker result is written to
# disk the moment the worker returns and a re-invocation reuses persisted
# chunks, so a variant's ~1-2 h showcase run can be split into two finite jobs
# (--max-workers 1) and an interruption loses at most the in-flight worker.

_CONDITIONS: tuple[str, ...] = ("vanilla", "wrapper")


def _chunk_path(chunks_dir: Path, slug: str, condition: str) -> Path:
    return chunks_dir / slug / f"{condition}.json"


def _pending_conditions(chunks_dir: Path, slug: str) -> list[str]:
    """Conditions of one variant with no persisted worker result yet, in run order."""
    return [c for c in _CONDITIONS if not _chunk_path(chunks_dir, slug, c).exists()]


def _persist_chunk(chunks_dir: Path, slug: str, result: dict[str, Any]) -> Path:
    """Write one worker result to its chunk file (atomic replace); return the path."""
    dest = _chunk_path(chunks_dir, slug, str(result["condition"]))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(dest)
    return dest


_PROVENANCE_KEYS: tuple[str, ...] = ("mlx_teacache_version", "mflux_version")


def _load_chunks(chunks_dir: Path, slug: str, *, reps: int) -> dict[str, dict[str, Any]] | None:
    """Both persisted worker results for one variant, or None while either is missing.

    Refuses (SystemExit) to pair chunks that were not measured under one setup:
    each chunk must be stamped with ``reps`` equal to this invocation's and with a
    ``provenance`` whose package versions match across the pair — otherwise a row
    could silently combine a vanilla timing from one release with a wrapper timing
    from another under a single provenance stamp."""
    if _pending_conditions(chunks_dir, slug):
        return None
    loaded = {
        c: cast(dict[str, Any], json.loads(_chunk_path(chunks_dir, slug, c).read_text())) for c in _CONDITIONS
    }
    hint = f"move {chunks_dir / slug} to the Trash for a fresh measurement"
    for c, chunk in loaded.items():
        path = _chunk_path(chunks_dir, slug, c)
        if "provenance" not in chunk or "reps" not in chunk:
            raise SystemExit(f"persisted chunk {path} has no reps/provenance stamp; {hint}")
        if int(chunk["reps"]) != reps:
            raise SystemExit(
                f"persisted chunk {path} was measured with reps={chunk['reps']} but this "
                f"invocation uses reps={reps}; {hint}"
            )
    for key in _PROVENANCE_KEYS:
        values = {c: loaded[c]["provenance"].get(key) for c in _CONDITIONS}
        if len(set(values.values())) != 1:
            raise SystemExit(
                f"{slug}: vanilla/wrapper chunks were measured on different {key} "
                f"({values['vanilla']} vs {values['wrapper']}); {hint}"
            )
    return loaded


def _orchestrate(
    cfg: VariantConfig,
    base_dir: Path,
    *,
    chunks_dir: Path,
    worker_budget: list[int],
    provenance: dict[str, str],
) -> dict[str, Any] | None:
    """Run the pending vanilla / wrapper subprocesses for one variant (within the
    remaining worker budget), persisting each result; return the merged JSON entry
    once both conditions are on disk, else None (partial — re-invoke to continue).

    ``worker_budget`` is a one-element list holding the number of workers this
    invocation may still spawn (a negative value means unlimited); it is decremented
    in place so the budget spans variants."""
    variant_dir = base_dir / cfg.slug
    vanilla_path = variant_dir / "vanilla.webp"
    wrapper_path = variant_dir / "wrapper.webp"
    image_for = {"vanilla": vanilla_path, "wrapper": wrapper_path}

    pending = _pending_conditions(chunks_dir, cfg.slug)
    reused = [c for c in _CONDITIONS if c not in pending]
    if reused:
        print(f"  RESUMING: reusing persisted {reused} from {chunks_dir / cfg.slug}")
    for condition in pending:
        if worker_budget[0] == 0:
            print(f"  worker budget exhausted before {cfg.slug}/{condition}; re-invoke to continue")
            break
        result = _run_one_worker(cfg.slug, condition, image_for[condition])
        if "aborted" in result:
            aborted = _chunk_path(chunks_dir, cfg.slug, condition).with_suffix(".aborted.json")
            aborted.parent.mkdir(parents=True, exist_ok=True)
            aborted.write_text(json.dumps(result, indent=2))
            print(
                f"\n== ABORTED by the memory watchdog on {cfg.slug}/{condition}: "
                f"{result['resident_bytes'] / 1024**3:.2f} GB resident > "
                f"{result['ceiling_bytes'] / 1024**3:.2f} GB ceiling; artifact {aborted}. "
                "Nothing persisted as a result. ==",
                flush=True,
            )
            raise SystemExit(4)
        # Stamp the chunk with this run's reps + provenance so a later invocation can
        # refuse to pair it with a chunk from a different setup.
        written = _persist_chunk(chunks_dir, cfg.slug, {**result, "reps": REPS, "provenance": provenance})
        print(f"  chunk persisted: {written}", flush=True)
        if worker_budget[0] > 0:
            worker_budget[0] -= 1

    loaded = _load_chunks(chunks_dir, cfg.slug, reps=REPS)
    if loaded is None:
        print(f"  PARTIAL: {cfg.slug} still pending {_pending_conditions(chunks_dir, cfg.slug)}; not merged")
        return None
    vanilla, wrapper = loaded["vanilla"], loaded["wrapper"]

    v = _condition_metrics(vanilla["rep_seconds"])
    w = _condition_metrics(wrapper["rep_seconds"])
    speedup_warm = _speedup(v["warm"], w["warm"])
    speedup_cold = _speedup(v["cold"], w["cold"])

    print(
        f"  cold: vanilla {v['cold']:.2f}s | wrapper {w['cold']:.2f}s "
        f"| speedup_cold {_fmt_speedup(speedup_cold)}"
    )
    if v["warm"] is not None and w["warm"] is not None:
        print(
            f"  warm: vanilla {v['warm']:.2f}s | wrapper {w['warm']:.2f}s "
            f"| speedup_warm {_fmt_speedup(speedup_warm)}"
        )
    else:
        print("  warm: (images-only preview — --reps 1, no warm timing)")

    return {
        "variant_id": cfg.variant_id,
        "num_inference_steps": cfg.num_inference_steps,
        "guidance": cfg.guidance,
        "height": cfg.height,
        "width": cfg.width,
        "quantize": cfg.quantize,
        "build": cfg.build or f"uniform q{cfg.quantize}",
        "vanilla": {
            "rep_seconds": vanilla["rep_seconds"],
            "cold_seconds": v["cold"],
            "warm_median_seconds": v["warm"],
            "peak_memory_gb": vanilla.get("peak_memory_gb"),
        },
        "wrapper": {
            "rep_seconds": wrapper["rep_seconds"],
            "cold_seconds": w["cold"],
            "warm_median_seconds": w["warm"],
            "skipped_per_rep": wrapper["skipped_per_rep"],
            "computed_per_rep": wrapper["computed_per_rep"],
            "skip_pattern_per_rep": wrapper.get("skip_pattern_per_rep", []),
            "max_consecutive_skips_per_rep": wrapper.get("max_consecutive_skips_per_rep", []),
            "rel_l1_thresh_used": wrapper["rel_l1_thresh_used"],
            "peak_memory_gb": wrapper.get("peak_memory_gb"),
        },
        "speedup_warm": speedup_warm,
        "speedup_cold": speedup_cold,
        "chunk_provenance": {c: loaded[c]["provenance"] for c in _CONDITIONS},
        "image_paths": {
            "vanilla": str(vanilla_path.relative_to(base_dir.parent.parent)),
            "wrapper": str(wrapper_path.relative_to(base_dir.parent.parent)),
        },
    }


def main() -> None:
    global REPS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker",
        action="store_true",
        help="(internal) run as a worker subprocess for one (variant, condition) pair.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=None,
        help="Override reps per condition (e.g. 1 for an images-only preview; default 3 for timing).",
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
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path(__file__).parent.parent / "tests" / "_artifacts" / "comparison_chunks",
        dest="chunks_dir",
        help=(
            "Directory for per-(variant, condition) worker results (git-ignored). Existing "
            "chunks are REUSED on re-invocation — move a variant's subdir to the Trash for a "
            "fresh measurement."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        dest="max_workers",
        help=(
            "Spawn at most this many worker subprocesses this invocation (one worker = one "
            "(variant, condition), all reps inside it), then exit; a variant is merged into "
            "the report only once both of its conditions are on disk. --max-workers 1 splits "
            "each variant's showcase run into two finite jobs; an invocation that completes "
            "no variant exits with status 3."
        ),
    )
    args = parser.parse_args()
    if args.reps is not None:  # applies to both orchestrator (report) and worker subprocess
        REPS = args.reps

    if args.worker:
        _worker_main(args)
        return

    base_dir: Path = args.output_root / "comparison"
    base_dir.mkdir(parents=True, exist_ok=True)
    report_path: Path = args.output_root / "comparison_report.json"

    provenance = _provenance()
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": provenance["generated_at"],
        "hardware": _detect_hardware(args.machine_label, args.ram_gb),
        "prompt": PROMPT,
        "seed": SEED,
        "height": HEIGHT,
        "width": WIDTH,
        "reps_per_condition": REPS,
        "isolation": "subprocess-per-condition",
        # Assembled incrementally (one --only run per variant, across versions);
        # each variant's "provenance" is authoritative, the top-level reflects the
        # latest write.
        "variants": {},
    }

    variants_to_run = tuple(v for v in VARIANTS if v.slug == args.only) if args.only else VARIANTS
    if args.only and not variants_to_run:
        raise SystemExit(f"--only {args.only!r} did not match any variant slug")

    if report_path.exists():
        # Always merge into the existing report: each completed variant overwrites
        # only its own row, so a --max-workers / --only run that finishes a subset
        # of variants can never drop the others. Move the report to the Trash for
        # a from-scratch file.
        report = json.loads(report_path.read_text())
        print(f"Merging into existing report at {report_path}")

    worker_budget = [args.max_workers if args.max_workers is not None else -1]
    merged_any = False
    for cfg in variants_to_run:
        print(
            f"\n=== {cfg.variant_id} (slug={cfg.slug}) — "
            f"{cfg.num_inference_steps} steps, guidance={cfg.guidance} ==="
        )
        entry = _orchestrate(
            cfg, base_dir, chunks_dir=args.chunks_dir, worker_budget=worker_budget, provenance=provenance
        )
        if entry is None:
            continue
        report = _merge_variant_into_report(report, cfg.slug, entry, provenance)
        merged_any = True

    if not merged_any:
        print(f"\nNo variant completed this invocation; report at {report_path} left untouched.")
        sys.exit(3)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport written: {report_path}")
    for slug, entry in report["variants"].items():
        warm = entry["wrapper"]["warm_median_seconds"]
        warm_str = (
            f"wrapper_warm={warm:.2f}s speedup_warm={_fmt_speedup(entry['speedup_warm'])}"
            if warm is not None
            else "(images-only preview)"
        )
        print(f"  {slug:24s} {warm_str} skipped[0]={entry['wrapper']['skipped_per_rep'][0]}")


if __name__ == "__main__":
    main()
