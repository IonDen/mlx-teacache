"""Three-way FLUX.2 Klein bench: distilled vs base vs base + mlx-teacache.

Answers the question from mflux issue 113: is Klein **Base + TeaCache** worth
running over the distilled Klein? Three conditions on one prompt and seed:

  - ``distilled``      flux2-klein-{4b,9b}, its default 4 steps, guidance 1.0
  - ``base``           flux2-klein-base-{4b,9b}, 50 steps, guidance 4.0 (real CFG)
  - ``base-teacache``  the base recipe wrapped in ``apply_teacache`` (default threshold)

Run it, and it prints a markdown table you can paste into the thread — wall-clock,
peak memory, and SSIM of each condition against both the distilled and the base
image — plus a JSON report under ``_artifacts/``.

Run as (one model per invocation)::

  uv run python scripts/bench_klein_base_vs_distilled.py --size 4b --quantize 4 --reps 3
  uv run python scripts/bench_klein_base_vs_distilled.py --size 9b --quantize 4 --reps 3

Architecture
------------
Each (condition, rep) runs in a SEPARATE subprocess, so every rep starts from a
cold MLX allocator; rep 1 also reads weights from cold disk, reps 2+ from the OS
page cache (``cold`` = rep 1, ``warm`` = median of reps 2+). The worker prints one
JSON line prefixed by ``WORKER_RESULT_SENTINEL``; the orchestrator persists it as a
chunk the instant it lands and skips chunks whose file already exists, so an
interruption loses at most one rep and ``--reps 3`` resumes where it stopped. Memory
caps come from ``_mlx_caps.install_caps`` and the active+cache watchdog from
``_mlx_watchdog`` — the same discipline every heavy script here uses.

Only a q4 run fits this 32 GB M1 Max; ``--quantize 8`` / ``none`` are for the 64 GB+
Macs the reply invites to run it and paste their table.
"""

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

# Shared portrait recipe — identical to scripts/bench_comparison.py so the images
# line up with the rest of COMPARISON.md. One prompt + seed across every condition.
PROMPT = (
    "Portrait of a young woman with auburn hair and green eyes, soft "
    "golden-hour window light, photorealistic, shallow depth of field, "
    "50mm prime lens, subtle freckles, neutral background, cinematic "
    "color grading."
)
SEED = 42
HEIGHT = 1024
WIDTH = 768
REPS = 3
HEADROOM_GIB = 4.0
DEFAULT_WIRED_CAP_GB = 22  # klein q4 peaks ~22 GB at 9B; install_caps clamps to the device

WORKER_RESULT_SENTINEL = "::BENCH_RESULT::"

CONDITIONS: tuple[str, ...] = ("distilled", "base", "base-teacache")

DISTILLED_STEPS = 4
DISTILLED_GUIDANCE = 1.0
BASE_STEPS = 50
BASE_GUIDANCE = 4.0

WEBP_QUALITY = 88
WEBP_METHOD = 6

_CONDITION_LABEL = {"distilled": "distilled", "base": "base", "base-teacache": "base+TeaCache"}


# ---------------------------------------------------------------------------
# Pure core — recipe map, metrics, chunk resume, SSIM, table. No mlx/mflux here.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ConditionRecipe:
    """How one condition is built and run."""

    condition: str
    model_config_name: str  # a ModelConfig factory name, e.g. "flux2_klein_base_4b"
    num_inference_steps: int
    guidance: float
    use_teacache: bool


def condition_recipe(*, size: str, condition: str) -> ConditionRecipe:
    """The recipe for one (size, condition). ``size`` is ``"4b"`` or ``"9b"``."""
    if size not in ("4b", "9b"):
        raise ValueError(f"unknown size {size!r}; expected '4b' or '9b'")
    if condition == "distilled":
        return ConditionRecipe(
            condition=condition,
            model_config_name=f"flux2_klein_{size}",
            num_inference_steps=DISTILLED_STEPS,
            guidance=DISTILLED_GUIDANCE,
            use_teacache=False,
        )
    if condition in ("base", "base-teacache"):
        return ConditionRecipe(
            condition=condition,
            model_config_name=f"flux2_klein_base_{size}",
            num_inference_steps=BASE_STEPS,
            guidance=BASE_GUIDANCE,
            use_teacache=(condition == "base-teacache"),
        )
    raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")


def parse_quantize(value: str) -> int | None:
    """Map the ``--quantize`` CLI choice to the mflux argument (``none`` -> bf16)."""
    if value == "none":
        return None
    if value in ("4", "8"):
        return int(value)
    raise ValueError(f"unknown quantize {value!r}; expected 'none', '8', or '4'")


def quantize_label(quantize: int | None) -> str:
    return "bf16" if quantize is None else f"q{quantize}"


def chunk_tag(*, size: str, quantize: int | None, height: int, width: int) -> str:
    """Directory/report tag for one model+recipe, e.g. ``4b_q4`` at the default
    768x1024 portrait, or ``9b_q4_512x512`` when the resolution is lowered — so a
    lower-resolution run never overwrites the default one's artifacts."""
    base = f"{size}_{quantize_label(quantize)}"
    if (width, height) == (WIDTH, HEIGHT):
        return base
    return f"{base}_{width}x{height}"


def rep_metrics(rep_seconds: list[float]) -> dict[str, float | None]:
    """Cold = rep 1 (cold disk), warm = median of reps 2+, median = of all reps.

    A one-rep preview has no warm measurement, so ``warm`` is ``None`` rather than
    crashing on ``statistics.median([])``."""
    return {
        "cold": rep_seconds[0],
        "warm": statistics.median(rep_seconds[1:]) if len(rep_seconds) > 1 else None,
        "median": statistics.median(rep_seconds),
    }


def headline_seconds(rep_seconds: list[float]) -> float:
    """The single "one image" wall-clock: steady-state (warm) when we have it,
    else the cold rep."""
    m = rep_metrics(rep_seconds)
    warm = m["warm"]
    return warm if warm is not None else cast(float, m["cold"])


def speedup_vs_base(*, base_seconds: float | None, cond_seconds: float | None) -> float | None:
    """``base / condition`` wall-clock ratio (>1 = faster than base), or ``None``
    when either side is missing or the condition took zero seconds."""
    if base_seconds is None or cond_seconds is None or cond_seconds == 0:
        return None
    return base_seconds / cond_seconds


# --- per-(condition, rep) chunk persistence + resume -------------------------


def chunk_path(results_dir: Path, condition: str, rep: int) -> Path:
    return results_dir / f"{condition}_rep{rep}.json"


def pending_chunks(conditions: list[str], reps: int, results_dir: Path) -> list[tuple[str, int]]:
    """(condition, rep) pairs with no persisted result yet, rep-outer so slow host
    drift over a multi-hour run lands on every condition alike."""
    return [
        (condition, rep)
        for rep in range(reps)
        for condition in conditions
        if not chunk_path(results_dir, condition, rep).exists()
    ]


def persist_chunk(results_dir: Path, result: dict[str, Any]) -> Path:
    """Write one worker result to its chunk file (atomic replace); return the path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = chunk_path(results_dir, str(result["condition"]), int(result["rep"]))
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(dest)
    return dest


def load_chunks(
    conditions: list[str], reps: int, results_dir: Path
) -> dict[str, list[dict[str, Any]]] | None:
    """All persisted results keyed by condition (ordered by rep), or None if any is missing."""
    if pending_chunks(conditions, reps, results_dir):
        return None
    return {
        condition: [
            cast(dict[str, Any], json.loads(chunk_path(results_dir, condition, rep).read_text()))
            for rep in range(reps)
        ]
        for condition in conditions
    }


def verify_chunk_recipes(
    conditions: list[str], reps: int, results_dir: Path, *, quantize: int | None, height: int, width: int
) -> None:
    """Refuse to reuse a persisted chunk measured at a different quantization or
    resolution, so a resume at another ``--quantize`` / ``--height`` / ``--width``
    cannot silently aggregate a mismatched timing under this run's report header."""
    for condition in conditions:
        for r in range(reps):
            path = chunk_path(results_dir, condition, r)
            if not path.exists():
                continue
            chunk = json.loads(path.read_text())
            got_q = chunk.get("quantize")
            if got_q != quantize:
                raise SystemExit(
                    f"persisted chunk {path} was measured with quantize={got_q} but this invocation "
                    f"uses quantize={quantize}; move {results_dir} to the Trash for a fresh measurement"
                )
            got_res = (chunk.get("width"), chunk.get("height"))
            if got_res != (None, None) and got_res != (width, height):
                raise SystemExit(
                    f"persisted chunk {path} was measured at resolution {got_res[0]}x{got_res[1]} but this "
                    f"invocation uses {width}x{height}; move {results_dir} to the Trash for a fresh measurement"
                )


# --- SSIM (perceptual similarity between the condition images) ---------------


def ssim_from_arrays(a: Any, b: Any) -> float:
    """SSIM of two uint8 RGB arrays (skimage, imported lazily)."""
    from skimage.metrics import structural_similarity

    return float(structural_similarity(a, b, channel_axis=-1, data_range=255))


def load_rgb_array(path: Path) -> Any:
    """Load an image as a uint8 RGB numpy array (PIL, imported lazily)."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def ssim_from_files(path_a: Path, path_b: Path) -> float:
    return ssim_from_arrays(load_rgb_array(path_a), load_rgb_array(path_b))


# --- aggregation + the paste-ready table ------------------------------------


def aggregate_condition(rep_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine one condition's per-rep chunks into a report entry."""
    rep_seconds = [float(r["elapsed_s"]) for r in rep_results]
    m = rep_metrics(rep_seconds)
    skipped = [int(r.get("stats_summary", {}).get("skipped_count", 0)) for r in rep_results]
    entry: dict[str, Any] = {
        "rep_seconds": rep_seconds,
        "cold_seconds": m["cold"],
        "warm_seconds": m["warm"],
        "median_seconds": m["median"],
        "headline_seconds": headline_seconds(rep_seconds),
        "peak_memory_gb": max(float(r["peak_memory_gb"]) for r in rep_results),
        "num_inference_steps": int(rep_results[0]["num_inference_steps"]),
    }
    if any(skipped):
        entry["skipped_per_rep"] = skipped
        entry["skipped_median"] = int(statistics.median(skipped))
        entry["rel_l1_thresh_used"] = rep_results[0].get("stats_summary", {}).get("rel_l1_thresh_used")
    return entry


@dataclass(frozen=True, slots=True, kw_only=True)
class TableRow:
    condition: str
    steps: int
    wall_clock_s: float
    speedup_vs_base: float | None
    peak_gb: float | None
    ssim_vs_distilled: float | None
    ssim_vs_base: float | None
    skipped: int | None


def table_rows_from_report(report: dict[str, Any]) -> list[TableRow]:
    """Build the table rows from a finished report: fill each condition's speedup
    against base and its SSIM against the distilled and base images."""
    conditions = report["conditions"]
    ssim = report.get("ssim", {})
    base_seconds = conditions.get("base", {}).get("headline_seconds")
    ssim_vs_distilled = {
        "distilled": 1.0,
        "base": ssim.get("base_vs_distilled"),
        "base-teacache": ssim.get("base_teacache_vs_distilled"),
    }
    ssim_vs_base = {
        "distilled": ssim.get("base_vs_distilled"),
        "base": 1.0,
        "base-teacache": ssim.get("base_teacache_vs_base"),
    }
    rows: list[TableRow] = []
    for cond in CONDITIONS:
        c = conditions.get(cond)
        if c is None:
            continue
        wall = c["headline_seconds"]
        rows.append(
            TableRow(
                condition=cond,
                steps=int(c["num_inference_steps"]),
                wall_clock_s=wall,
                speedup_vs_base=speedup_vs_base(base_seconds=base_seconds, cond_seconds=wall),
                peak_gb=c.get("peak_memory_gb"),
                ssim_vs_distilled=ssim_vs_distilled[cond],
                ssim_vs_base=ssim_vs_base[cond],
                skipped=c.get("skipped_median"),
            )
        )
    return rows


def _fmt_speedup(x: float | None) -> str:
    return f"{x:.2f}×" if x is not None else "—"


def _fmt_ssim(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "—"


def _fmt_steps(row: TableRow) -> str:
    if row.skipped:
        return f"{row.steps} (−{row.skipped} skipped)"
    return str(row.steps)


def render_markdown_table(rows: list[TableRow]) -> str:
    """A GitHub-ready markdown table: one row per condition, columns for
    wall-clock, speedup vs base, peak memory, and SSIM vs the distilled and base
    images."""
    header = "| Condition | Steps | Wall-clock (s) | vs base | Peak (GB) | SSIM vs distilled | SSIM vs base |"
    sep = "| --- | --- | ---: | ---: | ---: | ---: | ---: |"
    lines = [header, sep]
    for row in rows:
        peak = f"{row.peak_gb:.1f}" if row.peak_gb is not None else "—"
        lines.append(
            f"| {_CONDITION_LABEL[row.condition]} | {_fmt_steps(row)} | {row.wall_clock_s:.0f} | "
            f"{_fmt_speedup(row.speedup_vs_base)} | {peak} | "
            f"{_fmt_ssim(row.ssim_vs_distilled)} | {_fmt_ssim(row.ssim_vs_base)} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WORKER side — one (condition, rep) per subprocess.
# ---------------------------------------------------------------------------


def _load_klein(model_config_name: str, quantize: int | None) -> Any:
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    model_config = getattr(ModelConfig, model_config_name)()
    flux = Flux2Klein(quantize=quantize, model_config=model_config)
    flux.freeze()
    return flux


def _generate(
    flux: Any,
    *,
    num_inference_steps: int,
    guidance: float,
    height: int,
    width: int,
    save_path: Path | None,
) -> tuple[float, Any]:
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
    mx.synchronize()  # drain submitted GPU work before the clock stops
    elapsed = time.perf_counter() - start
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path=str(save_path), export_json_metadata=False)
    return elapsed, image


def _worker_main(args: argparse.Namespace) -> None:
    import mlx.core as mx
    from _mlx_caps import install_caps

    size: str = args.size
    condition: str = args.condition
    rep: int = args.rep
    quantize = parse_quantize(args.quantize)
    height: int = args.height
    width: int = args.width
    recipe = condition_recipe(size=size, condition=condition)

    wired_gb: int = args.wired_cap_gb
    wired_b, soft_b, cache_b = install_caps(wired_gb=wired_gb, soft_gb=wired_gb + 1, cache_gb=2.0)
    print(
        f"  [worker] {size}/{condition}/rep{rep}: caps wired={wired_b / 1024**3:.2f} GB "
        f"soft={soft_b / 1024**3:.2f} GB cache={cache_b / 1024**3:.2f} GB, "
        f"{width}x{height} {quantize_label(quantize)}, {recipe.num_inference_steps} steps g={recipe.guidance}",
        flush=True,
    )

    def _on_abort(payload: dict[str, int]) -> None:
        abort = {"aborted": "active-memory watchdog", "size": size, "condition": condition, "rep": rep}
        print(f"{WORKER_RESULT_SENTINEL}{json.dumps({**abort, **payload})}", flush=True)

    arm_mlx_watchdog(on_abort=_on_abort, headroom_gib=args.headroom_gib)
    started_at = datetime.now(timezone.utc).isoformat()
    save_path = Path(args.save_to) if args.save_to else None

    flux = _load_klein(recipe.model_config_name, quantize)
    load_peak = int(mx.get_peak_memory())
    mx.reset_peak_memory()

    stats_summary: dict[str, Any] = {}
    if recipe.use_teacache:
        from mlx_teacache import apply_teacache

        with apply_teacache(flux) as handle:
            elapsed, _ = _generate(
                flux,
                num_inference_steps=recipe.num_inference_steps,
                guidance=recipe.guidance,
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
            f"  {condition} rep {rep + 1}: {elapsed:.2f}s "
            f"(skipped {stats_summary['skipped_count']}/{recipe.num_inference_steps}, "
            f"peak {mx.get_peak_memory() / 1024**3:.2f} GB)",
            flush=True,
        )
    else:
        elapsed, _ = _generate(
            flux,
            num_inference_steps=recipe.num_inference_steps,
            guidance=recipe.guidance,
            height=height,
            width=width,
            save_path=save_path,
        )
        print(
            f"  {condition} rep {rep + 1}: {elapsed:.2f}s (peak {mx.get_peak_memory() / 1024**3:.2f} GB)",
            flush=True,
        )

    loop_peak = int(mx.get_peak_memory())
    result: dict[str, Any] = {
        "size": size,
        "condition": condition,
        "rep": rep,
        "quantize": quantize,
        "height": height,
        "width": width,
        "num_inference_steps": recipe.num_inference_steps,
        "guidance": recipe.guidance,
        "elapsed_s": elapsed,
        "peak_memory_gb": max(load_peak, loop_peak) / 1024**3,
        "load_peak_memory_gb": load_peak / 1024**3,
        "loop_peak_memory_gb": loop_peak / 1024**3,
        "cache_memory_gb": int(mx.get_cache_memory()) / 1024**3,
        "stats_summary": stats_summary,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "provenance": _provenance(),
    }
    print(f"{WORKER_RESULT_SENTINEL}{json.dumps(result)}", flush=True)


# ---------------------------------------------------------------------------
# ORCHESTRATOR side.
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
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "mlx_teacache_version": _mlx_teacache_version(),
        "mflux_version": _mflux_version(),
    }


def _macos_sysctl(key: str) -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _detect_hardware(
    *, quantize: int | None, machine_label: str | None, ram_gb: int | None
) -> dict[str, Any]:
    chip = (
        machine_label or _macos_sysctl("machdep.cpu.brand_string") or platform.processor() or "Apple Silicon"
    )
    ram_bytes = _macos_sysctl("hw.memsize")
    resolved_ram = ram_gb
    if resolved_ram is None and ram_bytes is not None:
        try:
            resolved_ram = round(int(ram_bytes) / (1024**3))
        except ValueError:
            resolved_ram = None
    return {
        "chip": chip,
        "ram_gb": resolved_ram,
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "mlx_teacache_version": _mlx_teacache_version(),
        "mflux_version": _mflux_version(),
        "quantize": quantize_label(quantize),
        "dtype": "bf16",
    }


def _parse_worker_line(stdout: str) -> dict[str, Any] | None:
    """The worker's sentinel-prefixed JSON payload; an abort payload wins over an
    earlier result line (the watchdog can fire during image.save)."""
    found: dict[str, Any] | None = None
    for line in stdout.splitlines():
        if line.startswith(WORKER_RESULT_SENTINEL):
            payload = cast(dict[str, Any], json.loads(line[len(WORKER_RESULT_SENTINEL) :]))
            if "aborted" in payload:
                return payload
            found = payload
    return found


def _image_path_for(images_dir: Path, condition: str, rep: int) -> Path | None:
    """Only rep 0 saves an image (the seed is fixed, so every rep is identical)."""
    if rep != 0:
        return None
    return images_dir / f"{condition}.png"


def _run_one_worker(
    *,
    size: str,
    quantize_str: str,
    condition: str,
    rep: int,
    height: int,
    width: int,
    wired_cap_gb: int,
    headroom_gib: float,
    save_to: Path | None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--size",
        size,
        "--quantize",
        quantize_str,
        "--condition",
        condition,
        "--rep",
        str(rep),
        "--height",
        str(height),
        "--width",
        str(width),
        "--wired-cap-gb",
        str(wired_cap_gb),
        "--headroom-gib",
        str(headroom_gib),
    ]
    if save_to is not None:
        cmd += ["--save-to", str(save_to)]
    print(f"\n>> spawning worker: {size} / {condition} / rep {rep}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    payload = _parse_worker_line(proc.stdout)
    if payload is not None and "aborted" in payload:
        return payload
    if proc.returncode != 0:
        raise RuntimeError(f"worker failed for {size}/{condition}/rep{rep}: exit {proc.returncode}")
    if payload is None:
        raise RuntimeError(
            f"worker for {size}/{condition}/rep{rep} did not emit a {WORKER_RESULT_SENTINEL} result line"
        )
    return payload


def _condition_image(images_dir: Path, condition: str) -> Path | None:
    """The saved image for a condition: the lossless PNG if it survives, else the
    committed webp (a resume after webp conversion)."""
    for suffix in (".png", ".webp"):
        candidate = images_dir / f"{condition}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _compute_ssim(images_dir: Path) -> dict[str, float]:
    """SSIM of base and base-teacache against distilled, and base-teacache against
    base — whatever images are present. Silently skips a pair whose image is missing."""
    imgs = {cond: _condition_image(images_dir, cond) for cond in CONDITIONS}
    ssim: dict[str, float] = {}
    d, b, t = imgs["distilled"], imgs["base"], imgs["base-teacache"]
    if b and d:
        ssim["base_vs_distilled"] = ssim_from_files(b, d)
    if t and d:
        ssim["base_teacache_vs_distilled"] = ssim_from_files(t, d)
    if t and b:
        ssim["base_teacache_vs_base"] = ssim_from_files(t, b)
    return ssim


def _convert_pngs_to_webp(images_dir: Path) -> None:
    """Re-encode each condition's PNG as webp (the committed format) and drop the
    PNG. SSIM is already computed from the PNGs before this runs."""
    from PIL import Image

    for cond in CONDITIONS:
        png = images_dir / f"{cond}.png"
        if not png.exists():
            continue
        webp = images_dir / f"{cond}.webp"
        with Image.open(png) as img:
            img.save(webp, format="WEBP", quality=WEBP_QUALITY, method=WEBP_METHOD)
        png.unlink()


def _build_report(
    *,
    size: str,
    quantize: int | None,
    reps: int,
    conditions: list[str],
    loaded: dict[str, list[dict[str, Any]]],
    ssim: dict[str, float],
    images_dir: Path,
    hardware: dict[str, Any],
    height: int,
    width: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "size": size,
        "quantize": quantize_label(quantize),
        "prompt": PROMPT,
        "seed": SEED,
        "height": height,
        "width": width,
        "reps": reps,
        "isolation": "subprocess-per-(condition,rep)",
        "hardware": hardware,
        "conditions": {cond: aggregate_condition(loaded[cond]) for cond in conditions},
        "ssim": ssim,
        "images_dir": str(images_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="(internal) run one (condition, rep).")
    parser.add_argument("--size", choices=["4b", "9b"], required=True, help="Klein model size.")
    parser.add_argument(
        "--quantize", choices=["none", "8", "4"], default="4", help="Quantization (default q4)."
    )
    parser.add_argument("--reps", type=int, default=REPS, help="Reps per condition (default 3).")
    parser.add_argument("--condition", choices=list(CONDITIONS), help="(worker mode) the condition.")
    parser.add_argument("--rep", type=int, default=0, help="(worker mode) the rep index.")
    parser.add_argument("--save-to", default=None, help="(worker mode) image destination path.")
    parser.add_argument(
        "--height", type=int, default=HEIGHT, help=f"Image height (default {HEIGHT}; lower for memory/power)."
    )
    parser.add_argument(
        "--width", type=int, default=WIDTH, help=f"Image width (default {WIDTH}; lower for memory/power)."
    )
    parser.add_argument(
        "--only", choices=list(CONDITIONS), default=None, help="Restrict to one condition (resume)."
    )
    parser.add_argument("--wired-cap-gb", type=int, default=DEFAULT_WIRED_CAP_GB, dest="wired_cap_gb")
    parser.add_argument("--headroom-gib", type=float, default=HEADROOM_GIB, dest="headroom_gib")
    parser.add_argument("--machine-label", default=None, dest="machine_label")
    parser.add_argument("--ram-gb", type=int, default=None, dest="ram_gb")
    parser.add_argument(
        "--max-chunks", type=int, default=None, dest="max_chunks", help="Spawn at most N workers, then exit."
    )
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--report", type=Path, default=None, help="Report JSON path (default under _artifacts/)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "tests" / "_artifacts" / "klein_base_vs_distilled_chunks",
        dest="results_dir",
        help="Per-(condition, rep) chunk directory (git-ignored). Existing chunks are REUSED.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=repo_root / "_artifacts" / "klein_base_vs_distilled",
        dest="images_dir",
    )
    args = parser.parse_args()

    if args.worker:
        _worker_main(args)
        return

    size: str = args.size
    quantize = parse_quantize(args.quantize)
    reps: int = args.reps
    height: int = args.height
    width: int = args.width
    tag = chunk_tag(size=size, quantize=quantize, height=height, width=width)
    conditions = [args.only] if args.only else list(CONDITIONS)

    results_dir: Path = args.results_dir / tag
    images_dir: Path = args.images_dir / tag
    report_path: Path = args.report or (repo_root / "_artifacts" / f"klein_base_vs_distilled_{tag}.json")

    verify_chunk_recipes(conditions, reps, results_dir, quantize=quantize, height=height, width=width)
    pending = pending_chunks(conditions, reps, results_dir)
    total = len(conditions) * reps
    if len(pending) < total:
        print(f"\n== RESUMING: {total - len(pending)}/{total} chunks already on disk under {results_dir} ==")
    to_run = pending if args.max_chunks is None else pending[: args.max_chunks]
    print(f"\n== running {len(to_run)} of {len(pending)} pending chunks (size={size} {tag}, reps={reps}) ==")

    for condition, rep in to_run:
        result = _run_one_worker(
            size=size,
            quantize_str=args.quantize,
            condition=condition,
            rep=rep,
            height=height,
            width=width,
            wired_cap_gb=args.wired_cap_gb,
            headroom_gib=args.headroom_gib,
            save_to=_image_path_for(images_dir, condition, rep),
        )
        if "aborted" in result:
            results_dir.mkdir(parents=True, exist_ok=True)
            aborted = chunk_path(results_dir, condition, rep).with_suffix(".aborted.json")
            aborted.write_text(json.dumps(result, indent=2))
            print(
                f"\n== ABORTED by the memory watchdog on {size}/{condition}/rep{rep}: "
                f"{result['resident_bytes'] / 1024**3:.2f} GB resident > "
                f"{result['ceiling_bytes'] / 1024**3:.2f} GB ceiling; artifact {aborted}. "
                "Nothing persisted; lower the recipe or caps before re-invoking. ==",
                flush=True,
            )
            raise SystemExit(4)
        written = persist_chunk(results_dir, result)
        print(f">> chunk persisted: {written}", flush=True)

    loaded = load_chunks(conditions, reps, results_dir)
    if loaded is None:
        remaining = pending_chunks(conditions, reps, results_dir)
        print(
            f"\n== PARTIAL: {total - len(remaining)}/{total} chunks persisted under {results_dir}; "
            f"{len(remaining)} pending — re-invoke to continue. No report written. =="
        )
        raise SystemExit(3)

    full_run = set(conditions) == set(CONDITIONS)
    ssim = _compute_ssim(images_dir) if full_run else {}
    if full_run:
        _convert_pngs_to_webp(images_dir)
    report = _build_report(
        size=size,
        quantize=quantize,
        reps=reps,
        conditions=conditions,
        loaded=loaded,
        ssim=ssim,
        images_dir=images_dir,
        hardware=_detect_hardware(quantize=quantize, machine_label=args.machine_label, ram_gb=args.ram_gb),
        height=height,
        width=width,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    print("\n== Klein: distilled vs base vs base + TeaCache ==")
    print(f"  size={size} {tag}, {width}x{height}, seed {SEED}, reps={reps}")
    print(f"  report: {report_path}")
    if not full_run:
        print(f"  (single-condition run --only {args.only}; SSIM/table need all three conditions)")
        return
    print()
    print(render_markdown_table(table_rows_from_report(report)))


if __name__ == "__main__":
    main()
