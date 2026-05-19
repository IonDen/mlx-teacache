"""Wall-clock benchmark: vanilla mflux vs. mlx-teacache wrapper.

Produces the speedup numbers cited in README.md's "Benchmarks" table. Pins
seed, prompt, image dimensions, step count, and dtype so the result is
reproducible across runs (within the bounds of Metal / thermal noise on the
measuring host).

For each --variant:
  1. Load the model at the same quantization README claims for (q4).
  2. Generate one warmup image (vanilla) to seed any one-time compile work.
  3. Time 3 vanilla generations + 3 wrapper generations (alternating reduces
     thermal bias across the two conditions but isn't worth the extra code).
  4. Report the median per condition + ratio + per-rep skip counts so the
     wall-clock improvement can be attributed to step-skipping or to
     other causes (e.g. avoiding mflux's mx.compile).

Run as:
  uv run python scripts/bench_speedup.py --variant klein-9b
  uv run python scripts/bench_speedup.py --variant klein-4b
  uv run python scripts/bench_speedup.py --variant klein-base-4b   # 50-step, g=4.0 (v0.4.1+)
  uv run python scripts/bench_speedup.py --variant klein-base-4b --guidance 1.0 --num-inference-steps 25  # v0.4.0 row
  uv run python scripts/bench_speedup.py --variant klein-base-9b   # 50-step, g=4.0 (v0.5.0)
  uv run python scripts/bench_speedup.py --variant flux1-dev

Three-way mode (--three-way, default on klein-base-4b and klein-base-9b) additionally runs a
wrapped-no-gate condition (rel_l1_thresh=0.0) to separate the v0.4
compile-avoidance effect from the v0.4.1 gating effect:
  A = vanilla                (no wrapper)
  B = wrapped, no gate       (rel_l1_thresh=0.0 — compile-avoidance only)
  C = wrapped, gated         (default threshold — full TeaCache)
  A/B = compile-avoidance contribution  [v0.4 effect]
  B/C = gating contribution             [v0.4.1 effect]

Output: prints summary to stdout, optionally writes JSON via --report.

The script also generates and saves a single vanilla + wrapper image pair
under tests/_artifacts/bench_images/<variant>/ for visual quality comparison.
The bench_images/ directory is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_teacache import apply_teacache

PROMPT = "a red apple on a wooden table"
SEED = 42
HEIGHT = 512
WIDTH = 512


def _load_flux2_klein(variant: str) -> Any:
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    if variant == "klein-4b":
        cfg = ModelConfig.flux2_klein_4b()
    elif variant == "klein-9b":
        cfg = ModelConfig.flux2_klein_9b()
    elif variant == "klein-base-4b":
        cfg = ModelConfig.flux2_klein_base_4b()
    elif variant == "klein-base-9b":
        cfg = ModelConfig.flux2_klein_base_9b()
    else:
        raise ValueError(f"unsupported klein variant: {variant!r}")
    flux = Flux2Klein(quantize=4, model_config=cfg)
    flux.freeze()
    return flux


def _load_flux1(variant: str) -> Any:
    from mflux.models.flux.variants.txt2img.flux import Flux1

    if variant == "flux1-dev":
        flux = Flux1.from_name("dev", quantize=4)
    elif variant == "flux1-schnell":
        flux = Flux1.from_name("schnell", quantize=4)
    else:
        raise ValueError(f"unsupported flux1 variant: {variant!r}")
    flux.freeze()
    return flux


def _variant_config(variant: str) -> dict[str, Any]:
    if variant in ("klein-4b", "klein-9b"):
        return {
            "loader": _load_flux2_klein,
            "num_inference_steps": 8,
            "guidance": 1.0,
        }
    if variant == "klein-base-4b":
        # Canonical upstream recipe (v0.4.1+). Override with
        # --guidance 1.0 --num-inference-steps 25 to reproduce the v0.4.0 row.
        return {
            "loader": _load_flux2_klein,
            "num_inference_steps": 50,
            "guidance": 4.0,
        }
    if variant == "klein-base-9b":
        # Canonical upstream recipe — same as base-4b. Coefficients reused
        # from base-4b verbatim (see src/mlx_teacache/coefficients.py).
        return {
            "loader": _load_flux2_klein,
            "num_inference_steps": 50,
            "guidance": 4.0,
        }
    if variant == "flux1-dev":
        return {
            "loader": _load_flux1,
            "num_inference_steps": 25,
            "guidance": 3.5,
        }
    if variant == "flux1-schnell":
        return {
            "loader": _load_flux1,
            "num_inference_steps": 4,
            "guidance": 1.0,
        }
    raise ValueError(f"unknown variant: {variant!r}")


def _generate(
    flux: Any,
    *,
    num_inference_steps: int,
    guidance: float,
    save_path: Path | None = None,
) -> tuple[float, Any]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "klein-4b",
            "klein-9b",
            "klein-base-4b",
            "klein-base-9b",
            "flux1-dev",
            "flux1-schnell",
        ],
    )
    parser.add_argument("--reps", type=int, default=3, help="timed reps per condition")
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
        help="Override the variant's default step count.",
    )
    parser.add_argument(
        "--three-way",
        action="store_true",
        default=None,
        help=(
            "Run vanilla + wrapped-no-gate + wrapped-gated conditions. Default True on "
            "klein-base-4b only. NOT default on klein-base-9b: the same-process three-way "
            "path runs 9 generations on a single flux instance, and a previous unguarded "
            "9B run hit system-level OOM on 32 GB. Pass --three-way explicitly to opt in "
            "on 9B, but the subprocess-per-rep refactor (v0.5.1) is the safe path."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional JSON output path for the summary",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(__file__).parent.parent / "tests" / "_artifacts" / "bench_images",
        help="root directory for saved bench images (one subdir per variant)",
    )
    parser.add_argument(
        "--mlx-memory-cap-gb",
        type=int,
        default=None,
        help=(
            "Soft MLX memory cap (mx.set_memory_limit). The worker also sets a HARD "
            "wired-memory cap via mx.set_wired_limit at (cap - 2) GB BEFORE the model "
            "loads. The wired cap is what actually prevents kernel panics (the soft cap "
            "alone is advisory — the 2026-05-19 kernel watchdog panic happened with "
            "set_memory_limit(24 GB) but no wired cap). Default: 22 GB on klein-base-9b "
            "→ 20 GB wired cap → ~12 GB OS headroom on a 32 GB Max. Unset (MLX default) "
            "on smaller variants."
        ),
    )
    args = parser.parse_args()

    # Memory guardrail. See CLAUDE.md "Memory guardrails for heavy generations on 32 GB".
    # Set BEFORE the model load.
    cap_gb = args.mlx_memory_cap_gb
    if cap_gb is None and args.variant == "klein-base-9b":
        cap_gb = 22
    if cap_gb is not None:
        wired_gb = max(1, cap_gb - 2)
        mx.set_wired_limit(int(wired_gb * 1024**3))
        mx.set_memory_limit(int(cap_gb * 1024**3))
        print(
            f"MLX caps: wired={wired_gb} GB (mx.set_wired_limit, hard), "
            f"memory={cap_gb} GB (mx.set_memory_limit, soft)."
        )

    cfg = _variant_config(args.variant)
    print(f"Loading {args.variant} (quantize=4)...")
    flux = cfg["loader"](args.variant)
    guidance = args.guidance if args.guidance is not None else cfg["guidance"]
    num_inference_steps = (
        args.num_inference_steps if args.num_inference_steps is not None else cfg["num_inference_steps"]
    )
    # Default three-way only on klein-base-4b. klein-base-9b stays two-way by
    # default because the same-process 9-generation path isn't memory-safe at
    # 9B on 32 GB (see CLAUDE.md "Memory guardrails for heavy generations on
    # 32 GB"). v0.5.1 refactors this to subprocess-per-rep and will flip the
    # 9B default to three-way.
    three_way = args.three_way if args.three_way is not None else args.variant == "klein-base-4b"
    if three_way and args.variant == "klein-base-9b":
        print(
            "WARNING: three-way mode on klein-base-9b runs 9 same-process generations.\n"
            "  This path is NOT memory-safe at 9B on 32 GB unified memory; a prior\n"
            "  unguarded run hit system-level OOM. The subprocess-per-rep refactor\n"
            "  (v0.5.1) is the right path. Close other apps and watch memory pressure\n"
            "  if you proceed.",
        )

    bench_dir = args.images_dir / args.variant

    print(f"\n== Warmup (vanilla, {num_inference_steps} steps) ==")
    warmup_t, _ = _generate(flux, num_inference_steps=num_inference_steps, guidance=guidance)
    print(f"warmup: {warmup_t:.2f}s")

    print(f"\n== Vanilla x{args.reps} ==")
    vanilla_times: list[float] = []
    for i in range(args.reps):
        save = bench_dir / "vanilla.png" if i == 0 else None
        t, _ = _generate(
            flux,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            save_path=save,
        )
        vanilla_times.append(t)
        suffix = f"  (saved {save.name})" if save else ""
        print(f"  rep {i + 1}: {t:.2f}s{suffix}")

    # Free intermediates between conditions (CLAUDE.md "Memory guardrails").
    mx.metal.clear_cache()

    nogate_times: list[float] = []
    if three_way:
        print(f"\n== Wrapped (no gate, rel_l1_thresh=0) x{args.reps} ==")
        for i in range(args.reps):
            save = bench_dir / "wrapper_nogate.png" if i == 0 else None
            with apply_teacache(flux, rel_l1_thresh=0.0) as h:
                t, _ = _generate(
                    flux,
                    num_inference_steps=num_inference_steps,
                    guidance=guidance,
                    save_path=save,
                )
                nogate_times.append(t)
            suffix = f"  (saved {save.name})" if save else ""
            print(f"  rep {i + 1}: {t:.2f}s  (rel_l1_thresh=0, no skipping){suffix}")
        mx.metal.clear_cache()

    print(f"\n== TeaCache wrapper x{args.reps} ==")
    wrapper_times: list[float] = []
    skipped_counts: list[int] = []
    computed_counts: list[int] = []
    for i in range(args.reps):
        save_name = "wrapper_gated.png" if three_way else "wrapper.png"
        save = bench_dir / save_name if i == 0 else None
        with apply_teacache(flux) as h:
            t, _ = _generate(
                flux,
                num_inference_steps=num_inference_steps,
                guidance=guidance,
                save_path=save,
            )
            wrapper_times.append(t)
            skipped_counts.append(h.stats.skipped_count)
            computed_counts.append(h.stats.computed_count)
        suffix = f"  (saved {save.name})" if save else ""
        print(f"  rep {i + 1}: {t:.2f}s  (skipped {skipped_counts[-1]}/{num_inference_steps} steps){suffix}")

    vanilla_med = statistics.median(vanilla_times)
    wrapper_med = statistics.median(wrapper_times)
    speedup = vanilla_med / wrapper_med

    print("\n== Summary ==")
    print(f"  variant:          {args.variant}")
    print(f"  num_inference_steps: {num_inference_steps}")
    print(f"  guidance:         {guidance}")
    print(f"  reps:             {args.reps}")
    print(f"  vanilla median:   {vanilla_med:.2f}s   (all: {[round(x, 2) for x in vanilla_times]})")
    print(f"  wrapper median:   {wrapper_med:.2f}s   (all: {[round(x, 2) for x in wrapper_times]})")
    print(f"  speedup (median): {speedup:.2f}x")
    print(f"  skipped/computed: {skipped_counts} / {computed_counts}")
    if three_way:
        nogate_med = statistics.median(nogate_times)
        compile_avoidance_ratio = vanilla_med / nogate_med
        gating_ratio = nogate_med / wrapper_med
        combined_ratio = vanilla_med / wrapper_med
        print(
            f"  three-way medians: vanilla {vanilla_med:.2f}s | no-gate {nogate_med:.2f}s | gated {wrapper_med:.2f}s"
        )
        print(f"  compile-avoidance (vanilla / no-gate): {compile_avoidance_ratio:.2f}x  [v0.4 effect]")
        print(f"  gating          (no-gate / gated):    {gating_ratio:.2f}x  [v0.4.1 effect]")
        print(f"  combined        (vanilla / gated):     {combined_ratio:.2f}x")
        print(f"  bench images:     {bench_dir}/{{vanilla,wrapper_nogate,wrapper_gated}}.png")
    else:
        print(f"  bench images:     {bench_dir}/{{vanilla,wrapper}}.png")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report_data: dict[str, Any] = {
            "variant": args.variant,
            "num_inference_steps": num_inference_steps,
            "guidance": guidance,
            "prompt": PROMPT,
            "seed": SEED,
            "height": HEIGHT,
            "width": WIDTH,
            "reps": args.reps,
            "vanilla_seconds": vanilla_times,
            "wrapper_seconds": wrapper_times,
            "vanilla_median": vanilla_med,
            "wrapper_median": wrapper_med,
            "speedup_median": speedup,
            "skipped_counts": skipped_counts,
            "computed_counts": computed_counts,
            "bench_images_dir": str(bench_dir),
        }
        if three_way:
            nogate_med = statistics.median(nogate_times)
            compile_avoidance_ratio = vanilla_med / nogate_med
            gating_ratio = nogate_med / wrapper_med
            combined_ratio = vanilla_med / wrapper_med
            report_data["nogate_seconds"] = nogate_times
            report_data["nogate_median"] = nogate_med
            report_data["compile_avoidance_ratio"] = compile_avoidance_ratio
            report_data["gating_ratio"] = gating_ratio
            report_data["combined_ratio"] = combined_ratio
        args.report.write_text(json.dumps(report_data, indent=2))
        print(f"  report:           {args.report}")


if __name__ == "__main__":
    main()
