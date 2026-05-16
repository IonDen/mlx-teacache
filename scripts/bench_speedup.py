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
  uv run python scripts/bench_speedup.py --variant flux1-dev

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
        choices=["klein-4b", "klein-9b", "flux1-dev", "flux1-schnell"],
    )
    parser.add_argument("--reps", type=int, default=3, help="timed reps per condition")
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
    args = parser.parse_args()

    cfg = _variant_config(args.variant)
    print(f"Loading {args.variant} (quantize=4)...")
    flux = cfg["loader"](args.variant)
    num_inference_steps = cfg["num_inference_steps"]
    guidance = cfg["guidance"]

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

    print(f"\n== TeaCache wrapper x{args.reps} ==")
    wrapper_times: list[float] = []
    skipped_counts: list[int] = []
    computed_counts: list[int] = []
    for i in range(args.reps):
        save = bench_dir / "wrapper.png" if i == 0 else None
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
    print(f"  bench images:     {bench_dir}/{{vanilla,wrapper}}.png")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
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
                },
                indent=2,
            )
        )
        print(f"  report:           {args.report}")


if __name__ == "__main__":
    main()
