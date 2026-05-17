"""Threshold-sweep reproducer for flux2-klein-base-4b's per-variant default.

This is the script that was used to choose `Provenance.default_thresh=0.17`
for `flux2-klein-base-4b` in v0.4.0. It generates a vanilla baseline at the
calibrated 25-step schedule, then sweeps the wrapper across a list of
`rel_l1_thresh` values, measuring skip count + wall-clock + SSIM-vs-vanilla
at each. The 0.17 default sits on the visible knee of the curve.

Run as:
    uv run python scripts/sweep_threshold_klein_base_4b.py

Produces:
    tests/_artifacts/sweep_klein_base_4b/vanilla.png
    tests/_artifacts/sweep_klein_base_4b/t<thresh>.png  (one per threshold)
    tests/_artifacts/sweep_klein_base_4b/results.json    (full summary)
    stdout: markdown table

The output directory is gitignored (`tests/_artifacts/`).

Measured on M1 Max 32GB, mflux 0.17.5, quantize=4, 512×512, seed=42,
guidance=1.0, red-apple prompt, 2026-05-17. Single-rep measurements — for
3-rep stable wall-clock numbers (the 1.41× README headline) run
`scripts/bench_speedup.py --variant klein-base-4b` instead.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from mlx_teacache import apply_teacache

PROMPT = "a red apple on a wooden table"
SEED = 42
HEIGHT = WIDTH = 512
STEPS = 25

# Coarse + fine, in one pass. The knee is between 0.165 and 0.175.
THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18]

OUT_DIR = Path(__file__).parent.parent / "tests" / "_artifacts" / "sweep_klein_base_4b"


def _gen(flux, *, save_path: Path) -> float:
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


def main() -> None:
    print("Loading flux2-klein-base-4b (quantize=4)...")
    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
    flux.freeze()

    print("\n== Warmup (vanilla, no save) ==")
    _gen(flux, save_path=OUT_DIR / "warmup.png")

    print(f"\n== Vanilla baseline (full {STEPS} steps) ==")
    van_path = OUT_DIR / "vanilla.png"
    van_t = _gen(flux, save_path=van_path)
    van_arr = _load(van_path)
    print(f"  vanilla: {van_t:.2f}s  saved {van_path}")

    print("\n== Threshold sweep ==")
    results = []
    for t in THRESHOLDS:
        print(f"\n  threshold={t:.3f}")
        wrap_path = OUT_DIR / f"t{t:.3f}.png"
        with apply_teacache(flux, rel_l1_thresh=t) as h:
            wrap_t = _gen(flux, save_path=wrap_path)
            skipped = h.stats.skipped_count
            computed = h.stats.computed_count
        wrap_arr = _load(wrap_path)
        score = float(ssim(van_arr, wrap_arr, channel_axis=-1, data_range=255))
        speedup = van_t / wrap_t
        results.append(
            {
                "threshold": t,
                "wrapper_seconds": wrap_t,
                "speedup_vs_vanilla_single_rep": speedup,
                "skipped": skipped,
                "computed": computed,
                "ssim_vs_vanilla": score,
                "image_path": str(wrap_path.relative_to(OUT_DIR.parent.parent.parent)),
            }
        )
        print(f"    skipped={skipped}/{STEPS}  time={wrap_t:.2f}s  speedup={speedup:.2f}x  SSIM={score:.4f}")

    summary = {
        "variant": "flux2-klein-base-4b",
        "num_inference_steps": STEPS,
        "guidance": 1.0,
        "prompt": PROMPT,
        "seed": SEED,
        "height": HEIGHT,
        "width": WIDTH,
        "vanilla_seconds": van_t,
        "thresholds": results,
        "note": (
            "Single-rep wall-clock measurements; subject to thermal noise. "
            "For the stable 3-rep median wall-clock numbers reported in the README "
            "(77.5s vanilla / 55.1s wrapper / 1.41x at threshold 0.17) run "
            "`scripts/bench_speedup.py --variant klein-base-4b`. SSIM is stable "
            "across reps since the wrapper output is deterministic at a given threshold."
        ),
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2))

    print(f"\n\n== Sweep summary (vanilla {van_t:.2f}s) ==")
    print("\n| Threshold | Skipped | Wall-clock | Single-rep speedup | SSIM vs vanilla |")
    print("|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['threshold']:.3f} | {r['skipped']}/{STEPS} | {r['wrapper_seconds']:.2f}s | "
            f"{r['speedup_vs_vanilla_single_rep']:.2f}x | {r['ssim_vs_vanilla']:.4f} |"
        )
    print(f"\n  Results JSON: {OUT_DIR / 'results.json'}")
    print(f"  PNGs: {OUT_DIR}/*.png")


if __name__ == "__main__":
    main()
