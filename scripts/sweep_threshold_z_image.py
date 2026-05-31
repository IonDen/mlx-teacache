"""Threshold sweep for z-image-base — picks DEFAULT_THRESH at the SSIM knee.

Generates a vanilla baseline at the pinned recipe, then sweeps the wrapper
across `rel_l1_thresh` values, recording skip count + wall-clock + SSIM-vs-
vanilla at each. DEFAULT_THRESH is set at the visible knee where SSIM holds the
FLUX.2-comparable high bar (>= ~0.97-0.99 measured, not the 0.85 test floor).

SEQUENCING: this RUNS only AFTER Phase 3 — it calls `apply_teacache(flux, ...)`,
which requires the z-image-base variant to be registered with a working
`integration.py`. (Calibration, by contrast, monkeypatches `_predict` directly
and does not need the variant.)

SIGNAL SELECTION (A vs B): run this once per candidate signal. The integration
exposes the signal choice (Phase 3 design); pass `--signal A|B`. Commit the
signal whose curve is usable AND whose held-out skip-vs-SSIM knee is better.

Run (AFTER Phase 3; HEAVY — vanilla baseline + one wrapped generation per
threshold at ~150-220s each):

    uv run python scripts/sweep_threshold_z_image.py --signal A

Produces tests/_artifacts/sweep_z_image/{vanilla,t<thresh>}.png + results.json.
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Pinned recipe (findings 2026-05-31).
PROMPT = "a red apple on a wooden table"
SEED = 42
HEIGHT = WIDTH = 512
STEPS = 50
GUIDANCE = 4.0
QUANTIZE = 8

# Coarse sweep; refine around the knee after the first pass.
THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.17, 0.20, 0.25, 0.30]

OUT_DIR = Path(__file__).parent.parent / "tests" / "_artifacts" / "sweep_z_image"


def _gen(flux, *, save_path: Path) -> float:
    start = time.perf_counter()
    image = flux.generate_image(
        prompt=PROMPT, seed=SEED, num_inference_steps=STEPS, height=HEIGHT, width=WIDTH, guidance=GUIDANCE
    )
    mx.eval(mx.zeros(1))
    elapsed = time.perf_counter() - start
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path=str(save_path), export_json_metadata=False)
    return elapsed


def _load(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signal", choices=["A", "B"], default="A", help="Which calibrated gate signal to sweep."
    )
    args = parser.parse_args()

    mx.set_wired_limit(int(20 * 1024**3))
    mx.set_memory_limit(int(22 * 1024**3))

    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.z_image.variants.z_image import ZImage

    from mlx_teacache import apply_teacache

    print(f"Loading z-image-base (quantize={QUANTIZE})...", flush=True)
    flux = ZImage(quantize=QUANTIZE, model_config=ModelConfig.z_image())
    flux.freeze()

    print("== Warmup ==", flush=True)
    _gen(flux, save_path=OUT_DIR / "warmup.png")
    print(f"== Vanilla baseline ({STEPS} steps) ==", flush=True)
    van_path = OUT_DIR / "vanilla.png"
    van_t = _gen(flux, save_path=van_path)
    van_arr = _load(van_path)
    print(f"  vanilla: {van_t:.2f}s", flush=True)

    results = []
    for t in THRESHOLDS:
        wrap_path = OUT_DIR / f"{args.signal}_t{t:.3f}.png"
        # NOTE: the `signal=` selector is a Phase-3 integration hook; until then
        # this sweeps the variant's committed signal. See header.
        with apply_teacache(flux, rel_l1_thresh=t) as h:
            wrap_t = _gen(flux, save_path=wrap_path)
            skipped, computed = h.stats.skipped_count, h.stats.computed_count
        score = float(ssim(van_arr, _load(wrap_path), channel_axis=-1, data_range=255))
        results.append(
            {
                "signal": args.signal,
                "threshold": t,
                "wrapper_seconds": wrap_t,
                "speedup_vs_vanilla_single_rep": van_t / wrap_t,
                "skipped": skipped,
                "computed": computed,
                "ssim_vs_vanilla": score,
            }
        )
        print(
            f"  signal={args.signal} t={t:.3f} skipped={skipped}/{STEPS} {wrap_t:.1f}s "
            f"speedup={van_t / wrap_t:.2f}x SSIM={score:.4f}",
            flush=True,
        )

    summary = {
        "variant": "z-image-base",
        "signal": args.signal,
        "num_inference_steps": STEPS,
        "guidance": GUIDANCE,
        "quantize": QUANTIZE,
        "prompt": PROMPT,
        "seed": SEED,
        "height": HEIGHT,
        "width": WIDTH,
        "vanilla_seconds": van_t,
        "thresholds": results,
        "note": "Single-rep wall-clock (thermal noise); SSIM is deterministic per threshold. "
        "Choose DEFAULT_THRESH at the knee where SSIM holds the high bar.",
    }
    (OUT_DIR / f"results_{args.signal}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR / f'results_{args.signal}.json'}")


if __name__ == "__main__":
    main()
