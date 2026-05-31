"""Threshold sweep for z-image-base — picks DEFAULT_THRESH at the SSIM knee.

Generates a vanilla baseline at the pinned recipe, then sweeps the wrapper
across `rel_l1_thresh` values, recording skip count + wall-clock + SSIM-vs-
vanilla at each. DEFAULT_THRESH is set at the visible knee where SSIM holds the
FLUX.2-comparable high bar (>= ~0.97-0.99 measured, not the 0.85 test floor).

The committed z-image-base variant uses Signal B (first-main-layer residual);
Signal A was rejected at calibration (R^2=0.069 vs B's 0.400 — see the
2026-05-31 calibration findings). So this sweeps the one committed signal; there
is no A/B selector. This run also serves as the skip-path quality validation at
the VALID recipe: at low thresholds (few skips) SSIM should stay high; if even a
handful of skips crater SSIM, that signals a reconstruction bug to debug.

SEQUENCING: RUNS only AFTER Phase 3 — it calls `apply_teacache(flux, ...)`,
which requires the registered z-image-base variant. (Calibration, by contrast,
monkeypatches `_predict` directly and does not need the variant.)

Run (HEAVY — vanilla baseline + one wrapped generation per threshold at
~150-220s each; skips speed the higher thresholds up):

    uv run python scripts/sweep_threshold_z_image.py

Produces tests/_artifacts/sweep_z_image/{vanilla,t<thresh>}.png + results_z_image.json.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Pinned recipe (findings 2026-05-31). Same prompt as the parity test + bench +
# comparison so every artifact is comparable.
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
        wrap_path = OUT_DIR / f"t{t:.3f}.png"
        with apply_teacache(flux, rel_l1_thresh=t) as h:
            wrap_t = _gen(flux, save_path=wrap_path)
            skipped, computed = h.stats.skipped_count, h.stats.computed_count
        score = float(ssim(van_arr, _load(wrap_path), channel_axis=-1, data_range=255))
        results.append(
            {
                "threshold": t,
                "wrapper_seconds": wrap_t,
                "speedup_vs_vanilla_single_rep": van_t / wrap_t,
                "skipped": skipped,
                "computed": computed,
                "ssim_vs_vanilla": score,
            }
        )
        print(
            f"  t={t:.3f} skipped={skipped} computed={computed} {wrap_t:.1f}s "
            f"speedup={van_t / wrap_t:.2f}x SSIM={score:.4f}",
            flush=True,
        )

    summary = {
        "variant": "z-image-base",
        "signal": "B",
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
        "Choose DEFAULT_THRESH at the knee where SSIM holds the high bar. skipped/computed are "
        "per denoising step (one shared CFG gate decision per step; each skip avoids both "
        "branches' 30-layer bodies).",
    }
    (OUT_DIR / "results_z_image.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR / 'results_z_image.json'}")


if __name__ == "__main__":
    main()
