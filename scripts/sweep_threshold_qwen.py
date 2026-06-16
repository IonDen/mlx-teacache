"""Threshold sweep for qwen-image — picks DEFAULT_THRESH at the SSIM knee.

Generates a vanilla baseline at the pinned recipe, then sweeps the wrapper across
`rel_l1_thresh` values, recording skip count + wall-clock + SSIM-vs-vanilla at
each. DEFAULT_THRESH is set at the visible knee where SSIM holds the high bar
(>= ~0.97-0.99 measured, not the 0.85 test floor).

The committed qwen-image variant gates on Signal A (modulated block-0 input;
calibrated R^2 0.946 — chosen over Signal B for caption-independence + cheaper
skips; see config.py). So this sweeps the one committed signal; there is no A/B
selector. This run also doubles as the skip-path quality validation at the valid
recipe: if a handful of skips crater SSIM, that signals a reconstruction bug.

SEQUENCING: runs AFTER the variant is registered — it calls `apply_teacache(flux,
rel_l1_thresh=...)`, which detects + wraps the QwenImage instance. (Calibration,
by contrast, proxies `flux.transformer` directly and does not need the variant.)

MEMORY: 20B q4 + CFG peaks ~27.6 GB on a 32 GB M1 Max — near the ceiling. The
wired cap is device-derived (NOT a hardcoded literal) and `mx.clear_cache()` runs
between thresholds so the peak stays ~one generation across the sweep.

Run (HEAVY — vanilla baseline + one wrapped generation per threshold; higher
thresholds skip steps and run faster). Main thread only:

    uv run python scripts/sweep_threshold_qwen.py

Produces tests/_artifacts/sweep_qwen/{vanilla,t<thresh>}.png + results_qwen.json.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Pinned recipe — matches the calibration + the bench_speedup red-apple recipe so
# every artifact is comparable. 512x512 is the memory fallback from 768x768.
PROMPT = "a red apple on a wooden table"
SEED = 42
HEIGHT = WIDTH = 512
STEPS = 20
GUIDANCE = 4.0
QUANTIZE = 4

# Coarse sweep; refine around the knee after the first pass.
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

OUT_DIR = Path(__file__).parent.parent / "tests" / "_artifacts" / "sweep_qwen"


def _gen(flux, *, save_path: Path) -> float:  # noqa: ANN001
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
    # Device-derived wired cap, strictly below the recommended working set (NOT a
    # hardcoded 20 GB literal — that broke on machines with a smaller ceiling).
    _max_set = mx.device_info()["max_recommended_working_set_size"]
    mx.set_wired_limit(int(_max_set * 0.85))

    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

    from mlx_teacache import apply_teacache

    print(f"Loading qwen-image (quantize={QUANTIZE})...", flush=True)
    flux = QwenImage(quantize=QUANTIZE, model_config=ModelConfig.qwen_image())
    flux.freeze()

    print("== Warmup ==", flush=True)
    _gen(flux, save_path=OUT_DIR / "warmup.png")
    mx.clear_cache()
    print(f"== Vanilla baseline ({STEPS} steps) ==", flush=True)
    van_path = OUT_DIR / "vanilla.png"
    van_t = _gen(flux, save_path=van_path)
    van_arr = _load(van_path)
    mx.clear_cache()
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
        mx.clear_cache()  # keep the peak ~one generation across the sweep (memory edge)

    summary = {
        "variant": "qwen-image",
        "signal": "A",
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
        "branches' 60-block bodies).",
    }
    (OUT_DIR / "results_qwen.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR / 'results_qwen.json'}")


if __name__ == "__main__":
    main()
