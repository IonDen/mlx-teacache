"""One-shot validation that flux2-klein-base-9b's reused base-4b coefficients
work at the canonical 50-step CFG recipe.

Generates one fixed prompt at seed 42, 1024x768, num_inference_steps=50,
guidance=4.0, both vanilla and wrapped via apply_teacache. Decodes through
the VAE, computes SSIM, writes _artifacts/validation_klein_base_9b.json.
Exits non-zero if SSIM < 0.95.

This is a release-gate run for v0.5.0 — not a generic benchmark. Run once
before tagging. Heavy generation; expect ~30-90 min on M1 Max.

Usage:
  uv run python scripts/validate_klein_base_9b.py
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Reuse the comparison prompt for continuity with COMPARISON.md.
PROMPT = (
    "Portrait of a young woman with auburn hair and green eyes, soft "
    "golden-hour window light, photorealistic, shallow depth of field, "
    "50mm prime lens, subtle freckles, neutral background, cinematic "
    "color grading."
)
SEED = 42
HEIGHT = 1024
WIDTH = 768
STEPS = 50
GUIDANCE = 4.0
SSIM_THRESHOLD = 0.95


def _detect_hardware() -> dict[str, Any]:
    chip = (
        subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        or "Apple Silicon"
    )
    ram_str = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()
    ram_bytes = int(ram_str) if ram_str else 0
    return {
        "chip": chip,
        "ram_gb": round(ram_bytes / (1024**3)),
        "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
    }


def _load_flux() -> Any:
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_9b())
    flux.freeze()
    return flux


def _generate(flux: Any) -> Any:
    import mlx.core as mx

    image = flux.generate_image(
        prompt=PROMPT,
        seed=SEED,
        num_inference_steps=STEPS,
        height=HEIGHT,
        width=WIDTH,
        guidance=GUIDANCE,
    )
    mx.eval(mx.zeros(1))  # flush GPU work before stopping any external clock
    return image


def _to_numpy(image: Any) -> Any:
    """mflux GeneratedImage exposes the PIL image at `.image`."""
    import numpy as np

    pil = image.image
    return np.array(pil).astype(np.float32) / 255.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "_artifacts" / "validation_klein_base_9b.json",
    )
    args = parser.parse_args()

    from skimage.metrics import structural_similarity as ssim

    from mlx_teacache import apply_teacache

    print(f"=== klein-base-9b validation: {STEPS} steps, guidance={GUIDANCE} ===")
    flux = _load_flux()

    print(">> vanilla generation")
    t0 = time.perf_counter()
    vanilla_image = _generate(flux)
    vanilla_seconds = time.perf_counter() - t0
    print(f"   {vanilla_seconds:.1f}s")
    vanilla_np = _to_numpy(vanilla_image)

    print(">> wrapper generation")
    t0 = time.perf_counter()
    with apply_teacache(flux) as handle:
        wrapper_image = _generate(flux)
        wrapper_seconds = time.perf_counter() - t0
        skipped = handle.stats.skipped_count
        computed = handle.stats.computed_count
        thresh = handle.rel_l1_thresh
    print(f"   {wrapper_seconds:.1f}s, skipped={skipped}/{computed + skipped}, thresh={thresh}")
    wrapper_np = _to_numpy(wrapper_image)

    score = float(ssim(vanilla_np, wrapper_np, data_range=1.0, channel_axis=-1))
    passed = score >= SSIM_THRESHOLD

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
        "hardware": _detect_hardware(),
        "prompt": PROMPT,
        "seed": SEED,
        "height": HEIGHT,
        "width": WIDTH,
        "num_inference_steps": STEPS,
        "guidance": GUIDANCE,
        "rel_l1_thresh_used": thresh,
        "vanilla_seconds": vanilla_seconds,
        "wrapper_seconds": wrapper_seconds,
        "wrapper_skipped": skipped,
        "wrapper_computed": computed,
        "ssim": score,
        "ssim_threshold": SSIM_THRESHOLD,
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nReport written: {args.output}")
    print(f"SSIM: {score:.4f} (threshold {SSIM_THRESHOLD})")
    print("RESULT: PASS" if passed else "RESULT: FAIL")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
