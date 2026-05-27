"""mflux + mlx-teacache on FLUX.2 Klein base 4B at 50 steps + CFG.

Target search query: "FLUX.2 Klein base speedup", "FLUX.2 mflux
faster", "non-distilled FLUX.2 Apple Silicon".

Expected output: writes `flux2_base_teacache.webp` next to this
script. Prints the TeaCache skip counts. On M1 Max under v0.6.0's
subprocess-per-rep harness, klein-base-4b at 50 steps + g=4.0
skips ~9 / 50 steps for a measured 1.23× wall-clock.

Run with:
    uv run python examples/mflux_teacache_flux2_base.py

Use the BASE (non-distilled) Flux2Klein variant — the distilled
4-8 step Klein defaults do NOT engage the polynomial gate by
design; see README 'Benchmarks → How the speedup happens'.
"""

from pathlib import Path

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

from mlx_teacache import apply_teacache

OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    print("loading Flux2Klein base 4B (quantize=4)...")
    model = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())

    print("wrapping with mlx-teacache...")
    handle = apply_teacache(model)

    print("generating: 'a red apple on a wooden table', 50 steps + g=4.0, seed=42...")
    generated = model.generate_image(
        seed=42,
        prompt="a red apple on a wooden table",
        num_inference_steps=50,
        width=512,
        height=512,
        guidance=4.0,
    )

    out_path = OUT_DIR / "flux2_base_teacache.webp"
    generated.image.save(out_path, "WEBP", quality=92)

    print(
        f"wrote {out_path}\n"
        f"TeaCache stats: skipped={handle.stats.skipped_count} / "
        f"computed={handle.stats.computed_count} (of 50 transformer calls)\n"
        f"variant: {getattr(handle, 'variant_id', 'unknown')}"
    )


if __name__ == "__main__":
    main()
