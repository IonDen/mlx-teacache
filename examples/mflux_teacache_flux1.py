"""mflux + mlx-teacache on FLUX.1-dev at 25 steps — the headline speedup.

Target search query: "mflux TeaCache FLUX.1", "speed up FLUX dev Mac",
"FLUX.1-dev faster Apple Silicon".

Expected output: writes `flux1_teacache.webp` next to this script.
Prints the TeaCache skip counts at the end so users can see the
gate actually firing (FLUX.1-dev at 25 steps + default threshold
skips ~6 of 25 steps for a measured 1.46× wall-clock on M1 Max).

Run with:
    uv run python examples/mflux_teacache_flux1.py
"""

from pathlib import Path

from mflux.models.flux.variants.txt2img.flux import Flux1

from mlx_teacache import apply_teacache

OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    print("loading Flux1 dev (quantize=4)...")
    model = Flux1.from_name("dev", quantize=4)

    print("wrapping with mlx-teacache...")
    handle = apply_teacache(model)

    print("generating: 'a red apple on a wooden table', 25 steps, seed=42...")
    generated = model.generate_image(
        seed=42,
        prompt="a red apple on a wooden table",
        num_inference_steps=25,
        width=512,
        height=512,
        guidance=3.5,
    )

    out_path = OUT_DIR / "flux1_teacache.webp"
    generated.image.save(out_path, "WEBP", quality=92)

    print(
        f"wrote {out_path}\n"
        f"TeaCache stats: skipped={handle.stats.skipped_count} / "
        f"computed={handle.stats.computed_count} (of 25 transformer calls)\n"
        f"variant: {getattr(handle, 'variant_id', 'unknown')}"
    )


if __name__ == "__main__":
    main()
