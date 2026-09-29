"""mflux + mlx-teacache + mlx-taef live preview — combined-use story.

Target search query: "mflux teacache taef", "fast FLUX previews
Mac", "combined Apple Silicon FLUX speedup".

Expected output: writes `combined_step{NN}.png` frames + a
`combined_final.webp` next to this script. Prints the TeaCache
skip count at the end.

Run with:
    uv run python examples/mflux_combined_with_taef.py

Requires both libraries: `pip install mlx-teacache[mflux] mlx-taef`.
FLUX.1-dev at 25 steps is a schedule where the gate skips steps, so
TeaCache and the previews both have something to show. For measured
speedups and skip counts, see the README benchmarks.
"""

from pathlib import Path

from mflux.models.flux.variants.txt2img.flux import Flux1
from mlx_taef.integrations.mflux import LivePreviewCallback

from mlx_teacache import apply_teacache

OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    print("loading Flux1 dev (quantize=4)...")
    model = Flux1.from_name("dev", quantize=4)

    print("wrapping with mlx-teacache...")
    handle = apply_teacache(model)

    callback = LivePreviewCallback(
        flux=model,
        variant="taef1",
        every=5,
        numbered_frames=True,
        save_to=OUT_DIR / "combined.png",
        latent_height=32,
        latent_width=32,
    )
    model.callbacks.register(callback)

    print("generating: 'a red apple on a wooden table', 25 steps + TeaCache + TAEF1, seed=42...")
    generated = model.generate_image(
        seed=42,
        prompt="a red apple on a wooden table",
        num_inference_steps=25,
        width=512,
        height=512,
        guidance=3.5,
    )

    final_path = OUT_DIR / "combined_final.webp"
    generated.image.save(final_path, "WEBP", quality=92)

    print(
        f"wrote {len(callback.saved_paths)} preview frames + {final_path}\n"
        f"TeaCache stats: skipped={handle.stats.skipped_count} / "
        f"computed={handle.stats.computed_count}\n"
        f"variant: {handle.variant_id}"
    )


if __name__ == "__main__":
    main()
