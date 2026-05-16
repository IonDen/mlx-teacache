#!/usr/bin/env python3
"""Generate the two committed init-image fixtures used by img2img tests.

Run once at fixture creation; commit the resulting PNGs. Deterministic via
numpy.random.default_rng(0). To regenerate (e.g., after a fixture format
change), delete the PNGs and re-run.

Usage:
    uv run python scripts/generate_init_image_fixtures.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

OUT_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "init_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_natural() -> Image.Image:
    """Pseudo-natural image: smoothed noise base + a soft horizon gradient."""
    rng = np.random.default_rng(0)
    base = rng.integers(120, 200, size=(512, 512, 3), dtype=np.uint8)
    # Add a vertical gradient (simulated horizon).
    gradient = np.linspace(0, 60, 512, dtype=np.int16)[:, None, None]
    arr = np.clip(base.astype(np.int16) + gradient, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def generate_synthetic() -> Image.Image:
    """Synthetic pattern: deterministic concentric rings + checkerboard."""
    rng = np.random.default_rng(1)
    yy, xx = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 512), indexing="ij")
    radii = np.sqrt(xx**2 + yy**2)
    rings = (np.sin(radii * 12) * 127 + 128).astype(np.uint8)
    checker = (((np.floor(xx * 8) + np.floor(yy * 8)) % 2) * 255).astype(np.uint8)
    r = rings
    g = checker
    b = rng.integers(40, 220, size=(512, 512), dtype=np.uint8)
    arr = np.stack([r, g, b], axis=-1)
    return Image.fromarray(arr, mode="RGB")


def main() -> None:
    natural_path = OUT_DIR / "natural_512.png"
    synthetic_path = OUT_DIR / "synthetic_512.png"
    generate_natural().save(natural_path, format="PNG", optimize=True)
    generate_synthetic().save(synthetic_path, format="PNG", optimize=True)
    print(f"wrote {natural_path}")
    print(f"wrote {synthetic_path}")


if __name__ == "__main__":
    main()
