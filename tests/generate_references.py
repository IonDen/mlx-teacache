# tests/generate_references.py
"""Regenerate reference latents for parity tests.

Usage:
    python tests/generate_references.py --variant flux1-dev
    python tests/generate_references.py --variant flux1-schnell
    python tests/generate_references.py --variant flux2-klein-4b

Outputs:
    tests/reference/<variant_id>/<prompt_slug>__seed42__steps25.safetensors

Each safetensors file contains a single tensor "latent" with the final
denoised latent (pre-VAE-decode). The fixtures are committed and SHA-pinned
in tests/fixtures.toml. Regenerate ONLY when fixtures are intentionally
updated; update fixtures.toml SHAs in the same commit.

Latent capture uses a private AfterLoopCallback (`_LatentCapture`) because
mflux's `generate_image` returns a `GeneratedImage` that does not expose the
pre-VAE latent directly. mflux calls `ctx.after_loop(latents)` with the
final pre-VAE latent right before VAE-decode — that's our hook point."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import mlx.core as mx

REFERENCE_PROMPTS = [
    "a red apple on a wooden table",
    "mountain landscape at sunset",
    "portrait of a woman",
    "abstract pattern with circles",
    "text saying HELLO",
]
DEFAULT_SEED = 42
DEFAULT_STEPS = 25
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512


class _LatentCapture:
    """Captures the final pre-VAE latent via after_loop. Use one per generation."""

    def __init__(self) -> None:
        self.latent: mx.array | None = None

    def call_after_loop(self, seed, prompt, latents, config, **_):  # noqa: ARG002
        self.latent = latents


def _slug(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")


def _save_reference_latent(out_path: Path, latent: mx.array) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_path), {"latent": latent})
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"{out_path.relative_to(out_path.parent.parent.parent)}  sha256={digest}")


def _generate_one(flux, prompt: str, *, num_steps: int, height: int, width: int, **gen_kwargs) -> mx.array:
    """Run one generation, capturing the final pre-VAE latent via after_loop."""
    cap = _LatentCapture()
    flux.callbacks.register(cap)
    try:
        flux.generate_image(
            seed=DEFAULT_SEED,
            prompt=prompt,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            **gen_kwargs,
        )
    finally:
        # Remove the capture callback to avoid leaks between prompts.
        for lst_name in ("after_loop_callbacks", "_callbacks", "callbacks"):
            lst = getattr(flux.callbacks, lst_name, None)
            if isinstance(lst, list):
                for i in range(len(lst) - 1, -1, -1):
                    if lst[i] is cap:
                        del lst[i]
    if cap.latent is None:
        raise RuntimeError(f"AfterLoop callback never fired for prompt {prompt!r}")
    return cap.latent


def _generate_flux1(variant: str) -> None:
    from mflux.models.flux.variants.txt2img.flux import Flux1

    model_name = variant.removeprefix("flux1-")
    flux = Flux1.from_name(model_name, quantize=4)
    flux.freeze()

    guidance = 3.5 if variant == "flux1-dev" else 0.0
    out_root = Path(__file__).parent / "reference" / variant
    for prompt in REFERENCE_PROMPTS:
        latent = _generate_one(
            flux,
            prompt,
            num_steps=DEFAULT_STEPS,
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            guidance=guidance,
        )
        out_path = out_root / f"{_slug(prompt)}__seed{DEFAULT_SEED}__steps{DEFAULT_STEPS}.safetensors"
        _save_reference_latent(out_path, latent)


def _generate_flux2(variant: str) -> None:
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    if variant != "flux2-klein-4b":
        raise ValueError(f"Only flux2-klein-4b is supported in v0.1, got {variant}")

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())
    flux.freeze()

    out_root = Path(__file__).parent / "reference" / variant
    for prompt in REFERENCE_PROMPTS:
        latent = _generate_one(
            flux,
            prompt,
            num_steps=DEFAULT_STEPS,
            height=DEFAULT_HEIGHT,
            width=DEFAULT_WIDTH,
            guidance=1.0,  # no CFG by default for Klein
        )
        out_path = out_root / f"{_slug(prompt)}__seed{DEFAULT_SEED}__steps{DEFAULT_STEPS}.safetensors"
        _save_reference_latent(out_path, latent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["flux1-dev", "flux1-schnell", "flux2-klein-4b"])
    args = parser.parse_args()
    if args.variant.startswith("flux1-"):
        _generate_flux1(args.variant)
    else:
        _generate_flux2(args.variant)


if __name__ == "__main__":
    main()
