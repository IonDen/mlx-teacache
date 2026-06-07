"""FLUX.1 image-quality parity — SSIM on VAE-decoded images.

Complements `test_parity_flux1.py`'s latent-level paired parity with an
end-to-end image-level metric. At `rel_l1_thresh=0.0` the wrapper already
produces bit-exact latents (proved in `test_parity_flux1.py`); this file
focuses on default-threshold quality: with caching engaged (package
default `rel_l1_thresh=0.20`), the decoded image should remain
perceptually close to the same-process vanilla baseline.

This is the upstream-standard validation pattern. ali-vilab TeaCache,
ComfyUI-TeaCache, and HuggingFace Diffusers FirstBlockCache all validate
caching layers via visual quality comparison, not bit-exact byte parity.

Cost: each test does 2× generation (~5 min) + 2× VAE decode (~30s).
PR-time uses one prompt; full 5-prompt suite is gated by `@pytest.mark.slow`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

# scikit-image only lives in the test-mflux group, not test-core. When this
# file is collected by the pure-core CI job the bare `from skimage` import
# fails before pytest can evaluate the `parity` marker. `importorskip` makes
# collection skip the whole module cleanly in that environment.
ssim = pytest.importorskip("skimage.metrics").structural_similarity

from mlx_teacache import apply_teacache  # noqa: E402

pytestmark = pytest.mark.parity


# ---------------------------------------------------------------------------
# Constants — kept in sync with test_parity_flux1.py
# ---------------------------------------------------------------------------


REFERENCE_PROMPTS = (
    "a red apple on a wooden table",
    "mountain landscape at sunset",
    "portrait of a woman",
    "abstract pattern with circles",
    "text saying HELLO",
)
PR_TIME_PROMPT = "a red apple on a wooden table"

# Default threshold (rel_l1_thresh=0.20 as of 2026-05-15) targets "visually
# lossless" speedup. Measurements on FLUX.1-dev / 25 steps / red-apple prompt
# (2026-05-15):
#   threshold   skipped   speedup   SSIM
#   0.10        0         1.07x     1.0000   (cache never engages)
#   0.15        0         1.13x     1.0000   (cache never engages)
#   0.20        6         1.46x     0.81-0.95+ (visually near-identical, sweet spot)
#   0.25        11        1.96x     0.57-0.93 (visible style changes on text/synthetic prompts)
# The 0.25 -> 0.20 default change was made because text-heavy prompts at 0.25
# rendered as dot-matrix when vanilla rendered neon tubes. SSIM is a conservative
# metric here; visual inspection confirmed 0.20 is indistinguishable from vanilla
# while 0.25 was clearly different. Full-suite (slow) gate is looser to absorb
# prompt-to-prompt variance like the HELLO prompt's SSIM ~0.81 at threshold=0.20.
_PR_GATE_SSIM = 0.90
_FULL_SUITE_SSIM = 0.80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _LatentCapture:
    def __init__(self) -> None:
        self.latent: mx.array | None = None

    def call_after_loop(self, seed, prompt, latents, config, **_):  # noqa: ARG002
        self.latent = latents


def _unregister(flux: Any, cap: _LatentCapture) -> None:
    for attr in ("after_loop", "in_loop", "before_loop", "interrupt"):
        lst = getattr(flux.callbacks, attr, None)
        if isinstance(lst, list):
            for i in range(len(lst) - 1, -1, -1):
                if lst[i] is cap:
                    del lst[i]


def _capture(flux: Any, **gen_kwargs: Any) -> mx.array:
    cap = _LatentCapture()
    flux.callbacks.register(cap)
    try:
        flux.generate_image(**gen_kwargs)
    finally:
        _unregister(flux, cap)
    if cap.latent is None:
        raise RuntimeError("call_after_loop never fired - latent capture failed")
    return cap.latent


def _gen_kwargs_dev(prompt: str) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "seed": 42,
        "num_inference_steps": 25,
        "height": 512,
        "width": 512,
        "guidance": 3.5,
    }


def _decode_to_uint8(flux: Any, packed_latent: mx.array, *, height: int, width: int) -> np.ndarray:
    """Decode a packed latent through mflux's VAE to an HxWx3 uint8 image."""
    from mflux.models.common.vae.vae_util import VAEUtil
    from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator

    unpacked = FluxLatentCreator.unpack_latents(
        latents=packed_latent,
        height=height,
        width=width,
    )
    decoded = VAEUtil.decode(
        vae=flux.vae,
        latent=unpacked,
        tiling_config=getattr(flux, "tiling_config", None),
    )
    # mflux VAE outputs (B, C, H, W) in roughly [-1, 1] at the model's
    # precision (bf16/fp16). np.asarray can't convert bf16 directly via the
    # buffer protocol, so cast to fp32 first.
    decoded_fp32 = decoded.astype(mx.float32)
    mx.eval(decoded_fp32)
    img_np = np.asarray(decoded_fp32[0]).transpose(1, 2, 0)
    img_np = np.clip((img_np + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    return img_np


@pytest.fixture(scope="module")
def flux1_dev() -> Any:
    from mflux.models.flux.variants.txt2img.flux import Flux1

    flux = Flux1.from_name("dev", quantize=4)
    flux.freeze()
    return flux


# ---------------------------------------------------------------------------
# Image-quality tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image_strength,init_image_name",
    [
        (0.0, None),  # txt2img baseline
        (0.5, "natural_512.png"),  # img2img with natural init image
    ],
)
def test_default_threshold_ssim_dev_pr_gate(
    flux1_dev: Any, image_strength: float, init_image_name: str | None
) -> None:
    """PR-time image-quality gate: wrapper at the package-default
    rel_l1_thresh must produce a decoded image whose SSIM vs same-process
    vanilla is >= _PR_GATE_SSIM. ~5 min walltime per parametrize case."""
    kw = _gen_kwargs_dev(PR_TIME_PROMPT)
    if init_image_name is not None:
        kw["image_path"] = str(Path(__file__).parent / "fixtures" / "init_images" / init_image_name)
        kw["image_strength"] = image_strength

    vanilla_latent = _capture(flux1_dev, **kw)
    with apply_teacache(flux1_dev) as h:  # uses package default rel_l1_thresh
        wrapper_latent = _capture(flux1_dev, **kw)
        skipped = h.stats.skipped_count
    # For txt2img the cache must engage at the default threshold. For img2img
    # with short active windows the skip count is not guaranteed > 0, so we
    # only assert for the txt2img case.
    if image_strength == 0.0:
        assert skipped >= 1, (
            "default threshold should skip at least one step on the PR-gate "
            "prompt (cache must engage at the default)"
        )

    vanilla_img = _decode_to_uint8(
        flux1_dev,
        vanilla_latent,
        height=kw["height"],
        width=kw["width"],
    )
    wrapper_img = _decode_to_uint8(
        flux1_dev,
        wrapper_latent,
        height=kw["height"],
        width=kw["width"],
    )
    score = ssim(vanilla_img, wrapper_img, channel_axis=-1, data_range=255)
    assert score >= _PR_GATE_SSIM, (
        f"SSIM {score:.4f} < {_PR_GATE_SSIM}; wrapper image "
        f"diverged from same-process vanilla baseline at default threshold "
        f"(image_strength={image_strength}, {skipped} steps skipped)"
    )


@pytest.mark.slow
@pytest.mark.parametrize("prompt", REFERENCE_PROMPTS)
def test_default_threshold_ssim_dev_full(flux1_dev: Any, prompt: str) -> None:
    """Nightly image-quality gate: all 5 reference prompts at the package
    default threshold. Looser SSIM gate than the PR-gate test to absorb
    prompt-to-prompt variance — high-frequency-detail prompts (text,
    synthetic patterns) show lower SSIM than natural images at the same
    threshold even when the wrapper output is visually equivalent."""
    kw = _gen_kwargs_dev(prompt)
    vanilla_latent = _capture(flux1_dev, **kw)
    with apply_teacache(flux1_dev) as h:  # uses package default rel_l1_thresh
        wrapper_latent = _capture(flux1_dev, **kw)
        skipped = h.stats.skipped_count
    del skipped  # not asserted; skipped_count varies per prompt at any threshold
    vanilla_img = _decode_to_uint8(flux1_dev, vanilla_latent, height=kw["height"], width=kw["width"])
    wrapper_img = _decode_to_uint8(flux1_dev, wrapper_latent, height=kw["height"], width=kw["width"])
    score = ssim(vanilla_img, wrapper_img, channel_axis=-1, data_range=255)
    assert score >= _FULL_SUITE_SSIM, f"SSIM {score:.4f} < {_FULL_SUITE_SSIM} on prompt {prompt!r}"
