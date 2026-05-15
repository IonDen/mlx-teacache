"""FLUX.2 Klein 4b image-quality parity — SSIM on VAE-decoded images.

Mirrors `test_image_quality_flux1.py` (v2.6) for FLUX.2. At
`rel_l1_thresh=0.0` the wrapper is already covered by the latent paired
parity in `test_parity_flux2.py`; this file focuses on default-threshold
quality (with caching engaged the decoded image should remain
perceptually close to same-process vanilla).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
import pytest

# scikit-image only lives in the test-mflux group, not test-core. See
# `test_image_quality_flux1.py` for the same pattern + rationale.
ssim = pytest.importorskip("skimage.metrics").structural_similarity

from mlx_teacache import apply_teacache  # noqa: E402

pytestmark = pytest.mark.parity


REFERENCE_PROMPTS = (
    "a red apple on a wooden table",
    "mountain landscape at sunset",
    "portrait of a woman",
    "abstract pattern with circles",
    "text saying HELLO",
)
PR_TIME_PROMPT = "a red apple on a wooden table"

# Calibrate against measurement before tightening. The python-ml-testing
# skill cites SSIM >= 0.90 as the "acceptable" zone for full VAEs; FLUX.2
# Klein 4b uses unmeasured-in-repo coefficients (placeholder until Task 29
# completes), so the SSIM here is a coarse correctness gate rather than a
# tight quality target. Tighten after Task 29 calibration + slow-suite
# measurement.
_DEFAULT_THRESHOLD_SSIM = 0.85


# ---------------------------------------------------------------------------
# Helpers — mirror test_image_quality_flux1.py
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


def _gen_kwargs_klein(prompt: str, *, guidance: float = 1.0) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "seed": 42,
        "num_inference_steps": 8,
        "height": 512,
        "width": 512,
        "guidance": guidance,
    }


def _decode_to_uint8(flux: Any, packed_latent: mx.array, *, height: int, width: int) -> np.ndarray:
    """Decode a FLUX.2 packed latent through mflux's VAE to HxWx3 uint8.

    The `after_loop` callback fires with `latents` shaped
    `(batch, latent_h*latent_w, channels)` (pre-reshape). mflux then reshapes
    to `(batch, channels, latent_h, latent_w)` and calls
    `vae.decode_packed_latents`. We mirror that here. See
    `mflux/models/flux2/variants/txt2img/flux2_klein.py` lines 114-118."""
    # FLUX.2 packed-latent scale factor: image dim / 16 (vae_scale_factor=8
    # combined with patch_size=2).
    latent_h = height // 16
    latent_w = width // 16
    batch = packed_latent.shape[0]
    channels = packed_latent.shape[-1]
    packed = packed_latent.reshape(batch, latent_h, latent_w, channels).transpose(0, 3, 1, 2)
    decoded = flux.vae.decode_packed_latents(packed)
    decoded_fp32 = decoded.astype(mx.float32)
    mx.eval(decoded_fp32)
    arr = np.asarray(decoded_fp32[0])
    # mflux VAE outputs (C, H, W) in roughly [-1, 1].
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    img_np = np.clip((arr + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    return img_np


@pytest.fixture(scope="module")
def flux2_klein() -> Any:
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())
    flux.freeze()
    return flux


# ---------------------------------------------------------------------------
# Image-quality tests
# ---------------------------------------------------------------------------


def test_default_threshold_ssim_klein_pr_gate(flux2_klein: Any) -> None:
    """PR-time gate: wrapper at the package-default rel_l1_thresh must
    produce a decoded image whose SSIM vs same-process vanilla is >= the gate."""
    kw = _gen_kwargs_klein(PR_TIME_PROMPT)
    vanilla_latent = _capture(flux2_klein, **kw)
    with apply_teacache(flux2_klein) as h:  # uses package default rel_l1_thresh
        wrapper_latent = _capture(flux2_klein, **kw)
        skipped = h.stats.skipped_count
    # Note: at num_inference_steps=8 with default skip windows there may be
    # very few eligible steps; skipped_count > 0 is not guaranteed. Don't
    # assert on it here; the SSIM gate is the actual quality test.
    del skipped  # explicitly unused

    vanilla_img = _decode_to_uint8(
        flux2_klein,
        vanilla_latent,
        height=kw["height"],
        width=kw["width"],
    )
    wrapper_img = _decode_to_uint8(
        flux2_klein,
        wrapper_latent,
        height=kw["height"],
        width=kw["width"],
    )
    score = ssim(vanilla_img, wrapper_img, channel_axis=-1, data_range=255)
    assert score >= _DEFAULT_THRESHOLD_SSIM, (
        f"SSIM {score:.4f} < {_DEFAULT_THRESHOLD_SSIM}; wrapper image "
        f"diverged from same-process vanilla baseline at default threshold"
    )


@pytest.mark.slow
@pytest.mark.parametrize("prompt", REFERENCE_PROMPTS)
def test_default_threshold_ssim_klein_full(flux2_klein: Any, prompt: str) -> None:
    """Nightly image-quality gate: all 5 reference prompts."""
    kw = _gen_kwargs_klein(prompt)
    vanilla_latent = _capture(flux2_klein, **kw)
    with apply_teacache(flux2_klein):  # uses package default rel_l1_thresh
        wrapper_latent = _capture(flux2_klein, **kw)
    vanilla_img = _decode_to_uint8(flux2_klein, vanilla_latent, height=kw["height"], width=kw["width"])
    wrapper_img = _decode_to_uint8(flux2_klein, wrapper_latent, height=kw["height"], width=kw["width"])
    score = ssim(vanilla_img, wrapper_img, channel_axis=-1, data_range=255)
    assert score >= _DEFAULT_THRESHOLD_SSIM, (
        f"SSIM {score:.4f} < {_DEFAULT_THRESHOLD_SSIM} on prompt {prompt!r}"
    )
