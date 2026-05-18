"""FLUX.2 Klein 4b image-quality parity — SSIM on VAE-decoded images.

Mirrors `test_image_quality_flux1.py` (v2.6) for FLUX.2. At
`rel_l1_thresh=0.0` the wrapper is already covered by the latent paired
parity in `test_parity_flux2.py`; this file focuses on default-threshold
quality (with caching engaged the decoded image should remain
perceptually close to same-process vanilla).
"""

from __future__ import annotations

from pathlib import Path
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


def _gen_kwargs_klein(
    prompt: str, *, variant_id: str = "flux2-klein-4b", guidance: float = 1.0, cfg: bool = False
) -> dict[str, Any]:
    """Generation kwargs for FLUX.2 Klein variants.

    Distilled Klein 4B / 9B use the 8-step default schedule (matches their
    runtime usage). base-4b uses the calibration-time 25-step schedule.
    cfg=True selects the upstream CFG recipe on base-4b: guidance=4.0 at
    the calibrated 50-step schedule. All other variants are unchanged."""
    if variant_id in ("flux2-klein-base-4b", "flux2-klein-base-9b") and cfg:
        num_inference_steps = 50
        guidance = 4.0
    elif variant_id in ("flux2-klein-4b", "flux2-klein-9b"):
        num_inference_steps = 8
    elif variant_id in ("flux2-klein-base-4b", "flux2-klein-base-9b"):
        num_inference_steps = 25
    else:
        raise ValueError(f"unsupported variant_id for _gen_kwargs_klein: {variant_id!r}")
    return {
        "prompt": prompt,
        "seed": 42,
        "num_inference_steps": num_inference_steps,
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


@pytest.fixture(
    scope="module",
    params=[
        "flux2-klein-4b",
        "flux2-klein-9b",
        "flux2-klein-base-4b",
        "flux2-klein-base-9b",
    ],
)
def flux2_klein(request) -> tuple[Any, str]:
    """Returns (flux instance, variant_id) so tests can pass variant_id to _gen_kwargs_klein."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    variant_id = request.param
    if variant_id == "flux2-klein-4b":
        cfg = ModelConfig.flux2_klein_4b()
    elif variant_id == "flux2-klein-9b":
        cfg = ModelConfig.flux2_klein_9b()
    elif variant_id == "flux2-klein-base-4b":
        cfg = ModelConfig.flux2_klein_base_4b()
    elif variant_id == "flux2-klein-base-9b":
        cfg = ModelConfig.flux2_klein_base_9b()
    else:
        pytest.fail(f"unhandled variant_id={variant_id!r}")
    flux = Flux2Klein(quantize=4, model_config=cfg)
    flux.freeze()
    return flux, variant_id


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
def test_default_threshold_ssim_klein_pr_gate(
    flux2_klein: tuple[Any, str], image_strength: float, init_image_name: str | None
) -> None:
    """PR-time gate: wrapper at the package-default rel_l1_thresh must
    produce a decoded image whose SSIM vs same-process vanilla is >= the gate."""
    flux, variant_id = flux2_klein
    kw = _gen_kwargs_klein(PR_TIME_PROMPT, variant_id=variant_id)
    if init_image_name is not None:
        kw["image_path"] = str(Path(__file__).parent / "fixtures" / "init_images" / init_image_name)
        kw["image_strength"] = image_strength

    vanilla_latent = _capture(flux, **kw)
    with apply_teacache(flux) as h:  # uses package default rel_l1_thresh
        wrapper_latent = _capture(flux, **kw)
        skipped = h.stats.skipped_count
    # Note: at num_inference_steps=8 with default skip windows there may be
    # very few eligible steps; skipped_count > 0 is not guaranteed. Don't
    # assert on it here; the SSIM gate is the actual quality test.
    del skipped  # explicitly unused

    vanilla_img = _decode_to_uint8(
        flux,
        vanilla_latent,
        height=kw["height"],
        width=kw["width"],
    )
    wrapper_img = _decode_to_uint8(
        flux,
        wrapper_latent,
        height=kw["height"],
        width=kw["width"],
    )
    score = ssim(vanilla_img, wrapper_img, channel_axis=-1, data_range=255)
    assert score >= _DEFAULT_THRESHOLD_SSIM, (
        f"SSIM {score:.4f} < {_DEFAULT_THRESHOLD_SSIM}; wrapper image "
        f"diverged from same-process vanilla baseline at default threshold "
        f"(image_strength={image_strength})"
    )


def test_ssim_pr_gate_cfg_klein_base_4b() -> None:
    """v0.4.1 release blocker: at default rel_l1_thresh (0.17), CFG-engaged
    generation on flux2-klein-base-4b at g=4.0/50 steps must produce SSIM
    >= 0.85 vs vanilla AND fire >= 1 skip. Skip-count assertion locks in
    the v0.4.1 engagement claim — without it the test would pass with 0
    skips and the feature would be dormant (v0.3 postmortem lesson)."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
    flux.freeze()
    kw = _gen_kwargs_klein(PR_TIME_PROMPT, variant_id="flux2-klein-base-4b", cfg=True)
    vanilla_latent = _capture(flux, **kw)
    with apply_teacache(flux) as h:  # uses per-variant default rel_l1_thresh=0.17
        wrapped_latent = _capture(flux, **kw)
        assert h.stats.skipped_count >= 1, (
            f"Expected >=1 skip under CFG; got {h.stats.skipped_count}. "
            f"If this fires reliably, fall into the 0-skip contingency: "
            f"run CFG-aware calibration via scripts/calibrate_flux2.py "
            f"--guidance 4.0 --num-inference-steps 50."
        )
    vanilla_img = _decode_to_uint8(flux, vanilla_latent, height=kw["height"], width=kw["width"])
    wrapped_img = _decode_to_uint8(flux, wrapped_latent, height=kw["height"], width=kw["width"])
    score = ssim(vanilla_img, wrapped_img, channel_axis=-1, data_range=255)
    assert score >= _DEFAULT_THRESHOLD_SSIM, (
        f"SSIM {score:.4f} < {_DEFAULT_THRESHOLD_SSIM}; wrapper image "
        f"diverged from same-process vanilla baseline under CFG "
        f"(guidance=4.0, num_inference_steps=50)"
    )


@pytest.mark.slow
@pytest.mark.parametrize("prompt", REFERENCE_PROMPTS)
def test_default_threshold_ssim_klein_full(flux2_klein: tuple[Any, str], prompt: str) -> None:
    """Nightly image-quality gate: all 5 reference prompts."""
    flux, variant_id = flux2_klein
    kw = _gen_kwargs_klein(prompt, variant_id=variant_id)
    vanilla_latent = _capture(flux, **kw)
    with apply_teacache(flux):  # uses package default rel_l1_thresh
        wrapper_latent = _capture(flux, **kw)
    vanilla_img = _decode_to_uint8(flux, vanilla_latent, height=kw["height"], width=kw["width"])
    wrapper_img = _decode_to_uint8(flux, wrapper_latent, height=kw["height"], width=kw["width"])
    score = ssim(vanilla_img, wrapper_img, channel_axis=-1, data_range=255)
    assert score >= _DEFAULT_THRESHOLD_SSIM, (
        f"SSIM {score:.4f} < {_DEFAULT_THRESHOLD_SSIM} on prompt {prompt!r}"
    )
