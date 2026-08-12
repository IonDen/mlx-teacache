"""FLUX.2 Klein image-quality parity — SSIM on VAE-decoded images.

Mirrors `test_image_quality_flux1.py` (v2.6) for FLUX.2. At
`rel_l1_thresh=0.0` the wrapper is already covered by the latent paired
parity in `test_parity_flux2.py`; this file focuses on default-threshold
quality (with caching engaged the decoded image should remain
perceptually close to same-process vanilla).

PR-gate matrix
--------------
Mode        | base-4b / base-9b            | distilled 4b / 9b
txt2img     | skip_count >= 1 + SSIM gate  | finiteness/shape only
img2img     | finiteness/shape only        | finiteness/shape only

Rationale for distilled txt2img: the gate premise (polynomial predicts body
residual distance) does not hold on 8-step distilled schedules — 0 skips
by design at the package default. An SSIM gate without a skip-count
assertion is the dormant-cache failure mode (passes green even when caching
is completely disabled). Finiteness + shape is the honest correctness gate
for those rows.

Rationale for img2img (all variants): the pre-existing 0.85 floor was an
unmeasured placeholder; a calibrated img2img gate needs its own dedicated
measurement.
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
from tests.conftest import expect_distilled_warning  # noqa: E402

pytestmark = pytest.mark.parity


REFERENCE_PROMPTS = (
    "a red apple on a wooden table",
    "mountain landscape at sunset",
    "portrait of a woman",
    "abstract pattern with circles",
    "text saying HELLO",
)
PR_TIME_PROMPT = "a red apple on a wooden table"

# ---------------------------------------------------------------------------
# Per-recipe SSIM floors.
#
# The base txt2img values are measured by the v0.8.0 release run; they are
# committed BELOW the measured value with a small headroom margin. The
# vanilla-vs-wrapper SSIM is deterministic per mflux/mlx pin but not a
# portable constant across hardware.
#
# Constants marked `None` are MEASURE-ME sentinels. Any test that reaches
# one at run time calls pytest.fail() loudly — placeholders must not
# silently pass.
# ---------------------------------------------------------------------------
# Measured 2026-06-09 (v0.8.0 release run, one pytest process per variant):
# red-apple prompt, seed 42, 25 steps, 512x512, guidance 1.0, q4, M1 Max 32GB,
# mflux 0.17.x / mlx pinned by uv.lock. base-4b: 3 skips, SSIM 0.9927;
# base-9b: 7 skips, SSIM 0.9920. Floors committed below measured with headroom.
_SSIM_BASE_4B_TXT2IMG: float | None = 0.95  # measured 0.9927
_SSIM_BASE_9B_TXT2IMG: float | None = 0.95  # measured 0.9920
_SSIM_SLOW_FLOOR = 0.80  # documented slow-suite variance floor (5-prompt suite)
_SSIM_CFG_BASE_4B = 0.85  # pre-existing CFG PR-gate floor (unchanged)
_SSIM_CFG_BASE_9B = 0.95  # _artifacts/validation_klein_base_9b.json: 0.986 measured, 12 skips


def _require_ssim(constant: float | None, label: str) -> float:
    """Return the constant or fail loudly if it is an unfilled sentinel."""
    if constant is None:
        pytest.fail(
            f"SSIM floor {label!r} has not been measured yet. "
            "Run the v0.8.0 heavy measurement (Step 2) and fill in the constant."
        )
    return constant  # type: ignore[return-value]  # pytest.fail() raises, never returns


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


_DISTILLED_VARIANTS = frozenset({"flux2-klein-4b", "flux2-klein-9b"})
_BASE_VARIANTS = frozenset({"flux2-klein-base-4b", "flux2-klein-base-9b"})

_BASE_TXT2IMG_SSIM: dict[str, float | None] = {
    "flux2-klein-base-4b": _SSIM_BASE_4B_TXT2IMG,
    "flux2-klein-base-9b": _SSIM_BASE_9B_TXT2IMG,
}


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
        pytest.param(0.0, None, id="txt2img"),
        pytest.param(0.5, "natural_512.png", id="img2img-s0.5"),
    ],
)
def test_default_threshold_ssim_klein_pr_gate(
    flux2_klein: tuple[Any, str], image_strength: float, init_image_name: str | None
) -> None:
    """PR-time gate: wrapper at the package-default rel_l1_thresh must
    produce a decoded image that passes the per-variant/mode quality bar.

    base variants + txt2img: skip_count >= 1 (proves cache engagement) +
        SSIM >= measured per-variant floor.
    distilled variants + txt2img: finiteness + shape only — the gate premise
        does not hold on 8-step distilled schedules; 0 skips by design.
    any variant + img2img: finiteness + shape only — the pre-existing 0.85
        was an unmeasured placeholder; img2img needs its own calibrated gate.
    """
    flux, variant_id = flux2_klein
    kw = _gen_kwargs_klein(PR_TIME_PROMPT, variant_id=variant_id)
    if init_image_name is not None:
        kw["image_path"] = str(Path(__file__).parent / "fixtures" / "init_images" / init_image_name)
        kw["image_strength"] = image_strength

    is_txt2img = image_strength == 0.0
    is_base = variant_id in _BASE_VARIANTS

    if is_txt2img and is_base:
        # Full gate: skip engagement + SSIM floor.
        vanilla_latent = _capture(flux, **kw)
        with apply_teacache(flux) as h:  # uses package default rel_l1_thresh
            wrapper_latent = _capture(flux, **kw)
            assert h.stats.skipped_count >= 1, (
                f"Expected >= 1 skip for {variant_id} txt2img at default threshold; "
                f"got {h.stats.skipped_count}. Cache is not engaging — check coefficients."
            )
        vanilla_img = _decode_to_uint8(flux, vanilla_latent, height=kw["height"], width=kw["width"])
        wrapper_img = _decode_to_uint8(flux, wrapper_latent, height=kw["height"], width=kw["width"])
        score = ssim(vanilla_img, wrapper_img, channel_axis=-1, data_range=255)
        # Emitted before the floor lookup so a measurement run (-s) records the
        # value even while the floor constant is still the unfilled sentinel.
        print(f"::SSIM_MEASURE:: {variant_id} txt2img skipped={h.stats.skipped_count} ssim={score:.4f}")
        floor = _require_ssim(
            _BASE_TXT2IMG_SSIM[variant_id], f"_SSIM_{variant_id.upper().replace('-', '_')}_TXT2IMG"
        )
        assert score >= floor, (
            f"SSIM {score:.4f} < {floor}; wrapper image diverged from "
            f"same-process vanilla baseline for {variant_id} txt2img"
        )
    else:
        # Finiteness + shape only.
        # distilled txt2img: gate premise does not hold on 8-step distilled
        #   schedules; 0 skips by design at the package default threshold.
        # img2img (any variant): unmeasured placeholder removed; a dedicated
        #   img2img gate needs its own calibrated measurement.
        vanilla_latent = _capture(flux, **kw)
        # uses package default rel_l1_thresh; distilled variants warn at apply time.
        with expect_distilled_warning(variant_id), apply_teacache(flux):
            wrapper_latent = _capture(flux, **kw)
        arr = np.asarray(wrapper_latent.astype(mx.float32))
        assert np.isfinite(arr).all(), f"wrapper latent contains non-finite values for {variant_id}"
        assert arr.shape == np.asarray(vanilla_latent.astype(mx.float32)).shape, (
            f"wrapper latent shape {arr.shape} != vanilla shape for {variant_id}"
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
    assert score >= _SSIM_CFG_BASE_4B, (
        f"SSIM {score:.4f} < {_SSIM_CFG_BASE_4B}; wrapper image "
        f"diverged from same-process vanilla baseline under CFG "
        f"(guidance=4.0, num_inference_steps=50)"
    )


def test_ssim_pr_gate_cfg_klein_base_9b() -> None:
    """CFG gate for flux2-klein-base-9b: at default rel_l1_thresh (0.17),
    g=4.0 / 50 steps must produce SSIM >= 0.95 vs vanilla AND fire >= 1
    skip.

    Threshold source: _artifacts/validation_klein_base_9b.json —
    measured SSIM 0.986, 12 skips at rel_l1_thresh=0.17. Floor is set
    at 0.95 to accommodate per-machine numerical variance while still
    catching regressions.
    """
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_9b())
    flux.freeze()
    kw = _gen_kwargs_klein(PR_TIME_PROMPT, variant_id="flux2-klein-base-9b", cfg=True)
    vanilla_latent = _capture(flux, **kw)
    with apply_teacache(flux) as h:  # uses per-variant default rel_l1_thresh=0.17
        wrapped_latent = _capture(flux, **kw)
        assert h.stats.skipped_count >= 1, (
            f"Expected >=1 skip for base-9b under CFG; got {h.stats.skipped_count}. "
            f"If this fires reliably, run CFG-aware calibration via "
            f"scripts/calibrate_flux2.py --variant klein-base-9b --guidance 4.0 "
            f"--num-inference-steps 50."
        )
    vanilla_img = _decode_to_uint8(flux, vanilla_latent, height=kw["height"], width=kw["width"])
    wrapped_img = _decode_to_uint8(flux, wrapped_latent, height=kw["height"], width=kw["width"])
    score = ssim(vanilla_img, wrapped_img, channel_axis=-1, data_range=255)
    assert score >= _SSIM_CFG_BASE_9B, (
        f"SSIM {score:.4f} < {_SSIM_CFG_BASE_9B}; wrapper image "
        f"diverged from same-process vanilla baseline under CFG for base-9b "
        f"(guidance=4.0, num_inference_steps=50)"
    )


@pytest.mark.slow
@pytest.mark.parametrize("prompt", REFERENCE_PROMPTS)
def test_default_threshold_ssim_klein_full(flux2_klein: tuple[Any, str], prompt: str) -> None:
    """Nightly image-quality gate: all 5 reference prompts.

    base variants: skip_count >= 1 (proves cache engagement) + SSIM >= slow-
        suite variance floor (0.80).
    distilled variants: finiteness only — gate premise does not hold on
        8-step distilled schedules; 0 skips by design at the package default.
    """
    flux, variant_id = flux2_klein
    kw = _gen_kwargs_klein(prompt, variant_id=variant_id)
    is_base = variant_id in _BASE_VARIANTS

    if is_base:
        vanilla_latent = _capture(flux, **kw)
        with apply_teacache(flux) as h:  # uses package default rel_l1_thresh
            wrapper_latent = _capture(flux, **kw)
            assert h.stats.skipped_count >= 1, (
                f"Expected >= 1 skip for {variant_id} slow suite at default threshold; "
                f"got {h.stats.skipped_count}."
            )
        vanilla_img = _decode_to_uint8(flux, vanilla_latent, height=kw["height"], width=kw["width"])
        wrapper_img = _decode_to_uint8(flux, wrapper_latent, height=kw["height"], width=kw["width"])
        score = ssim(vanilla_img, wrapper_img, channel_axis=-1, data_range=255)
        assert score >= _SSIM_SLOW_FLOOR, (
            f"SSIM {score:.4f} < {_SSIM_SLOW_FLOOR} on prompt {prompt!r} for {variant_id}"
        )
    else:
        # distilled: gate premise does not hold on 8-step distilled schedules;
        # finiteness is the honest correctness gate. Always distilled here
        # (the `is_base` branch above handles the base variants), so the
        # apply-time warning always fires.
        vanilla_latent = _capture(flux, **kw)
        with expect_distilled_warning(variant_id), apply_teacache(flux):
            wrapper_latent = _capture(flux, **kw)
        arr = np.asarray(wrapper_latent.astype(mx.float32))
        assert np.isfinite(arr).all(), (
            f"wrapper latent contains non-finite values for {variant_id} on prompt {prompt!r}"
        )
