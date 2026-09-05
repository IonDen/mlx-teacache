"""FLUX.1 Krea [dev] parity tests — paired same-process methodology, the same
protocol as tests/test_parity_flux1.py (see that module's docstring for why a
committed latent is not the oracle). Krea shares the FLUX.1 transformer and the
FLUX.1 proxy strategy, so the threshold-zero gate must reproduce vanilla math
exactly and restore() must leave no trace.

Gated weights: black-forest-labs/FLUX.1-Krea-dev needs an accepted licence and
an HF token. Recipe: the model card's 28 steps / guidance 4.5, 512x512, q4.
"""

from __future__ import annotations

import warnings as _w
from typing import Any

import mlx.core as mx
import pytest

from mlx_teacache import apply_teacache
from mlx_teacache.errors import TeaCacheDisabledWarning

pytestmark = pytest.mark.parity

PR_TIME_PROMPT = "a red apple on a wooden table"
_COSINE_GATE = 0.95  # same conservative bar as FLUX.1-dev


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
        raise RuntimeError("call_after_loop never fired — latent capture failed")
    return cap.latent


def _gen_kwargs(prompt: str) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "seed": 42,
        "num_inference_steps": 28,
        "height": 512,
        "width": 512,
        "guidance": 4.5,
    }


def _cosine(a: mx.array, b: mx.array) -> float:
    return float(mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b)))


@pytest.fixture(scope="module")
def krea_dev() -> Any:
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux.variants.txt2img.flux import Flux1

    flux = Flux1(quantize=4, model_config=ModelConfig.krea_dev())
    flux.freeze()
    return flux


def test_paired_parity_krea_pr_gate(krea_dev: Any) -> None:
    """Threshold-zero wrapper == same-process vanilla, and restore() leaves no trace."""
    kw = _gen_kwargs(PR_TIME_PROMPT)
    vanilla_before = _capture(krea_dev, **kw)
    with _w.catch_warnings():
        _w.simplefilter("ignore", TeaCacheDisabledWarning)
        ctx = apply_teacache(krea_dev, rel_l1_thresh=0.0)
    with ctx as h:
        wrapper = _capture(krea_dev, **kw)
        skipped = h.stats.skipped_count
    vanilla_after = _capture(krea_dev, **kw)
    assert mx.array_equal(vanilla_before, wrapper), (
        "wrapper at rel_l1_thresh=0 must match same-process vanilla"
    )
    assert mx.array_equal(vanilla_before, vanilla_after), "restore() left a trace"
    assert skipped == 0


def test_default_threshold_engages_and_stays_close_to_vanilla(krea_dev: Any) -> None:
    """At the shipped default the gate must skip at least one step and the latent
    must stay within the cosine gate of a same-process vanilla baseline."""
    kw = _gen_kwargs(PR_TIME_PROMPT)
    vanilla = _capture(krea_dev, **kw)
    with apply_teacache(krea_dev) as h:
        wrapper = _capture(krea_dev, **kw)
        skipped = h.stats.skipped_count
    assert skipped >= 1, "the shipped default should skip at least one step"
    cos = _cosine(wrapper, vanilla)
    assert cos >= _COSINE_GATE, f"cosine {cos:.4f} < {_COSINE_GATE}"


_SKIP_BAND = (9, 11)  # sweep 2026-09-05: 10 of the 26 active steps at the 0.30 default
_SSIM_FLOOR = 0.95  # measured 0.990 on this recipe; 0.35 already falls to 0.890


def test_krea_default_threshold_skip_count_band(krea_dev: Any) -> None:
    """The threshold sweep put the knee at 0.30 with 10 skips on the red-apple
    recipe; a regression to 0 (dormant) or to 13+ (0.40's operating point, under
    the SSIM bar) must go red here."""
    kw = _gen_kwargs(PR_TIME_PROMPT)
    with apply_teacache(krea_dev) as h:
        _capture(krea_dev, **kw)
        skipped = h.stats.skipped_count
    assert _SKIP_BAND[0] <= skipped <= _SKIP_BAND[1], f"skip count {skipped} outside {_SKIP_BAND}"


def test_krea_image_quality_ssim_at_default_threshold(krea_dev: Any, tmp_path: Any) -> None:
    """User-facing quality gate at the shipped default on the sweep recipe."""
    import numpy as np
    from skimage.metrics import structural_similarity as ssim

    def _image(path: Any, **extra: Any) -> Any:
        img = krea_dev.generate_image(**_gen_kwargs(PR_TIME_PROMPT))
        img.save(path=str(path), export_json_metadata=False)
        from PIL import Image

        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    vanilla = _image(tmp_path / "vanilla.png")
    with apply_teacache(krea_dev) as h:
        wrapper = _image(tmp_path / "wrapper.png")
        skipped = h.stats.skipped_count
    score = float(ssim(vanilla, wrapper, channel_axis=-1, data_range=255))
    assert skipped >= 1
    assert score >= _SSIM_FLOOR, f"SSIM {score:.4f} < {_SSIM_FLOOR} at the default (skipped={skipped})"
