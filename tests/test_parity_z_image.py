"""Z-Image base parity tests — paired same-process methodology.

Mirrors test_parity_flux2.py's CFG release-blocker pattern. Z-Image's vanilla
`_predict` is `mx.compile`-wrapped on M3+ (eager on M1/M2 via
AppleSiliconUtil.is_m1_or_m2); our TeaCache integration replaces `_predict`
with an eager-Python wrapper that re-walks ZImageTransformer.__call__ so the
per-step gate runs every step. We gate threshold=0 parity with cosine
similarity (not bit-exact): the calibration self-check measured cos >= 0.999
for the re-walk vs `transformer(...)`, so 0.99 here absorbs prompt-to-prompt
dispatch noise with margin.

Correctness only — a short 8-step schedule exercises the prelude / layer-0
gate signal / 30-layer body / residual cache / CFG-combine / tail paths. The
speedup + image-quality validation use the pinned 50-step recipe in the
Phase-4 bench + SSIM gates, not here.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest

from mlx_teacache import apply_teacache

pytestmark = pytest.mark.parity

# threshold=0 parity gate. Calibration self-check measured cos >= 0.999 on the
# re-walk; 0.99 absorbs prompt variance. Tighten toward the measured min if the
# parity run shows consistently higher.
_ZIMAGE_COSINE_GATE = 0.99

PROMPT = "a red apple on a wooden table"
_GEN_KW: dict[str, Any] = {
    "prompt": PROMPT,
    "seed": 42,
    "num_inference_steps": 8,  # correctness schedule; 50-step is the bench recipe
    "height": 512,
    "width": 512,
    "guidance": 4.0,  # CFG active — the primary Z-Image path
}


class _LatentCapture:
    """Captures the pre-VAE latent via mflux's after_loop callback."""

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


def _cosine(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    return float(mx.sum(af * bf) / (mx.linalg.norm(af) * mx.linalg.norm(bf)))


@pytest.fixture(scope="module")
def zimage_base() -> Any:
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.z_image.variants.z_image import ZImage

    flux = ZImage(quantize=8, model_config=ModelConfig.z_image())
    flux.freeze()
    return flux


def test_cfg_parity_at_threshold_zero(zimage_base: Any) -> None:
    """Release blocker: at rel_l1_thresh=0 the gated CFG path (re-walk of
    ZImageTransformer.__call__ + CFG combine `pos + g*(pos-neg)`) must match
    real mflux generation within Metal noise, and never skip."""
    flux = zimage_base

    vanilla_before = _capture(flux, **_GEN_KW)
    with apply_teacache(flux, rel_l1_thresh=0.0) as h:
        wrapper = _capture(flux, **_GEN_KW)
        skipped = h.stats.skipped_count
    vanilla_after = _capture(flux, **_GEN_KW)

    cos = _cosine(vanilla_before, wrapper)
    assert cos >= _ZIMAGE_COSINE_GATE, (
        f"CFG wrapper at rel_l1_thresh=0, guidance=4.0 cosine vs same-process vanilla "
        f"= {cos:.6f} < {_ZIMAGE_COSINE_GATE}; max_abs_diff="
        f"{float(mx.max(mx.abs(vanilla_before - wrapper))):.4e}"
    )
    assert skipped == 0, f"threshold=0 must never skip; got skipped={skipped}"
    # restore() must leave no trace — both vanilla runs hit the same vanilla path.
    assert mx.array_equal(vanilla_before, vanilla_after), (
        "restore() left a trace; vanilla_after differs from vanilla_before"
    )


def test_skip_path_engages_and_produces_finite_latent(zimage_base: Any) -> None:
    """Mechanical correctness of the skip path (schedule-robust): a threshold
    high enough to force skips must (a) actually skip >=1 step (the gate fires),
    and (b) reconstruct a FINITE latent via `main_out = unified_in +
    cached_residual` per CFG branch (no NaN/inf from a shape/broadcast bug).

    Deliberately NO cosine/quality bound here: the coefficients are calibrated
    at 50 steps, so on this short 8-step schedule the per-step rel-L1 is much
    larger and the polynomial extrapolates out-of-range → the gate over-skips
    (rel_l1_thresh=0.5 skips ~5/8). 8-step cosine is therefore an invalid
    quality oracle for 50-step-calibrated coefficients. Skip QUALITY is gated at
    the pinned 50-step recipe: scripts/sweep_threshold_z_image.py (SSIM knee) +
    the Phase-4 image-quality SSIM test. The threshold=0 cosine parity above is
    the compute-path correctness gate."""
    flux = zimage_base

    with apply_teacache(flux, rel_l1_thresh=0.5) as h:
        wrapper = _capture(flux, **_GEN_KW)
        skipped = h.stats.skipped_count

    assert skipped >= 1, f"rel_l1_thresh=0.5 on 8 steps should skip >=1 step; got {skipped}"
    assert bool(mx.all(mx.isfinite(wrapper))), "skipped-step reconstruction produced non-finite latent"
