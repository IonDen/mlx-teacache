"""Z-Image base parity tests — paired same-process methodology.

Mirrors test_parity_flux2.py's CFG release-blocker pattern. Z-Image's vanilla
`_predict` is `mx.compile`-wrapped except on base and Pro M1/M2, where mflux
returns the eager function (AppleSiliconUtil.is_m1_or_m2, which excludes Max
and Ultra; unchanged from 0.17.5 through 0.19.x); our TeaCache integration
replaces `_predict` with an eager-Python wrapper that re-walks
ZImageTransformer.__call__ so the per-step gate runs every step. We gate threshold=0 parity with cosine
similarity (not bit-exact): the calibration self-check measured cos >= 0.999
for the re-walk vs `transformer(...)`, so 0.99 here absorbs prompt-to-prompt
dispatch noise with margin.

Correctness only — a short 8-step schedule exercises the prelude / layer-0
gate signal / 30-layer body / residual cache / CFG-combine / tail paths. The
speedup + image-quality validation use the pinned 50-step recipe in the
Phase-4 bench + SSIM gates, not here.
"""

from __future__ import annotations

import warnings as _w
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from mlx_teacache import apply_teacache
from mlx_teacache.errors import TeaCacheDisabledWarning

# PIL + skimage live in the [mflux]/test extra and are absent in the pure-core CI
# env. importorskip skips this module cleanly at collection there (it is parity-only
# anyway) instead of erroring the whole pure-core lane on a module-top import.
Image = pytest.importorskip("PIL.Image")
ssim = pytest.importorskip("skimage.metrics").structural_similarity

pytestmark = pytest.mark.parity

# threshold=0 parity gate. Calibration self-check measured cos >= 0.999 on the
# re-walk; 0.99 absorbs prompt variance. Tighten toward the measured min if the
# parity run shows consistently higher.
_ZIMAGE_COSINE_GATE = 0.99

# User-facing quality gate at the committed sweep recipe (512x512, q8, 50 steps,
# guidance=4.0) + the shipped DEFAULT_THRESH=0.12. Locked from the v0.10.0 validation
# run under consecutive-delta anchoring (2026-08-15): this test measured SSIM 0.9913
# (unchanged from the pre-anchoring sweep, tests/_artifacts/sweep_z_image/results_z_image.json),
# and the same recipe in the committed bench _artifacts/v0.10.0_bench_z_image.json skipped
# 15/48 active steps in every rep (max streak 1). The floor leaves ~0.02 of headroom for
# cross-machine Metal variance; the band brackets the observed 15 by +-5 — a dormant
# cache (0) or a runaway one (>20) both trip it.
_SSIM_FLOOR = 0.97
_SKIP_BAND = (10, 20)
_SSIM_STEPS = 50
_ARTIFACTS = Path(__file__).parent / "_artifacts" / "parity_z_image"

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


def _gen_image_array(flux: Any, *, save_path: Path, steps: int) -> np.ndarray:
    image = flux.generate_image(
        prompt=PROMPT, seed=42, num_inference_steps=steps, height=512, width=512, guidance=4.0
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        save_path.unlink()  # mflux image.save() appends _1 on collision instead of overwriting
    image.save(path=str(save_path), export_json_metadata=False)
    return np.array(Image.open(save_path).convert("RGB"), dtype=np.uint8)


@pytest.fixture(scope="module")
def zimage_base() -> Any:
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.z_image.variants.z_image import ZImage

    flux = ZImage(quantize=8, model_config=ModelConfig.z_image())
    flux.freeze()
    # mflux builds RopeEmbedder.freqs_cis as lazy arrays it never evaluates. A
    # compiled vanilla run traces that pending graph into the compiled function;
    # the first eager pass through the transformer (ours, or any other) then
    # materialises the tables with the eager kernels, and every later compiled
    # run captures those values instead. Measured on M1 Max, 2026-09-06, with
    # MLX 0.32.2 (mflux 0.19.1): a bare mx.eval of the tables with no TeaCache
    # in the process moves the next compiled vanilla latent by max-abs 6.25e-2
    # (cosine 0.999977), the exact shift the restore-trace assertions below
    # saw. With MLX 0.31.2 (mflux 0.18.0 or 0.19.1) the same eval changes
    # nothing, so it is MLX 0.32's compile that inlines the pending graph.
    # Evaluating the tables once here keeps vanilla bit-repeatable so those
    # assertions test restore() and nothing else.
    mx.eval(*flux.transformer.rope_embedder.freqs_cis)
    return flux


def test_cfg_parity_at_threshold_zero(zimage_base: Any) -> None:
    """Release blocker: at rel_l1_thresh=0 the gated CFG path (re-walk of
    ZImageTransformer.__call__ + CFG combine `pos + g*(pos-neg)`) must match
    real mflux generation within Metal noise, and never skip."""
    flux = zimage_base

    vanilla_before = _capture(flux, **_GEN_KW)
    with _w.catch_warnings():
        _w.simplefilter("ignore", TeaCacheDisabledWarning)
        ctx = apply_teacache(flux, rel_l1_thresh=0.0)
    with ctx as h:
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


def test_image_quality_ssim_at_default_threshold(zimage_base: Any) -> None:
    """User-facing quality gate at the committed sweep recipe (512x512, q8, 50
    steps, guidance=4.0) + the shipped DEFAULT_THRESH (0.12): caching MUST engage
    (skipped > 0, within the expected band — the dormant/runaway-cache guard) and
    SSIM vs vanilla MUST hold the PR-gate floor. Images saved for manual inspection.
    z-image-base is not distilled (DEFAULT_THRESH=0.12, not None), so apply_teacache
    here must NOT raise TeaCacheNoBenefitWarning."""
    flux = zimage_base
    van = _gen_image_array(flux, save_path=_ARTIFACTS / "vanilla.png", steps=_SSIM_STEPS)
    with apply_teacache(flux) as h:  # builtin DEFAULT_THRESH (0.12)
        wrap = _gen_image_array(flux, save_path=_ARTIFACTS / "wrapper_default_thresh.png", steps=_SSIM_STEPS)
        skipped, computed = h.stats.skipped_count, h.stats.computed_count
    score = float(ssim(van, wrap, channel_axis=-1, data_range=255))
    assert _SKIP_BAND[0] <= skipped <= _SKIP_BAND[1], (
        f"skip count {skipped} outside the expected band {_SKIP_BAND} at DEFAULT_THRESH "
        f"(computed={computed}) — a dormant or runaway cache"
    )
    assert score >= _SSIM_FLOOR, f"SSIM {score:.4f} < {_SSIM_FLOOR} at DEFAULT_THRESH (skipped={skipped})"
