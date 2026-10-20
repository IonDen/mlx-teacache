"""Qwen-Image parity tests — paired same-process methodology.

Mirrors test_parity_z_image.py. Qwen-Image has no `_predict` / `mx.compile`; the
TeaCache integration proxies `flux.transformer` and re-walks
QwenTransformer.__call__ so the per-step CFG gate runs every step. threshold=0
parity is cosine (not bit-exact): the calibration self-check measured cos >=
0.999 for the re-walk vs the unwrapped transformer, so 0.99 here absorbs
prompt-to-prompt Metal-dispatch noise with margin.

Three gates:
  - threshold=0 latent cosine + restore-no-trace  (compute-path correctness)
  - SSIM at the shipped DEFAULT_THRESH on the calibrated 50-step recipe, with a
    skip-engagement assert                          (the user-facing quality gate)
  - high-threshold skip path produces a finite latent (skip-reconstruction)

Generated images are written under tests/_artifacts/parity_qwen/ (gitignored) for
manual inspection — they are NOT deleted.

Real weights — pytest.mark.parity (deselected from the per-PR lanes). 20B near the
32 GB ceiling: the fixture sets a device-derived wired cap, registers mflux's
MemorySaver (frees the text encoders once the prompt is encoded, the same way the
bench and sweep workers do) and mx.clear_cache() runs between tests to keep the peak
~one generation.

Every gate runs once per checkpoint, ids ``original`` (Qwen/Qwen-Image) and ``2512``
(Qwen/Qwen-Image-2512), each against its own variant's DEFAULT_THRESH. Run ONE id per
pytest process (``-k original`` or ``-k 2512``): the repo rule for parity files is one
model per process, and two 20B checkpoints in one session are an OOM trap (the session hook in
conftest.py refuses one that selects both). The fixture releases its model's weights at teardown,
but the allocator pool and the process footprint are not something to rely on.
"""

import sys
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from mlx_teacache import apply_teacache
from mlx_teacache.errors import TeaCacheDisabledWarning
from mlx_teacache.variants.qwen_image import config as _original_config
from mlx_teacache.variants.qwen_image_2512 import config as _config_2512
from tests._memory_guard import release_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# PIL + skimage live in the [mflux]/test extra and are absent in the pure-core CI
# env. importorskip skips this module cleanly at collection there (it is parity-only
# anyway) instead of erroring the whole pure-core lane on a module-top import.
Image = pytest.importorskip("PIL.Image")
ssim = pytest.importorskip("skimage.metrics").structural_similarity

pytestmark = pytest.mark.parity

_COSINE_GATE = 0.99  # threshold=0 re-walk parity (calibration self-check measured >= 0.999)
# Default-threshold quality floor. Under the v0.10.0 gate this fixture's own saved
# stock-q4 pair measures SSIM 0.9672 at the shipped thresh 0.30 (0.9781 under 0.9.x, whose
# 768x768 sweep on the mixed-precision build read 0.9873). 0.95 sits 0.017 below the
# current number — real margin, but not the ~0.04 it once was; if a re-sweep moves the
# default, re-derive this from the new pair rather than inheriting it.
_SSIM_GATE = 0.95
# Expected skip band at thresh 0.30 under consecutive-delta anchoring: the v0.10.0 bench
# at this exact recipe (768x768, 50 steps, g=4.0, q4, seed 42, red apple) skipped 33 of 48
# active steps in every rep. 0.9.x skipped 24 here; the old 15-35 band let that shift pass
# unnoticed, so this one is centred on the measured value and would go red on either the
# 0.9.x count or a dormant cache.
_MIN_SKIP, _MAX_SKIP = 28, 38
# Expected skip band for 2512 at its DEFAULT_THRESH 0.25: the threshold sweep on this checkpoint
# (scripts/sweep_threshold_qwen.py --model Qwen/Qwen-Image-2512 --plain-q4, 2026-10-20, same recipe) skipped
# 29 of 48 active steps. The band is 29 +/- 5, the same convention as the original's 28-38 around its
# measured 33: it goes red on a dormant cache or on a threshold moved far enough to change the count by more
# than five steps.
_MIN_SKIP_2512, _MAX_SKIP_2512 = 24, 34
PROMPT = "a red apple on a wooden table"
SEED = 42
HEIGHT = WIDTH = 768
GUIDANCE = 4.0
_ARTIFACTS = Path(__file__).parent / "_artifacts" / "parity_qwen"


class _LatentCapture:
    """Captures the pre-VAE latent via mflux's after_loop callback."""

    def __init__(self) -> None:
        self.latent: mx.array | None = None

    def call_after_loop(self, seed, prompt, latents, config, **_):  # noqa: ANN001, ARG002
        self.latent = latents


def _unregister(flux: Any, cap: _LatentCapture) -> None:
    for attr in ("after_loop", "in_loop", "before_loop", "interrupt"):
        lst = getattr(flux.callbacks, attr, None)
        if isinstance(lst, list):
            for i in range(len(lst) - 1, -1, -1):
                if lst[i] is cap:
                    del lst[i]


def _capture_latent(flux: Any, **gen_kwargs: Any) -> mx.array:
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
    af, bf = a.astype(mx.float32), b.astype(mx.float32)
    return float(mx.sum(af * bf) / (mx.linalg.norm(af) * mx.linalg.norm(bf)))


def _gen_image_array(flux: Any, *, save_path: Path, steps: int) -> np.ndarray:
    image = flux.generate_image(
        prompt=PROMPT, seed=SEED, num_inference_steps=steps, height=HEIGHT, width=WIDTH, guidance=GUIDANCE
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path=str(save_path), export_json_metadata=False)
    return np.array(Image.open(save_path).convert("RGB"), dtype=np.uint8)


@dataclass(frozen=True)
class _Checkpoint:
    """One parametrization of the file: the model plus what its gates expect."""

    flux: Any
    variant_id: str
    default_thresh: float
    min_skip: int
    max_skip: int


# id -> (HF model, variant id, DEFAULT_THRESH read from that variant's own config, skip band)
_CHECKPOINTS = {
    "original": (
        "Qwen/Qwen-Image",
        "qwen-image",
        _original_config.DEFAULT_THRESH,
        (_MIN_SKIP, _MAX_SKIP),
    ),
    "2512": (
        "Qwen/Qwen-Image-2512",
        "qwen-image-2512",
        _config_2512.DEFAULT_THRESH,
        (_MIN_SKIP_2512, _MAX_SKIP_2512),
    ),
}


@pytest.fixture(scope="module", params=list(_CHECKPOINTS), ids=list(_CHECKPOINTS))
def qwen_case(request: pytest.FixtureRequest) -> Iterator[_Checkpoint]:
    """Load one checkpoint (module-scoped per param: pytest tears the first down before the
    second is built) with the text encoders freed after prompt encoding."""
    from _memory_saver import make_memory_saver
    from _qwen_config import qwen_config
    from mflux.callbacks.instances.memory_saver import MemorySaver
    from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

    model, variant_id, default_thresh, (min_skip, max_skip) = _CHECKPOINTS[request.param]
    _max_set = mx.device_info()["max_recommended_working_set_size"]
    mx.set_wired_limit(int(_max_set * 0.85))  # device-derived; 20B near the 32 GB edge
    flux = QwenImage(quantize=4, model_config=qwen_config(model))
    flux.freeze()
    # Same registration as scripts/sweep_threshold_qwen.py: keep_transformer=True frees only
    # the text encoders, cache_limit_bytes=None leaves our caps and the untiled VAE decode alone.
    flux.callbacks.register(make_memory_saver(flux, MemorySaver))
    case = _Checkpoint(
        flux=flux,
        variant_id=variant_id,
        default_thresh=default_thresh,
        min_skip=min_skip,
        max_skip=max_skip,
    )
    yield case
    # pytest still holds `case` as this fixture's cached result while the teardown runs, so deleting names here would
    # free nothing: release the weights through the model object itself.
    release_model(flux)


@pytest.fixture(autouse=True)
def _clear_cache_between_tests():  # noqa: ANN202
    yield
    mx.clear_cache()  # keep the peak ~one generation across the module (memory edge)


def test_cfg_parity_at_threshold_zero(qwen_case: _Checkpoint) -> None:
    """At rel_l1_thresh=0 the gated CFG path (re-walk of QwenTransformer.__call__
    per branch + mflux's external combine) must match real generation within Metal
    noise, and never skip. Short 8-step schedule = compute-path correctness."""
    flux = qwen_case.flux
    kw = dict(prompt=PROMPT, seed=SEED, num_inference_steps=8, height=HEIGHT, width=WIDTH, guidance=GUIDANCE)

    vanilla_before = _capture_latent(flux, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TeaCacheDisabledWarning)
        ctx = apply_teacache(flux, rel_l1_thresh=0.0)
    with ctx as h:
        wrapper = _capture_latent(flux, **kw)
        skipped = h.stats.skipped_count
    vanilla_after = _capture_latent(flux, **kw)

    cos = _cosine(vanilla_before, wrapper)
    assert cos >= _COSINE_GATE, (
        f"thresh=0 cosine vs same-process vanilla = {cos:.6f} < {_COSINE_GATE}; "
        f"max_abs_diff={float(mx.max(mx.abs(vanilla_before - wrapper))):.4e}"
    )
    assert skipped == 0, f"threshold=0 must never skip; got skipped={skipped}"
    assert mx.array_equal(vanilla_before, vanilla_after), (
        "restore() left a trace; vanilla_after differs from vanilla_before"
    )


def test_image_quality_ssim_at_default_threshold(qwen_case: _Checkpoint) -> None:
    """User-facing quality gate at the calibrated 50-step recipe + the shipped
    DEFAULT_THRESH: caching MUST engage (skipped > 0 — the dormant-cache
    guard) and SSIM vs vanilla MUST hold the PR-gate floor. Images saved for
    manual inspection. Each checkpoint is gated at its own variant's DEFAULT_THRESH."""
    flux = qwen_case.flux
    out_dir = _ARTIFACTS / qwen_case.variant_id
    van = _gen_image_array(flux, save_path=out_dir / "vanilla.png", steps=50)
    ctx = apply_teacache(flux)  # builtin DEFAULT_THRESH of the matched variant
    with ctx as h:
        assert h.variant_id == qwen_case.variant_id, (
            f"matched variant {h.variant_id!r}, expected {qwen_case.variant_id!r}"
        )
        assert h.rel_l1_thresh == qwen_case.default_thresh, (
            f"applied thresh {h.rel_l1_thresh} != {qwen_case.variant_id} DEFAULT_THRESH {qwen_case.default_thresh}"
        )
        wrap = _gen_image_array(flux, save_path=out_dir / "wrapper_default_thresh.png", steps=50)
        skipped, computed = h.stats.skipped_count, h.stats.computed_count
    score = float(ssim(van, wrap, channel_axis=-1, data_range=255))
    assert qwen_case.min_skip <= skipped <= qwen_case.max_skip, (
        f"skip count {skipped} outside the expected band [{qwen_case.min_skip}, {qwen_case.max_skip}] at "
        f"DEFAULT_THRESH {qwen_case.default_thresh} (computed={computed}) — a dormant or runaway cache"
    )
    assert score >= _SSIM_GATE, (
        f"SSIM {score:.4f} < {_SSIM_GATE} at DEFAULT_THRESH {qwen_case.default_thresh} (skipped={skipped})"
    )


def test_skip_path_engages_and_produces_finite_latent(qwen_case: _Checkpoint) -> None:
    """Mechanical skip-path correctness: a high threshold forces skips and the
    reconstructed latent (img_in + cached_residual per CFG branch) stays finite —
    no NaN/inf from a shape/broadcast bug. No quality bound on the short schedule
    (8-step over-skips the 50-step-calibrated coefficients); skip QUALITY is gated
    by the SSIM test above at the pinned recipe."""
    flux = qwen_case.flux
    kw = dict(prompt=PROMPT, seed=SEED, num_inference_steps=8, height=HEIGHT, width=WIDTH, guidance=GUIDANCE)
    ctx = apply_teacache(flux, rel_l1_thresh=0.5)
    with ctx as h:
        latent = _capture_latent(flux, **kw)
        skipped = h.stats.skipped_count
    assert skipped >= 1, f"rel_l1_thresh=0.5 on 8 steps should skip >=1 step; got {skipped}"
    assert bool(mx.all(mx.isfinite(latent))), "skipped-step reconstruction produced non-finite latent"
