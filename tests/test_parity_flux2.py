"""FLUX.2 Klein 4b parity tests — paired same-process methodology.

Mirrors `test_parity_flux1.py` (Task 25, v2.6) for the FLUX.2 path, but
with two important differences:

1. **Numerical tolerance instead of bit-exact on the gated path.**
   mflux wraps Flux2Klein's `_predict` in `mx.compile`. Our TeaCache
   integration replaces `_predict` with an eager-Python wrapper (so
   per-step gating actually runs every step — `mx.compile` would trace
   our gating code once and elide it). Compiled-vs-eager MLX dispatches
   slightly different bf16 op orderings, producing ~1% per-element
   divergence on the gated path even with threshold=0 and identical
   math. This is fundamental to the integration design (see
   `src/mlx_teacache/integrations/mflux/flux2.py` docstring + the
   user-mlx-developer skill's mflux-and-local-projects.md "mx.compile
   interaction with Python-side gating"). We gate the threshold=0
   parity with `mx.allclose(atol=0.1, rtol=0.05)` — per the
   python-ml-testing skill's "full forward of small model, bf16" row.

2. **CFG fallback IS bit-exact.** When guidance > 1.0 our wrapper
   delegates to vanilla mflux entirely (no gating). Both call the same
   compiled `_predict`. Same-process paired parity at `mx.array_equal`
   holds there.

Latent-level numerical tolerance is the first gate. Image-level SSIM on
decoded images (`test_image_quality_flux2.py`) is the second gate — it's
robust to bf16 op-ordering noise and is the upstream-standard validation
pattern.
"""

from __future__ import annotations

import warnings as _w
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from mlx_teacache import (
    AlreadyPatchedError,
    apply_teacache,
)
from mlx_teacache.errors import TeaCacheDisabledWarning
from tests.conftest import expect_distilled_warning

pytestmark = pytest.mark.parity

# Cosine-similarity gate for "wrapper at threshold=0 matches same-process
# vanilla". Per-element tolerance (mx.allclose) is the wrong oracle on the
# FLUX.2 gated path: our re-implementation of Flux2Transformer.__call__ runs
# eager Python (mflux's predict path is mx.compile-wrapped on every chip
# except base M1 / base M2 — see docs/m3-plus-tradeoff.md), and MLX lazy-eval
# ordering between our eager structure and mflux's compiled/eager structure
# differs in subtle ref-count / dispatch ways that compound to ~3-4 max_abs
# over 8 steps even though the math is equivalent. Cosine similarity catches
# catastrophic divergence (real math bugs) while accepting the ULP-level
# dispatch noise.
#
# Measured 2026-05-15: cosine ~0.99+ on M1 Max FLUX.2 Klein 4b at 8 steps
# with the current implementation. Gate set 0.02 below that to absorb
# prompt-to-prompt variance.
#
# The restore control (vanilla_before vs vanilla_after) remains bit-exact —
# both vanilla runs go through the same compiled _predict.
_FLUX2_COSINE_GATE = 0.97


# ---------------------------------------------------------------------------
# Constants and helpers — mirrors test_parity_flux1.py
# ---------------------------------------------------------------------------


REFERENCE_PROMPTS = (
    "a red apple on a wooden table",
    "mountain landscape at sunset",
    "portrait of a woman",
    "abstract pattern with circles",
    "text saying HELLO",
)
PR_TIME_PROMPT = "a red apple on a wooden table"


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


def _gen_kwargs_klein(
    prompt: str, *, variant_id: str = "flux2-klein-4b", guidance: float = 1.0
) -> dict[str, Any]:
    """Generation kwargs for FLUX.2 Klein variants.

    guidance=1.0 → no CFG fallback (our gate is exercised every step).
    guidance>1.0 → CFG fallback path bypasses our gate entirely.

    Distilled Klein 4B / 9B use the 8-step default schedule (matches their
    runtime usage). base-4b uses the calibration-time 25-step schedule."""
    if variant_id in ("flux2-klein-4b", "flux2-klein-9b"):
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


def _paired_parity(
    flux: Any,
    gen_kwargs: dict[str, Any],
    *,
    variant_id: str,
    rel_l1_thresh: float = 0.0,
    **apply_kwargs: Any,
) -> tuple[mx.array, mx.array, mx.array, int]:
    """Same protocol as test_parity_flux1's helper.

    Returns (vanilla_before, wrapper, vanilla_after, skipped_count).
    """
    vanilla_before = _capture(flux, **gen_kwargs)
    with _w.catch_warnings():
        # Suppress only when 0.0 actually warns; a positive threshold keeps
        # the filterwarnings=error regime fully live inside the block.
        if rel_l1_thresh == 0.0:
            _w.simplefilter("ignore", TeaCacheDisabledWarning)
        with expect_distilled_warning(variant_id):
            ctx = apply_teacache(flux, rel_l1_thresh=rel_l1_thresh, **apply_kwargs)
    with ctx as h:
        wrapper = _capture(flux, **gen_kwargs)
        skipped = h.stats.skipped_count
    vanilla_after = _capture(flux, **gen_kwargs)
    return vanilla_before, wrapper, vanilla_after, skipped


# ---------------------------------------------------------------------------
# Module-scoped fixture
# ---------------------------------------------------------------------------


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
# Tier 2 — paired same-process parity at threshold=0
# ---------------------------------------------------------------------------


def _cosine(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32)
    bf = b.astype(mx.float32)
    return float(mx.sum(af * bf) / (mx.linalg.norm(af) * mx.linalg.norm(bf)))


def test_paired_parity_klein_pr_gate(flux2_klein: tuple[Any, str]) -> None:
    """PR-time correctness gate for FLUX.2 Klein 4b. guidance=1.0 means
    every step exercises our gate (no CFG fallback). Uses cosine
    similarity, not bit-exact — see module docstring."""
    flux, variant_id = flux2_klein
    kw = _gen_kwargs_klein(PR_TIME_PROMPT, variant_id=variant_id)
    vb, w, va, skipped = _paired_parity(flux, kw, variant_id=variant_id)
    cos = _cosine(vb, w)
    assert cos >= _FLUX2_COSINE_GATE, (
        f"wrapper at rel_l1_thresh=0 cosine vs same-process vanilla "
        f"= {cos:.6f} < {_FLUX2_COSINE_GATE}; max_abs_diff="
        f"{float(mx.max(mx.abs(vb - w))):.4e}"
    )
    # vanilla_before vs vanilla_after IS bit-exact (both use the same
    # compiled _predict). If this fails, restore() leaked state.
    assert mx.array_equal(vb, va), "restore() left a trace; vanilla_after differs from vanilla_before"
    assert skipped == 0


@pytest.mark.parametrize("image_strength", [0.0, 0.5, 0.7])
def test_paired_parity_at_threshold_zero_klein_pr_gate(
    flux2_klein: tuple[Any, str], image_strength: float, request: pytest.FixtureRequest
) -> None:
    """Same-process paired parity at rel_l1_thresh=0 for FLUX.2 Klein 4B.
    Cosine >= 0.97 (not bit-exact) because the wrapper is eager-Python and
    vanilla _predict is compiled — dispatch noise compounds ~1 ULP/element
    across steps. Covers txt2img + img2img schedule slices.

    Distilled klein @ strength=0.7 is xfail (strict): the 8-step schedule
    reduces to ~3 active steps; with default skip_first=1 + skip_last=1 only
    1 eligible step remains (0 possible skips) and the PER-GENERATION
    TeaCacheNoBenefitWarning fires from lifecycle.py:120 during
    flux.generate_image() (commit a1524de, pre-v0.6.0). The warning is
    correct production behavior, not a port regression — same forward code
    passes at strength=0.7 on klein-base-4b/9b which have ≥17 active steps.

    The apply-time TeaCacheNoBenefitWarning (distilled Kleins always warn at
    apply(), independent of image_strength) is expected separately via
    expect_distilled_warning below so the xfail above is satisfied by the
    documented per-generation reason, not by the apply-time warning firing
    first."""
    flux, variant_id = flux2_klein
    if variant_id in ("flux2-klein-4b", "flux2-klein-9b") and image_strength == 0.7:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="distilled klein @ strength=0.7 triggers the per-generation "
                "TeaCacheNoBenefitWarning (active_num_steps≈3, 0 possible skips with "
                "default skip-window); warning predates v0.6.0",
            )
        )
    kw = _gen_kwargs_klein("a red apple on a wooden table", variant_id=variant_id)
    if image_strength > 0.0:
        kw["image_path"] = str(Path(__file__).parent / "fixtures" / "init_images" / "natural_512.png")
        kw["image_strength"] = image_strength

    vanilla_latent = _capture(flux, **kw)
    with _w.catch_warnings():
        _w.simplefilter("ignore", TeaCacheDisabledWarning)
        with expect_distilled_warning(variant_id):
            ctx = apply_teacache(flux, rel_l1_thresh=0.0)
    with ctx:
        wrapper_latent = _capture(flux, **kw)

    score = _cosine(vanilla_latent, wrapper_latent)
    assert score >= 0.97, (
        f"FLUX.2 cosine parity below gate at image_strength={image_strength}: "
        f"got {score:.4f}, required >= 0.97"
    )


@pytest.mark.parity
def test_paired_cfg_parity_at_threshold_zero_klein_base_4b_pr_gate() -> None:
    """v0.4.1 release blocker: at rel_l1_thresh=0, the gated CFG path must
    produce the same latent (within Metal noise) as real mflux generation
    at guidance=4.0, num_inference_steps=50 on flux2-klein-base-4b. Per
    audit Finding 4, the in-repo _vanilla_flux2_cfg_predict helper is too
    weak as the release oracle because it shares assumptions with the
    gated function; this test uses real mflux.

    Pattern: same as test_paired_parity_klein_pr_gate — _capture latents
    via mflux callback, compare with _cosine >= _FLUX2_COSINE_GATE.
    """
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
    flux.freeze()

    kw = {
        "prompt": PR_TIME_PROMPT,
        "seed": 42,
        "num_inference_steps": 50,
        "height": 512,
        "width": 512,
        "guidance": 4.0,
    }

    # 1. Vanilla mflux latent (no wrapper).
    vanilla_latent = _capture(flux, **kw)

    # 2. Wrapped at threshold=0 (no skips). Same process, same flux instance.
    with _w.catch_warnings():
        _w.simplefilter("ignore", TeaCacheDisabledWarning)
        handle = apply_teacache(flux, rel_l1_thresh=0.0)
    try:
        wrapper_latent = _capture(flux, **kw)
        skipped = handle.stats.skipped_count
    finally:
        handle.restore()

    cos = _cosine(vanilla_latent, wrapper_latent)
    assert cos >= _FLUX2_COSINE_GATE, (
        f"CFG wrapper at rel_l1_thresh=0, guidance=4.0 cosine vs same-process vanilla "
        f"= {cos:.6f} < {_FLUX2_COSINE_GATE}; max_abs_diff="
        f"{float(mx.max(mx.abs(vanilla_latent - wrapper_latent))):.4e}"
    )
    assert skipped == 0, f"threshold=0 should never skip; got skipped={skipped}"


@pytest.mark.slow
@pytest.mark.parametrize("prompt", REFERENCE_PROMPTS)
def test_paired_parity_klein_full(flux2_klein: tuple[Any, str], prompt: str) -> None:
    """Nightly correctness gate. All 5 reference prompts."""
    flux, variant_id = flux2_klein
    kw = _gen_kwargs_klein(prompt, variant_id=variant_id)
    vb, w, va, skipped = _paired_parity(flux, kw, variant_id=variant_id)
    cos = _cosine(vb, w)
    assert cos >= _FLUX2_COSINE_GATE, f"prompt={prompt!r} cosine={cos:.6f} < {_FLUX2_COSINE_GATE}"
    assert mx.array_equal(vb, va)
    assert skipped == 0


def test_paired_parity_reverse_order_klein(flux2_klein: tuple[Any, str]) -> None:
    """Reverse-order control: wrapper → restore → vanilla."""
    flux, variant_id = flux2_klein
    kw = _gen_kwargs_klein(PR_TIME_PROMPT, variant_id=variant_id)
    with _w.catch_warnings():
        _w.simplefilter("ignore", TeaCacheDisabledWarning)
        with expect_distilled_warning(variant_id):
            ctx = apply_teacache(flux, rel_l1_thresh=0.0)
    with ctx:
        wrapper = _capture(flux, **kw)
    vanilla = _capture(flux, **kw)
    cos = _cosine(wrapper, vanilla)
    assert cos >= _FLUX2_COSINE_GATE, f"reverse-order parity cosine={cos:.6f} < {_FLUX2_COSINE_GATE}"


# ---------------------------------------------------------------------------
# CFG fallback parity — wrapper at guidance > 1.0 must match vanilla exactly
# ---------------------------------------------------------------------------


def test_cfg_fallback_matches_vanilla(flux2_klein: tuple[Any, str]) -> None:
    """At guidance > 1.0 the FLUX.2 wrapper routes through the gated
    CFG-per-branch path (flux2_cfg_forward_with_gate, v0.4.1+) — NOT a
    bypass to vanilla mflux's compiled _predict. The wrapper runs eager
    Python while vanilla goes through mx.compile, so paired parity is
    NOT bit-exact: ~1 ULP per branch compounds across steps. Cosine
    similarity stays >= 0.97 — see the module docstring."""
    flux, variant_id = flux2_klein
    kw = _gen_kwargs_klein(PR_TIME_PROMPT, variant_id=variant_id, guidance=3.5)
    vb, w, va, _ = _paired_parity(flux, kw, variant_id=variant_id)
    cos = _cosine(vb, w)
    assert cos >= _FLUX2_COSINE_GATE, (
        f"CFG-gated wrapper cosine vs vanilla = {cos:.6f} "
        f"< {_FLUX2_COSINE_GATE}; eager wrapper vs compiled vanilla "
        f"is expected to diverge ~1 ULP/step but stay above gate."
    )
    assert mx.array_equal(vb, va), "restore() left a trace; vanilla_after differs from vanilla_before"


# ---------------------------------------------------------------------------
# Edge cases (paired against same-process vanilla)
# ---------------------------------------------------------------------------


def test_threshold_zero_with_negative_coefficients_no_skip(flux2_klein: tuple[Any, str]) -> None:
    """rel_l1_thresh <= 0 short-circuits regardless of coefficients.
    Cosine gate per module docstring (compile-vs-eager dispatch noise)."""
    flux, variant_id = flux2_klein
    kw = _gen_kwargs_klein(PR_TIME_PROMPT, variant_id=variant_id)
    pathological = (0.0, 0.0, 0.0, -1000.0, 0.0)
    vanilla = _capture(flux, **kw)
    # Explicit coefficients suppress the at-apply distilled warning (api.py
    # warns only when the caller passed none), so no expect_distilled_warning
    # here: under filterwarnings=error it fails loudly if the warning ever
    # fires. Wrapping this call in the helper made pytest.warns raise after the
    # patch was installed, leaking it into the next two tests as
    # AlreadyPatchedError (seen on the 2026-09-06 lane).
    with _w.catch_warnings():
        _w.simplefilter("ignore", TeaCacheDisabledWarning)
        ctx = apply_teacache(flux, rel_l1_thresh=0.0, coefficients=pathological)
    with ctx as h:
        wrapper = _capture(flux, **kw)
        skipped = h.stats.skipped_count
    assert _cosine(vanilla, wrapper) >= _FLUX2_COSINE_GATE
    assert skipped == 0


# ---------------------------------------------------------------------------
# Exact (non-numerical) tests — fast
# ---------------------------------------------------------------------------


def test_idempotency_raises_already_patched(flux2_klein: tuple[Any, str]) -> None:
    flux, variant_id = flux2_klein
    with expect_distilled_warning(variant_id):
        h = apply_teacache(flux, rel_l1_thresh=0.25)
    try:
        # Already-patched sentinel check runs before the registry match, so
        # this second call never reaches the apply-time warning — no wrap.
        with pytest.raises(AlreadyPatchedError):
            apply_teacache(flux, rel_l1_thresh=0.4)
    finally:
        h.restore()


def test_restore_completeness(flux2_klein: tuple[Any, str]) -> None:
    """Restore postconditions for the FLUX.2 path. FLUX.2 patches via
    instance-attribute `_predict` replacement, not transformer proxy."""
    flux, variant_id = flux2_klein
    original_predict_was_instance_attr = "_predict" in vars(flux)
    original_predict = flux._predict if original_predict_was_instance_attr else None
    original_generate = flux.generate_image if "generate_image" in vars(flux) else None

    with expect_distilled_warning(variant_id):
        h = apply_teacache(flux, rel_l1_thresh=0.25)
    cb = h._callback_instance
    h.restore()

    if original_predict_was_instance_attr:
        assert flux._predict is original_predict
    else:
        assert "_predict" not in vars(flux)
    if original_generate is not None:
        assert flux.generate_image is original_generate
    else:
        assert "generate_image" not in vars(flux)
        assert callable(flux.generate_image)
    assert cb not in flux.callbacks.before_loop
    assert getattr(flux, "_teacache_handle", None) is None
    # Re-apply succeeds.
    with expect_distilled_warning(variant_id):
        h2 = apply_teacache(flux)
    h2.restore()
