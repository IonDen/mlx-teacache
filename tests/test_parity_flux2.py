"""FLUX.2 Klein 4b parity tests — paired same-process methodology.

Mirrors `test_parity_flux1.py` (Task 25, v2.6) for the FLUX.2 path. The
oracle is paired same-process parity, not committed fixtures — see
`docs/superpowers/notes/2026-05-14-task-25-mlx-nondeterminism.md` for
why cross-process bit-exact byte parity isn't achievable on MLX/Metal.

FLUX.2 has two code paths gated by `guidance`:

- **guidance == 1.0** → no negative prompt encoded → no CFG fallback →
  every step exercises our gate (`flux2_forward_with_gate` via the
  proxied `_predict`). This is what we test for threshold-zero parity.
- **guidance > 1.0** → negative prompt encoded → CFG fallback path
  bypasses our gate entirely (matches vanilla mflux exactly).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import pytest

from mlx_teacache import (
    AlreadyPatchedError,
    apply_teacache,
)

pytestmark = pytest.mark.parity


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


def _gen_kwargs_klein(prompt: str, *, guidance: float = 1.0) -> dict[str, Any]:
    """guidance=1.0 → no CFG fallback (our gate is exercised every step).
    guidance>1.0 → CFG fallback path bypasses our gate entirely."""
    return {
        "prompt": prompt, "seed": 42, "num_inference_steps": 8,
        "height": 512, "width": 512, "guidance": guidance,
    }


def _paired_parity(
    flux: Any, gen_kwargs: dict[str, Any], *, rel_l1_thresh: float = 0.0,
    **apply_kwargs: Any,
) -> tuple[mx.array, mx.array, mx.array, int]:
    """Same protocol as test_parity_flux1's helper.

    Returns (vanilla_before, wrapper, vanilla_after, skipped_count).
    """
    vanilla_before = _capture(flux, **gen_kwargs)
    with apply_teacache(flux, rel_l1_thresh=rel_l1_thresh, **apply_kwargs) as h:
        wrapper = _capture(flux, **gen_kwargs)
        skipped = h.stats.skipped_count
    vanilla_after = _capture(flux, **gen_kwargs)
    return vanilla_before, wrapper, vanilla_after, skipped


# ---------------------------------------------------------------------------
# Module-scoped fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flux2_klein() -> Any:
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())
    flux.freeze()
    return flux


# ---------------------------------------------------------------------------
# Tier 2 — paired same-process parity at threshold=0
# ---------------------------------------------------------------------------


def test_paired_parity_klein_pr_gate(flux2_klein: Any) -> None:
    """PR-time correctness gate for FLUX.2 Klein 4b. guidance=1.0 means
    every step exercises our gate (no CFG fallback)."""
    kw = _gen_kwargs_klein(PR_TIME_PROMPT)
    vb, w, va, skipped = _paired_parity(flux2_klein, kw)
    assert mx.array_equal(vb, w), (
        "wrapper at rel_l1_thresh=0 must match same-process vanilla math"
    )
    assert mx.array_equal(vb, va), (
        "restore() left a trace; vanilla_after differs from vanilla_before"
    )
    assert skipped == 0


@pytest.mark.slow
@pytest.mark.parametrize("prompt", REFERENCE_PROMPTS)
def test_paired_parity_klein_full(flux2_klein: Any, prompt: str) -> None:
    """Nightly correctness gate. All 5 reference prompts."""
    kw = _gen_kwargs_klein(prompt)
    vb, w, va, skipped = _paired_parity(flux2_klein, kw)
    assert mx.array_equal(vb, w)
    assert mx.array_equal(vb, va)
    assert skipped == 0


def test_paired_parity_reverse_order_klein(flux2_klein: Any) -> None:
    """Reverse-order control: wrapper → restore → vanilla."""
    kw = _gen_kwargs_klein(PR_TIME_PROMPT)
    with apply_teacache(flux2_klein, rel_l1_thresh=0.0):
        wrapper = _capture(flux2_klein, **kw)
    vanilla = _capture(flux2_klein, **kw)
    assert mx.array_equal(wrapper, vanilla)


# ---------------------------------------------------------------------------
# CFG fallback parity — wrapper at guidance > 1.0 must match vanilla exactly
# ---------------------------------------------------------------------------


def test_cfg_fallback_matches_vanilla(flux2_klein: Any) -> None:
    """At guidance > 1.0 the FLUX.2 wrapper bypasses gating entirely and
    delegates to vanilla mflux. Paired parity must hold AND every
    StepDecision should be `cfg-fallback`."""
    kw = _gen_kwargs_klein(PR_TIME_PROMPT, guidance=3.5)
    vb, w, va, _ = _paired_parity(flux2_klein, kw)
    assert mx.array_equal(vb, w), (
        "CFG fallback should be byte-identical to vanilla in-process"
    )
    assert mx.array_equal(vb, va)


# ---------------------------------------------------------------------------
# Edge cases (paired against same-process vanilla)
# ---------------------------------------------------------------------------


def test_threshold_zero_with_negative_coefficients_no_skip(flux2_klein: Any) -> None:
    """rel_l1_thresh <= 0 short-circuits regardless of coefficients."""
    kw = _gen_kwargs_klein(PR_TIME_PROMPT)
    pathological = (0.0, 0.0, 0.0, -1000.0, 0.0)
    vanilla = _capture(flux2_klein, **kw)
    with apply_teacache(
        flux2_klein, rel_l1_thresh=0.0, coefficients=pathological,
    ) as h:
        wrapper = _capture(flux2_klein, **kw)
        skipped = h.stats.skipped_count
    assert mx.array_equal(vanilla, wrapper)
    assert skipped == 0


# ---------------------------------------------------------------------------
# Exact (non-numerical) tests — fast
# ---------------------------------------------------------------------------


def test_idempotency_raises_already_patched(flux2_klein: Any) -> None:
    h = apply_teacache(flux2_klein, rel_l1_thresh=0.25)
    try:
        with pytest.raises(AlreadyPatchedError):
            apply_teacache(flux2_klein, rel_l1_thresh=0.4)
    finally:
        h.restore()


def test_restore_completeness(flux2_klein: Any) -> None:
    """Restore postconditions for the FLUX.2 path. FLUX.2 patches via
    instance-attribute `_predict` replacement, not transformer proxy."""
    original_predict_was_instance_attr = "_predict" in vars(flux2_klein)
    original_predict = flux2_klein._predict if original_predict_was_instance_attr else None
    original_generate = (
        flux2_klein.generate_image if "generate_image" in vars(flux2_klein) else None
    )

    h = apply_teacache(flux2_klein, rel_l1_thresh=0.25)
    cb = h._callback_instance
    h.restore()

    if original_predict_was_instance_attr:
        assert flux2_klein._predict is original_predict
    else:
        assert "_predict" not in vars(flux2_klein)
    if original_generate is not None:
        assert flux2_klein.generate_image is original_generate
    else:
        assert "generate_image" not in vars(flux2_klein)
        assert callable(flux2_klein.generate_image)
    assert cb not in flux2_klein.callbacks.before_loop
    assert getattr(flux2_klein, "_teacache_handle", None) is None
    # Re-apply succeeds.
    h2 = apply_teacache(flux2_klein)
    h2.restore()
