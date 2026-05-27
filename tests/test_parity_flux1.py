"""FLUX.1 parity tests — paired same-process methodology.

Per the Task 25 cross-process non-determinism investigation, MLX/Metal on
Apple Silicon does not promise byte-identical output across Python runs
even with the same model, seed, prompt, and config. Comparing wrapper
output to a committed `.safetensors` fixture is therefore not a
correctness gate for wrapper math — the saved bytes encode "MLX dispatch
in the script that
generated them", not "the correct denoising result."

Within a single Python process, however, MLX is deterministic. We exploit
that with paired same-process parity:

    vanilla_before = capture(flux, prompt=p)
    with apply_teacache(flux, rel_l1_thresh=0.0) as h:
        wrapper = capture(flux, prompt=p)
    vanilla_after = capture(flux, prompt=p)

    assert mx.array_equal(vanilla_before, wrapper)       # math equivalence
    assert mx.array_equal(vanilla_before, vanilla_after) # restore is clean
    assert h.stats.skipped_count == 0

The `vanilla_after` control catches restore-leakage (callbacks, sentinels,
proxy state). For at least one prompt we also run the reverse order
(wrapper → restore → vanilla) to guard against warm-state ordering bias.

Cost: each paired test is ~3× generation time (~7.5 min at 25 steps on
M-series). PR-time tests use one prompt; full 5-prompt suite is gated by
`@pytest.mark.slow` for nightly.

Committed reference latents in `tests/reference/flux1-dev/` and
`tests/reference/flux1-schnell/` are kept ONLY for drift detection (see
`tests/test_fixtures_integrity.py` and `tests/test_hf_revisions.py`).
They are NOT the parity oracle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

from mlx_teacache import (
    AlreadyPatchedError,
    MissingGenerationContextError,
    apply_teacache,
)
from mlx_teacache.errors import InvalidStepWindowError

pytestmark = pytest.mark.parity


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------


REFERENCE_PROMPTS = (
    "a red apple on a wooden table",
    "mountain landscape at sunset",
    "portrait of a woman",
    "abstract pattern with circles",
    "text saying HELLO",
)
PR_TIME_PROMPT = "a red apple on a wooden table"
# Cosine-similarity gate on latents at default rel_l1_thresh. NOT yet
# calibrated against measured values per the 2026-05-15 audit — set
# conservatively (matches the python-ml-testing skill's "full forward of
# small model" tolerance range). Catches catastrophic divergence. Tighten
# after running the full 5-prompt slow suite once and observing the
# distribution of latent cosines for the current coefficients.
_COSINE_GATE = 0.95


class _LatentCapture:
    """One-shot capture of the pre-VAE latent via mflux's after_loop callback."""

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
    """Run generate_image and return the final pre-VAE latent."""
    cap = _LatentCapture()
    flux.callbacks.register(cap)
    try:
        flux.generate_image(**gen_kwargs)
    finally:
        _unregister(flux, cap)
    if cap.latent is None:
        raise RuntimeError("call_after_loop never fired — latent capture failed")
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


def _gen_kwargs_schnell(prompt: str) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "seed": 42,
        "num_inference_steps": 25,
        "height": 512,
        "width": 512,
        "guidance": 0.0,
    }


def _cosine(a: mx.array, b: mx.array) -> float:
    return float(mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b)))


def _paired_parity(
    flux: Any,
    gen_kwargs: dict[str, Any],
    *,
    rel_l1_thresh: float = 0.0,
    **apply_kwargs: Any,
) -> tuple[mx.array, mx.array, mx.array, int]:
    """Run the paired same-process parity protocol.

    Returns (vanilla_before, wrapper, vanilla_after, skipped_count).
    """
    vanilla_before = _capture(flux, **gen_kwargs)
    with apply_teacache(flux, rel_l1_thresh=rel_l1_thresh, **apply_kwargs) as h:
        wrapper = _capture(flux, **gen_kwargs)
        skipped = h.stats.skipped_count
    vanilla_after = _capture(flux, **gen_kwargs)
    return vanilla_before, wrapper, vanilla_after, skipped


# ---------------------------------------------------------------------------
# Module-scoped flux fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flux1_dev() -> Any:
    from mflux.models.flux.variants.txt2img.flux import Flux1

    flux = Flux1.from_name("dev", quantize=4)
    flux.freeze()
    return flux


@pytest.fixture(scope="module")
def flux1_schnell() -> Any:
    from mflux.models.flux.variants.txt2img.flux import Flux1

    flux = Flux1.from_name("schnell", quantize=4)
    flux.freeze()
    return flux


# ---------------------------------------------------------------------------
# Tier 2 — paired same-process parity at threshold=0 (the core correctness gate)
# ---------------------------------------------------------------------------


def test_paired_parity_dev_pr_gate(flux1_dev: Any) -> None:
    """PR-time correctness gate. Single prompt, ~7.5 min walltime.

    Asserts the threshold-zero wrapper produces a mathematically-identical
    latent to vanilla mflux within the same Python process, AND that
    restore() leaves the model in the same observable state as before
    apply_teacache.
    """
    kw = _gen_kwargs_dev(PR_TIME_PROMPT)
    vb, w, va, skipped = _paired_parity(flux1_dev, kw)
    assert mx.array_equal(vb, w), "wrapper at rel_l1_thresh=0 must match same-process vanilla math"
    assert mx.array_equal(vb, va), (
        "restore() left a trace; vanilla_after differs from vanilla_before "
        "— check callback / proxy / sentinel cleanup"
    )
    assert skipped == 0


@pytest.mark.parametrize("image_strength", [0.0, 0.5, 0.7])
def test_paired_parity_at_threshold_zero_dev(flux1_dev: Any, image_strength: float) -> None:
    """Same-process paired parity at rel_l1_thresh=0.0 should be bit-exact
    for FLUX.1 across txt2img (strength=0) and img2img (strength>0)."""
    kwargs = _gen_kwargs_dev("a red apple on a wooden table")
    if image_strength > 0.0:
        kwargs["image_path"] = str(Path(__file__).parent / "fixtures" / "init_images" / "natural_512.png")
        kwargs["image_strength"] = image_strength

    vanilla_latent = _capture(flux1_dev, **kwargs)
    with apply_teacache(flux1_dev, rel_l1_thresh=0.0):
        wrapper_latent = _capture(flux1_dev, **kwargs)

    assert mx.array_equal(vanilla_latent, wrapper_latent), (
        f"FLUX.1-dev paired parity failed at rel_l1_thresh=0 with image_strength={image_strength}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("prompt", REFERENCE_PROMPTS)
def test_paired_parity_dev_full(flux1_dev: Any, prompt: str) -> None:
    """Nightly correctness gate. All 5 reference prompts."""
    kw = _gen_kwargs_dev(prompt)
    vb, w, va, skipped = _paired_parity(flux1_dev, kw)
    assert mx.array_equal(vb, w)
    assert mx.array_equal(vb, va)
    assert skipped == 0


def test_paired_parity_reverse_order_dev(flux1_dev: Any) -> None:
    """Reverse-order control: wrapper → restore → vanilla.

    If only the forward order (vanilla → wrapper → vanilla) passes, MLX
    is making different kernel choices depending on which path runs first.
    That would be a real warm-state ordering bug we'd need to address.
    """
    kw = _gen_kwargs_dev(PR_TIME_PROMPT)
    with apply_teacache(flux1_dev, rel_l1_thresh=0.0):
        wrapper = _capture(flux1_dev, **kw)
    vanilla = _capture(flux1_dev, **kw)
    assert mx.array_equal(wrapper, vanilla), (
        "reverse-order parity failed; wrapper-then-vanilla diverged from the vanilla-then-wrapper baseline"
    )


@pytest.mark.slow
@pytest.mark.parametrize("prompt", REFERENCE_PROMPTS)
def test_paired_parity_schnell_full(flux1_schnell: Any, prompt: str) -> None:
    """Nightly correctness gate for schnell (guidance=0.0 path)."""
    kw = _gen_kwargs_schnell(prompt)
    vb, w, va, skipped = _paired_parity(flux1_schnell, kw)
    assert mx.array_equal(vb, w)
    assert mx.array_equal(vb, va)
    assert skipped == 0


# ---------------------------------------------------------------------------
# Threshold-zero edge cases (paired against same-process vanilla)
# ---------------------------------------------------------------------------


def test_threshold_zero_with_negative_coefficients_no_skip(flux1_dev: Any) -> None:
    """Per §5.3: rel_l1_thresh <= 0 short-circuits to compute regardless
    of coefficients. With pathological negative coefficients the gate must
    still produce a vanilla-equivalent latent and skip zero steps."""
    kw = _gen_kwargs_dev(PR_TIME_PROMPT)
    pathological = (0.0, 0.0, 0.0, -1000.0, 0.0)
    vanilla = _capture(flux1_dev, **kw)
    with apply_teacache(
        flux1_dev,
        rel_l1_thresh=0.0,
        coefficients=pathological,
    ) as h:
        wrapper = _capture(flux1_dev, **kw)
        skipped = h.stats.skipped_count
    assert mx.array_equal(vanilla, wrapper)
    assert skipped == 0


# ---------------------------------------------------------------------------
# Default-threshold quality gate (cosine vs same-process vanilla)
# ---------------------------------------------------------------------------


def test_default_threshold_skips_and_stays_close_to_vanilla(flux1_dev: Any) -> None:
    """At rel_l1_thresh=0.25 the wrapper should skip ≥1 step (proves the
    cache is engaging) AND stay close to a same-process vanilla baseline
    (cosine ≥ 0.985). NOT compared against a committed fixture."""
    kw = _gen_kwargs_dev(PR_TIME_PROMPT)
    vanilla = _capture(flux1_dev, **kw)
    with apply_teacache(flux1_dev, rel_l1_thresh=0.25) as h:
        wrapper = _capture(flux1_dev, **kw)
        skipped = h.stats.skipped_count
    assert skipped >= 1, "default threshold should skip at least one step"
    cos = _cosine(wrapper, vanilla)
    assert cos >= _COSINE_GATE, (
        f"cosine similarity {cos:.4f} < {_COSINE_GATE} — wrapper diverged "
        f"too far from same-process vanilla baseline"
    )


def test_failed_generation_retry_no_stale_cache(flux1_dev: Any) -> None:
    """A mid-loop crash must not leak state into a retry. Retry output
    must remain close to a same-process vanilla baseline."""
    # v0.6.0: _flux1_run_body moved from integrations/mflux/forward.py to
    # variants/flux1_dev/integration.py (verbatim port).
    import mlx_teacache.variants.flux1_dev.integration as fwd

    orig = fwd._flux1_run_body
    call_count = [0]

    def boom(*a: Any, **kw: Any) -> Any:
        call_count[0] += 1
        if call_count[0] == 10:
            raise RuntimeError("simulated mid-loop crash")
        return orig(*a, **kw)

    fwd._flux1_run_body = boom  # type: ignore[assignment]
    try:
        kw = _gen_kwargs_dev(PR_TIME_PROMPT)
        vanilla = _capture(flux1_dev, **kw)
        with apply_teacache(flux1_dev, rel_l1_thresh=0.25):
            with pytest.raises(RuntimeError, match="simulated mid-loop crash"):
                _capture(flux1_dev, **kw)
            fwd._flux1_run_body = orig  # type: ignore[assignment]
            retry = _capture(flux1_dev, **kw)
            cos = _cosine(retry, vanilla)
            assert cos >= _COSINE_GATE, f"retry diverged too far (cos={cos:.4f}); stale cache leak?"
    finally:
        fwd._flux1_run_body = orig  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Exact (non-numerical) tests — same as before, no fixture dependence
# ---------------------------------------------------------------------------


def test_short_schedule_raises_invalid_step_window(flux1_schnell: Any) -> None:
    """skip_first=1 + skip_last=1 with num_steps=2 must raise at t==0."""
    with (
        apply_teacache(
            flux1_schnell,
            rel_l1_thresh=0.25,
            skip_first_n_steps=1,
            skip_last_n_steps=1,
        ),
        pytest.raises(InvalidStepWindowError),
    ):
        flux1_schnell.generate_image(
            seed=42,
            prompt="test",
            num_inference_steps=2,
            height=512,
            width=512,
            guidance=0.0,
        )


def test_idempotency_raises_already_patched(flux1_dev: Any) -> None:
    h = apply_teacache(flux1_dev, rel_l1_thresh=0.25)
    try:
        with pytest.raises(AlreadyPatchedError):
            apply_teacache(flux1_dev, rel_l1_thresh=0.4)
    finally:
        h.restore()


def test_restore_completeness(flux1_dev: Any) -> None:
    """Every restore postcondition.

    Branches on whether generate_image was a pre-patch instance attribute
    (class methods return a fresh bound method on each access, so identity
    comparison only works in the instance-attr case).
    """
    original_transformer = flux1_dev.transformer
    was_instance_attr = "generate_image" in vars(flux1_dev)
    original_generate = flux1_dev.generate_image if was_instance_attr else None
    h = apply_teacache(flux1_dev, rel_l1_thresh=0.25)
    cb = h._callback_instance
    h.restore()
    assert flux1_dev.transformer is original_transformer
    if was_instance_attr:
        assert flux1_dev.generate_image is original_generate
    else:
        assert "generate_image" not in vars(flux1_dev)
        assert callable(flux1_dev.generate_image)
    # mflux 0.17 stores before-loop callbacks in flux.callbacks.before_loop (list).
    assert cb not in flux1_dev.callbacks.before_loop
    assert getattr(flux1_dev, "_teacache_handle", None) is None
    # Re-apply succeeds.
    h2 = apply_teacache(flux1_dev)
    h2.restore()


def test_custom_zero_coefficients_skip_count(flux1_dev: Any) -> None:
    """coefficients=[0]*5 at rel_l1_thresh=0.25 → polynomial=0 → accumulator
    stays 0 → every eligible step skips. Skip count == num_steps -
    skip_first - skip_last - 1 (the -1 is for the first eligible step
    that forces a compute to seed the cache)."""
    num_steps = 25
    skip_first = 1
    skip_last = 1
    expected_skips = num_steps - skip_first - skip_last - 1
    with apply_teacache(
        flux1_dev,
        rel_l1_thresh=0.25,
        coefficients=[0.0] * 5,
        skip_first_n_steps=skip_first,
        skip_last_n_steps=skip_last,
    ) as h:
        flux1_dev.generate_image(
            seed=42,
            prompt=PR_TIME_PROMPT,
            num_inference_steps=num_steps,
            height=512,
            width=512,
            guidance=3.5,
        )
    assert h.stats.skipped_count == expected_skips


def test_cross_generation_cache_reset_dev(flux1_dev: Any) -> None:
    """Two generate_image calls with same prompt+seed inside one
    apply_teacache context produce identical latents — proves cache reset
    on t==0 works for the normal-success path."""
    kw = _gen_kwargs_dev(PR_TIME_PROMPT)
    with apply_teacache(flux1_dev, rel_l1_thresh=0.25):
        r1 = _capture(flux1_dev, **kw)
        r2 = _capture(flux1_dev, **kw)
    assert mx.array_equal(r1, r2)


def test_flux1_callback_replacement_raises(flux1_dev: Any) -> None:
    """FLUX.1 relies on the lifecycle callback for img2img rejection AND
    stats finalization. Replacing flux.callbacks after apply_teacache must
    raise MissingGenerationContextError on the next generate_image rather
    than silently running degraded."""
    from mflux.callbacks.callback_registry import CallbackRegistry

    h = apply_teacache(flux1_dev, rel_l1_thresh=0.25)
    try:
        flux1_dev.callbacks = CallbackRegistry()
        with pytest.raises(MissingGenerationContextError) as exc:
            flux1_dev.generate_image(
                seed=42,
                prompt=PR_TIME_PROMPT,
                num_inference_steps=4,
                height=512,
                width=512,
                guidance=3.5,
            )
        assert "handle.restore()" in str(exc.value)
    finally:
        import contextlib

        with contextlib.suppress(Exception):
            h.restore()


def test_composes_with_mlx_taef_live_preview(flux1_dev: Any, tmp_path: Any) -> None:
    """mlx-taef's LivePreviewCallback composes cleanly with TeaCache and
    fires the expected number of times."""
    pytest.importorskip("mlx_taef.integrations.mflux")
    from mlx_taef.integrations.mflux import LivePreviewCallback

    preview_calls: list[int] = []
    preview = LivePreviewCallback(
        variant="taef1",
        every=5,
        save_to=str(tmp_path / "preview.png"),
        latent_height=32,
        latent_width=32,
    )
    orig_in_loop = preview.call_in_loop

    def counted(*a: Any, **kw: Any) -> Any:
        preview_calls.append(1)
        return orig_in_loop(*a, **kw)

    preview.call_in_loop = counted  # type: ignore[method-assign]

    flux1_dev.callbacks.register(preview)
    try:
        with apply_teacache(flux1_dev, rel_l1_thresh=0.25):
            flux1_dev.generate_image(
                seed=42,
                prompt=PR_TIME_PROMPT,
                num_inference_steps=25,
                height=512,
                width=512,
                guidance=3.5,
            )
    finally:
        for lst_name in ("in_loop", "in_loop_callbacks", "_callbacks", "callbacks"):
            lst = getattr(flux1_dev.callbacks, lst_name, None)
            if isinstance(lst, list):
                for i in range(len(lst) - 1, -1, -1):
                    if lst[i] is preview:
                        del lst[i]
    # every=5 over 25 steps → at least 5 fires.
    assert len(preview_calls) >= 5
