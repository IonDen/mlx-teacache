# tests/test_forward_flux1.py
"""Unit tests for FLUX.1 forward helper.

These tests exercise the prelude/body/tail wiring of
`flux1_forward_with_gate` against a synthetic transformer that mimics
just enough of mflux's Transformer surface to run the code path. The
deep numerical parity test against real mflux weights lives in
`tests/test_parity_flux1.py` (Task 25).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest

from mlx_teacache.cache import TeaCacheState
from mlx_teacache.stats import TeaCacheStats
from mlx_teacache.variants.flux1_dev.config import COEFFICIENTS as _FLUX1_DEV_COEFFICIENTS
from mlx_teacache.variants.flux1_dev.integration import flux1_forward_with_gate

# ---------------------------------------------------------------------------
# Fake transformer
# ---------------------------------------------------------------------------


class _FakeJointBlock:
    """Just enough of JointTransformerBlock for the slow path's
    `_flux1_extract_mod_input` (block_0.norm1) — returns a tuple whose
    first element is a usable mod_in tensor. The fast path skips this
    entirely, so for fast-path tests it's never called."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def norm1(self, hidden_states: mx.array, text_embeddings: mx.array) -> tuple[mx.array, ...]:
        # Return a mod_in of shape matching hidden_states; other tuple elements
        # are placeholders.
        return (hidden_states, mx.zeros((1,)), mx.zeros((1,)), mx.zeros((1,)), mx.zeros((1,)))


class _FakeInner:
    """Synthetic Transformer surface — methods/attrs that
    `flux1_forward_with_gate` (and `_flux1_run_body`) touch."""

    def __init__(self, *, text_seq: int = 2, img_seq: int = 4, dim: int = 8) -> None:
        self.text_seq = text_seq
        self.img_seq = img_seq
        self.dim = dim
        self.transformer_blocks: list[Any] = [_FakeJointBlock(dim=dim)]
        self.single_transformer_blocks: list[Any] = [object()]
        # pos_embed / time_text_embed are passed through to compute_*; the fake
        # compute_* methods ignore them, so any sentinel works.
        self.pos_embed = object()
        self.time_text_embed = object()

    def x_embedder(self, hidden_states: mx.array) -> mx.array:
        return mx.zeros((1, self.img_seq, self.dim))

    def context_embedder(self, prompt_embeds: mx.array) -> mx.array:
        return mx.zeros((1, self.text_seq, self.dim))

    def compute_text_embeddings(
        self,
        t: int,
        pooled: mx.array,
        time_text_embed: Any,
        config: Any,
    ) -> mx.array:
        return mx.zeros((1, self.dim))

    def compute_rotary_embeddings(
        self,
        prompt_embeds: mx.array,
        pos_embed: Any,
        config: Any,
        kontext_image_ids: Any,
    ) -> mx.array:
        return mx.zeros((1, self.text_seq + self.img_seq, self.dim // 2))

    def _apply_joint_transformer_block(
        self,
        *,
        idx: int,
        block: Any,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: mx.array,
        controlnet_block_samples: Any,
    ) -> tuple[mx.array, mx.array]:
        return encoder_hidden_states, hidden_states

    def _apply_single_transformer_block(
        self,
        *,
        idx: int,
        block: Any,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: mx.array,
        controlnet_single_block_samples: Any,
    ) -> mx.array:
        return hidden_states

    def norm_out(self, x: mx.array, text_embeddings: mx.array) -> mx.array:
        return x

    def proj_out(self, x: mx.array) -> mx.array:
        return x


def _make_handle(
    *, rel_l1_thresh: float, skip_first: int = 0, skip_last: int = 0, num_inference_steps: int = 25
) -> Any:
    """Minimal handle stub with the attributes forward.py reads."""
    coefficients = _FLUX1_DEV_COEFFICIENTS
    state = SimpleNamespace(
        cache=TeaCacheState(),
        stats=TeaCacheStats(),
    )
    return SimpleNamespace(
        rel_l1_thresh=rel_l1_thresh,
        coefficients=coefficients,
        skip_first_n_steps=skip_first,
        skip_last_n_steps=skip_last,
        _state=state,
        _gen_ctx=SimpleNamespace(active_num_steps=num_inference_steps),
    )


def _run_one_step(handle: Any, *, t: int, num_inference_steps: int = 4) -> mx.array:
    inner = _FakeInner()
    config = SimpleNamespace(num_inference_steps=num_inference_steps)
    return flux1_forward_with_gate(
        inner,
        handle,
        t=t,
        config=config,
        hidden_states=mx.zeros((1, 4, 4)),
        prompt_embeds=mx.zeros((1, 2, 4)),
        pooled_prompt_embeds=mx.zeros((1, 4)),
    )


# ---------------------------------------------------------------------------
# Threshold-zero fast path
# ---------------------------------------------------------------------------


def test_threshold_zero_does_not_build_cache_tensors():
    """At rel_l1_thresh <= 0, no future step can ever skip — so the wrapper
    must NOT build cached_residual / previous_mod_input. Keeping those
    intermediates alive blocks Metal in-place buffer donation and perturbs
    downstream kernel dispatch."""
    handle = _make_handle(rel_l1_thresh=0.0)
    _run_one_step(handle, t=0)
    assert handle._state.cache.cached_residual is None
    assert handle._state.cache.previous_mod_input is None


def test_threshold_zero_records_computed_decision():
    handle = _make_handle(rel_l1_thresh=0.0)
    _run_one_step(handle, t=0)
    staged = handle._state.stats._staging.decisions
    assert len(staged) == 1
    assert staged[0].decision == "computed"
    assert staged[0].rel_l1 is None


def test_threshold_zero_advances_step_counter():
    handle = _make_handle(rel_l1_thresh=0.0)
    _run_one_step(handle, t=0)
    assert handle._state.cache.step_counter == 1


def test_threshold_zero_skip_window_validation_still_runs_at_t0():
    """The skip-window validation lives outside the gate; the fast path
    must still raise if the configuration is invalid."""
    from mlx_teacache.errors import InvalidStepWindowError

    handle = _make_handle(rel_l1_thresh=0.0, skip_first=2, skip_last=2, num_inference_steps=3)
    with pytest.raises(InvalidStepWindowError):
        _run_one_step(handle, t=0, num_inference_steps=3)


def test_threshold_negative_treated_as_zero_fast_path():
    """Negative thresholds short-circuit identically to zero."""
    handle = _make_handle(rel_l1_thresh=-0.5)
    _run_one_step(handle, t=0)
    assert handle._state.cache.cached_residual is None
    assert handle._state.cache.previous_mod_input is None


# ---------------------------------------------------------------------------
# Slow path (positive threshold) — sanity that we didn't break anything
# ---------------------------------------------------------------------------


def test_positive_threshold_first_step_seeds_cache():
    """At rel_l1_thresh > 0, the first step must seed cached_residual and
    previous_mod_input so that subsequent steps can potentially skip."""
    handle = _make_handle(rel_l1_thresh=0.25)
    _run_one_step(handle, t=0)
    assert handle._state.cache.cached_residual is not None
    assert handle._state.cache.previous_mod_input is not None


def test_slow_path_raises_on_mod_input_shape_drift():
    """The defensive shape check (integration.py:315) catches a cached mod_in
    whose shape no longer matches the current step's — e.g. a resolution change
    mid-generation. Seed a previous_mod_input with a clearly different shape than
    the fake's mod_in (1, img_seq=4, dim=8) and run one slow-path step."""
    from mlx_teacache.errors import TransformerShapeError

    handle = _make_handle(rel_l1_thresh=0.25)
    handle._state.cache.previous_mod_input = mx.zeros((1, 99, 8))
    with pytest.raises(TransformerShapeError) as excinfo:
        _run_one_step(handle, t=9)
    # bug caught: reporting mflux's absolute scheduler step (9) instead of the
    # per-generation counter (0) — every other variant reports the counter.
    assert excinfo.value.step_idx == 0


# ---------------------------------------------------------------------------
# Skip-reconstruction identity
# ---------------------------------------------------------------------------


class _FakeInnerWithOffset(_FakeInner):
    """_FakeInner whose joint block adds a fill of 0.5 to hidden_states.

    This makes body_out_concat != body_in_concat (the offset is +0.5 on the
    image-token slice), so cached_residual is a non-zero tensor and the
    skip-reconstruction identity can fail for the right reason if the
    reconstruction formula is broken (e.g., `+` flipped to `-`).
    """

    def _apply_joint_transformer_block(
        self,
        *,
        idx: int,
        block: Any,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: mx.array,
        controlnet_block_samples: Any,
    ) -> tuple[mx.array, mx.array]:
        return encoder_hidden_states, hidden_states + mx.full(hidden_states.shape, 0.5)


def test_skip_step_reconstructs_body_out_from_cached_residual():
    """Step 1 skip path: body_out_concat = body_in_concat + cached_residual.

    Uses coefficients=(0,0,0,0,0) so poly_eval always returns 0, new_acc=0
    which is < rel_l1_thresh=0.5, forcing the gate to "skipped" on step 1.

    Mutation check: flip `+` to `-` in the skip branch (integration.py line ~365)
    and the assertion fails — the fake produces 0.5-nonzero residuals so the sign
    change is observable. Revert the src change before committing.
    """
    # All-zero coefficients → poly_eval(x) = 0 for any x → accumulated = 0 < thresh → skip
    zero_coeffs = (0.0, 0.0, 0.0, 0.0, 0.0)
    handle = SimpleNamespace(
        rel_l1_thresh=0.5,
        coefficients=zero_coeffs,
        skip_first_n_steps=0,
        skip_last_n_steps=0,
        _state=SimpleNamespace(
            cache=TeaCacheState(),
            stats=TeaCacheStats(),
        ),
        _gen_ctx=SimpleNamespace(active_num_steps=4),
    )

    inner = _FakeInnerWithOffset()
    common_kwargs: dict[str, Any] = dict(
        config=SimpleNamespace(num_inference_steps=4),
        hidden_states=mx.zeros((1, 4, 4)),
        prompt_embeds=mx.zeros((1, 2, 4)),
        pooled_prompt_embeds=mx.zeros((1, 4)),
    )

    # Step 0 (seed): computes, caches residual = body_out_concat - body_in_concat.
    flux1_forward_with_gate(inner, handle, t=0, **common_kwargs)
    assert len(handle._state.stats._staging.decisions) == 1, "seed step must stage one decision"

    # Capture cached_residual immediately after the seed step.
    cached_residual = handle._state.cache.cached_residual
    assert cached_residual is not None, "seed step must populate cached_residual"
    # Pin the cache-WRITE formula against an independently derived reference:
    # body_in_concat is all-zeros (both embedders return zeros) and the fake
    # adds +0.5 to the image-token slice only, so
    # residual = body_out - body_in = concat(zeros(text_seq), full(0.5, img_seq)).
    # Deriving this from the fake's arithmetic (not from SUT state) means a
    # sign-flip at the cache-write site reds here instead of cancelling out.
    expected_residual = mx.concatenate(
        [
            mx.zeros((1, inner.text_seq, inner.dim)),
            mx.full((1, inner.img_seq, inner.dim), 0.5),
        ],
        axis=1,
    )
    mx.eval(cached_residual)
    assert bool(mx.all(mx.abs(cached_residual - expected_residual) < 1e-5)), (
        "cached_residual does not match the independently derived seed residual: "
        f"max abs diff = {float(mx.max(mx.abs(cached_residual - expected_residual)))}"
    )

    # Build expected skip output: (body_in_concat_step1 + cached_residual)[:, enc_dim:, :]
    # x_embedder and context_embedder both return zeros, so body_in_concat_step1 = zeros.
    enc_dim = inner.text_seq  # 2
    body_in_concat_step1 = mx.zeros((1, inner.text_seq + inner.img_seq, inner.dim))
    expected_body_out_concat = body_in_concat_step1 + cached_residual
    # The tail slices off the encoder tokens and applies norm_out / proj_out (both identity).
    expected = expected_body_out_concat[:, enc_dim:, ...]

    # Step 1 (skip): gate should return "skipped"; forward reuses cached_residual.
    actual = flux1_forward_with_gate(inner, handle, t=1, **common_kwargs)

    # Verify skip was recorded.
    staged = handle._state.stats._staging.decisions
    assert len(staged) == 2, f"expected 2 staged decisions, got {len(staged)}"
    assert staged[1].decision == "skipped", f"step 1 should be skipped, got {staged[1].decision!r}"

    # Verify reconstruction identity (host-synced).
    mx.eval(actual, expected)
    assert bool(mx.all(mx.abs(actual - expected) < 1e-5)), (
        f"skip reconstruction mismatch: max abs diff = {float(mx.max(mx.abs(actual - expected)))}"
    )
