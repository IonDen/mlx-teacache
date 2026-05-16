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
from mlx_teacache.coefficients import load_builtin
from mlx_teacache.integrations.mflux.forward import flux1_forward_with_gate
from mlx_teacache.stats import TeaCacheStats

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
    coefficients, _ = load_builtin("flux1-dev")
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
    downstream kernel dispatch (see
    docs/superpowers/notes/2026-05-14-task-25-mlx-nondeterminism*.md)."""
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
