# tests/test_forward_flux2.py
"""Unit tests for FLUX.2 forward helper.

These tests exercise `flux2_forward_with_gate` against a synthetic
Flux2Transformer-shaped fake. They focus on the threshold-zero fast path
(v2.6) — the invariant that at `rel_l1_thresh <= 0` the wrapper does NOT
build the gating tensors (`mod_in`, `body_in_concat`, `cached_residual`,
`previous_mod_input`). Deep numerical parity against real Flux2Klein
weights lives in `test_parity_flux2.py` (Task 26).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.core as mx

from mlx_teacache.cache import TeaCacheState
from mlx_teacache.coefficients import load_builtin
from mlx_teacache.integrations.mflux.forward import flux2_forward_with_gate
from mlx_teacache.stats import TeaCacheStats

# ---------------------------------------------------------------------------
# Fake Flux2Transformer
# ---------------------------------------------------------------------------


class _FakeFlux2Block:
    """Stand-in for Flux2TransformerBlock — pass-through.

    `norm1` must exist because `_flux2_extract_mod_input` reaches into
    `inner.transformer_blocks[0].norm1` to apply the ada-LN-zero
    normalization that produces the gating signal (mirrors real mflux's
    Flux2TransformerBlock.__call__ lines 32-36)."""

    def __init__(self) -> None:
        self.norm1 = lambda x: x  # pass-through LayerNorm

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        temb_mod_params_img: Any,
        temb_mod_params_txt: Any,
        image_rotary_emb: Any,
    ) -> tuple[mx.array, mx.array]:
        return encoder_hidden_states, hidden_states


class _FakeFlux2SingleBlock:
    def __call__(
        self,
        hidden_states: mx.array,
        temb_mod_params: Any,
        image_rotary_emb: Any,
    ) -> mx.array:
        return hidden_states


class _FakeFlux2Inner:
    """Synthetic Flux2Transformer surface — just enough for the fast path."""

    def __init__(
        self,
        *,
        text_seq: int = 2,
        img_seq: int = 4,
        dim: int = 8,
        rope_dim: int = 4,
    ) -> None:
        self.text_seq = text_seq
        self.img_seq = img_seq
        self.dim = dim
        self.rope_dim = rope_dim
        self.transformer_blocks: list[Any] = [_FakeFlux2Block()]
        self.single_transformer_blocks: list[Any] = [_FakeFlux2SingleBlock()]

    def x_embedder(self, hidden_states: mx.array) -> mx.array:
        return mx.zeros((1, self.img_seq, self.dim))

    def context_embedder(self, encoder_hidden_states: mx.array) -> mx.array:
        return mx.zeros((1, self.text_seq, self.dim))

    def pos_embed(self, ids: mx.array) -> tuple[mx.array, mx.array]:
        # Flux2 RoPE returns (cos, sin) tuples; shape (seq, rope_dim).
        seq = int(ids.shape[0])
        return mx.zeros((seq, self.rope_dim)), mx.zeros((seq, self.rope_dim))

    def time_guidance_embed(self, timestep: mx.array, _ignored: Any) -> mx.array:
        return mx.zeros((1, self.dim))

    def _mod_set(self, b: int) -> tuple[mx.array, mx.array, mx.array]:
        # Real Flux2Modulation expand_dims to [B, 1, D] before splitting.
        return (
            mx.zeros((b, 1, self.dim)),  # shift
            mx.zeros((b, 1, self.dim)),  # scale
            mx.zeros((b, 1, self.dim)),  # gate
        )

    def double_stream_modulation_img(self, temb: mx.array) -> tuple[Any, Any]:
        # Real Flux2Modulation with mod_param_sets=2 returns a nested tuple:
        #   ((shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp))
        b = int(temb.shape[0])
        return (self._mod_set(b), self._mod_set(b))

    def double_stream_modulation_txt(self, temb: mx.array) -> tuple[Any, Any]:
        b = int(temb.shape[0])
        return (self._mod_set(b), self._mod_set(b))

    def single_stream_modulation(self, temb: mx.array) -> tuple[Any, Any]:
        b = int(temb.shape[0])
        # mflux returns a list/tuple where [0] is the modulation params
        params = (
            mx.zeros((b, self.dim)),
            mx.zeros((b, self.dim)),
            mx.zeros((b, self.dim)),
        )
        return (params, None)

    def norm_out(self, x: mx.array, temb: mx.array) -> mx.array:
        return x

    def proj_out(self, x: mx.array) -> mx.array:
        return x


def _make_handle(*, rel_l1_thresh: float, num_inference_steps: int = 4) -> Any:
    """Minimal handle stub. FLUX.2 forward reads `handle._gen_ctx.active_num_steps`
    for the gate (not from config), so we set it directly."""
    coefficients, _ = load_builtin("flux2-klein-4b")
    state = SimpleNamespace(cache=TeaCacheState(), stats=TeaCacheStats())
    gen_ctx = SimpleNamespace(active_num_steps=num_inference_steps)
    return SimpleNamespace(
        rel_l1_thresh=rel_l1_thresh,
        coefficients=coefficients,
        skip_first_n_steps=0,
        skip_last_n_steps=0,
        _state=state,
        _gen_ctx=gen_ctx,
    )


def _run_one_step(handle: Any) -> mx.array:
    inner = _FakeFlux2Inner()
    return flux2_forward_with_gate(
        inner,
        handle,
        hidden_states=mx.zeros((1, 4, 4)),
        encoder_hidden_states=mx.zeros((1, 2, 4)),
        timestep=mx.array([1000.0]),
        img_ids=mx.zeros((4, 3)),
        txt_ids=mx.zeros((2, 3)),
    )


# ---------------------------------------------------------------------------
# Threshold-zero fast path
# ---------------------------------------------------------------------------


def test_threshold_zero_does_not_build_cache_tensors():
    """At rel_l1_thresh <= 0 the FLUX.2 fast path must NOT build
    cached_residual / previous_mod_input. Mirrors the FLUX.1 invariant."""
    handle = _make_handle(rel_l1_thresh=0.0)
    _run_one_step(handle)
    assert handle._state.cache.cached_residual is None
    assert handle._state.cache.previous_mod_input is None


def test_threshold_zero_records_computed_decision():
    handle = _make_handle(rel_l1_thresh=0.0)
    _run_one_step(handle)
    staged = handle._state.stats._staging.decisions
    assert len(staged) == 1
    assert staged[0].decision == "computed"
    assert staged[0].rel_l1 is None


def test_threshold_zero_advances_step_counter():
    handle = _make_handle(rel_l1_thresh=0.0)
    _run_one_step(handle)
    assert handle._state.cache.step_counter == 1


def test_threshold_negative_treated_as_zero_fast_path():
    handle = _make_handle(rel_l1_thresh=-0.5)
    _run_one_step(handle)
    assert handle._state.cache.cached_residual is None
    assert handle._state.cache.previous_mod_input is None


# ---------------------------------------------------------------------------
# Slow path sanity check
# ---------------------------------------------------------------------------


def test_positive_threshold_first_step_seeds_cache():
    """At rel_l1_thresh > 0 the FLUX.2 slow path must seed cached_residual
    and previous_mod_input so a future step can potentially skip."""
    handle = _make_handle(rel_l1_thresh=0.25)
    _run_one_step(handle)
    assert handle._state.cache.cached_residual is not None
    assert handle._state.cache.previous_mod_input is not None
