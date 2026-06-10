# tests/test_cfg_branch_independence.py
"""Verify that the CFG-gated FLUX.2 forward keeps positive and negative branch
residuals independent: each branch is cached and reconstructed from its OWN
residual, never the other branch's.

Module-level import contract
----------------------------
`from mlx_teacache.variants.flux2_klein_base_4b.integration import
 flux2_cfg_forward_with_gate` does NOT pull mflux at import time — confirmed
by `uv run python -c "import sys; sys.modules['mflux'] = None; from
mlx_teacache.variants.flux2_klein_base_4b.integration import
flux2_cfg_forward_with_gate; print('OK')"` printing OK.

However, *calling* the function triggers a lazy `from mflux.models.common.config.
model_config import ModelConfig` inside the function body. The test therefore
requires mflux at runtime and is listed in conftest._MFLUX_FILES so the
pure-core CI lane skips it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.core as mx

from mlx_teacache._kernel.cache import TeaCacheState
from mlx_teacache._kernel.stats import TeaCacheStats
from mlx_teacache.variants.flux2_klein_base_4b.integration import flux2_cfg_forward_with_gate

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

_TEXT_SEQ = 2
_IMG_SEQ = 3
_DIM = 8
_ROPE_DIM = 4

# Branch-identifying fill values for prompt_embeds / negative_prompt_embeds.
# context_embedder doubles them (see _FakeCFGFlux2Inner below), so the
# embedded encoders are 1.0 (pos) and -0.5 (neg), and the per-branch
# residuals work out to uniform fills of 1.0 and -0.5 respectively — see
# the docstring of test_cfg_branches_reconstruct_from_own_residual.
_POS_FILL = 0.5  # prompt_embeds fill value
_NEG_FILL = -0.25  # negative_prompt_embeds fill value


# ---------------------------------------------------------------------------
# Fake inner transformer for CFG tests
# ---------------------------------------------------------------------------


class _FakeCFGBlock:
    """Transformer block stub.

    Arithmetic (so residuals are non-zero and branch-distinguishable):
      - doubles encoder_hidden_states
      - adds mean(encoder_hidden_states) as a scalar offset to hidden_states

    This couples the *image* portion of the body output to the branch's
    encoder, so both text and image positions of the residual differ between
    positive and negative branches.

    `norm1` must exist because `_flux2_extract_mod_input` reaches into
    `inner.transformer_blocks[0].norm1` to compute the gating signal — the
    same pattern as `_FakeFlux2Block.norm1` in test_forward_flux2.py."""

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
        offset = mx.mean(encoder_hidden_states)  # scalar, branch-specific
        return encoder_hidden_states * 2.0, hidden_states + offset


class _FakeCFGSingleBlock:
    """Single-stream block stub — pass-through."""

    def __call__(
        self,
        hidden_states: mx.array,
        temb_mod_params: Any,
        image_rotary_emb: Any,
    ) -> mx.array:
        return hidden_states


class _FakeCFGFlux2Inner:
    """Minimal Flux2Transformer surface for CFG tests.

    Key design choices:
    * context_embedder doubles its input so enc_pos = 2*prompt_embeds and
      enc_neg = 2*negative_prompt_embeds.
    * x_embedder returns zeros — body_in is always zero.
    * Transformer block doubles enc and injects mean(enc) into image positions
      (see _FakeCFGBlock above).
    * norm1 is identity — the mod_in TeaCache signal equals body_in (zeros),
      so rel_l1 == 0 on step 1 and the zero-coefficient polynomial accumulates
      0 < threshold → skip.
    * norm_out and proj_out are identity (pass-through tail)."""

    def __init__(self) -> None:
        self.transformer_blocks = [_FakeCFGBlock()]
        self.single_transformer_blocks = [_FakeCFGSingleBlock()]

    def x_embedder(self, hidden_states: mx.array) -> mx.array:
        return mx.zeros((1, _IMG_SEQ, _DIM))

    def context_embedder(self, encoder_hidden_states: mx.array) -> mx.array:
        # Double the input — makes per-branch residuals analytically tractable.
        return encoder_hidden_states * 2.0

    def pos_embed(self, ids: mx.array) -> tuple[mx.array, mx.array]:
        seq = int(ids.shape[0])
        return mx.zeros((seq, _ROPE_DIM)), mx.zeros((seq, _ROPE_DIM))

    def time_guidance_embed(self, timestep: mx.array, _ignored: Any) -> mx.array:
        return mx.zeros((1, _DIM))

    def _mod_set(self, b: int) -> tuple[mx.array, mx.array, mx.array]:
        return (
            mx.zeros((b, 1, _DIM)),  # shift  → zero shift: mod_in = body_in = zeros
            mx.zeros((b, 1, _DIM)),  # scale  → zero scale: (1+0)*norm_in = norm_in
            mx.zeros((b, 1, _DIM)),  # gate
        )

    def double_stream_modulation_img(self, temb: mx.array) -> tuple[Any, Any]:
        b = int(temb.shape[0])
        return (self._mod_set(b), self._mod_set(b))

    def double_stream_modulation_txt(self, temb: mx.array) -> tuple[Any, Any]:
        b = int(temb.shape[0])
        return (self._mod_set(b), self._mod_set(b))

    def single_stream_modulation(self, temb: mx.array) -> tuple[Any, Any]:
        b = int(temb.shape[0])
        params = (mx.zeros((b, _DIM)), mx.zeros((b, _DIM)), mx.zeros((b, _DIM)))
        return (params, None)

    def norm_out(self, x: mx.array, temb: mx.array) -> mx.array:
        return x

    def proj_out(self, x: mx.array) -> mx.array:
        return x


# ---------------------------------------------------------------------------
# Handle factory
# ---------------------------------------------------------------------------


def _make_cfg_handle(*, num_inference_steps: int = 2) -> Any:
    """Minimal handle for CFG forward tests.

    Uses zero-valued degree-4 coefficients so poly_eval always returns 0.0:
    accumulated_distance stays 0.0 on every step after the seed, which is
    strictly below rel_l1_thresh=1.0 — guaranteeing a skip on step 1."""
    zero_coeffs: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    state = SimpleNamespace(cache=TeaCacheState(), stats=TeaCacheStats())
    gen_ctx = SimpleNamespace(active_num_steps=num_inference_steps)
    return SimpleNamespace(
        rel_l1_thresh=1.0,
        coefficients=zero_coeffs,
        skip_first_n_steps=0,
        skip_last_n_steps=0,
        _state=state,
        _gen_ctx=gen_ctx,
    )


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------


def _make_inputs() -> dict[str, Any]:
    """Construct the keyword arguments for flux2_cfg_forward_with_gate.

    prompt_embeds / negative_prompt_embeds are uniform fills at _POS_FILL /
    _NEG_FILL — chosen so that after context_embedder doubles them the
    per-branch residuals are uniform fills of 1.0 and -0.5 throughout the
    full (text + image) sequence dimension."""
    return dict(
        hidden_states=mx.zeros((1, _IMG_SEQ, _DIM)),
        prompt_embeds=mx.full((1, _TEXT_SEQ, _DIM), _POS_FILL),
        text_ids=mx.zeros((_TEXT_SEQ, 3)),
        negative_prompt_embeds=mx.full((1, _TEXT_SEQ, _DIM), _NEG_FILL),
        negative_text_ids=mx.zeros((_TEXT_SEQ, 3)),
        guidance=2.0,
        timestep=mx.array([500.0]),
        img_ids=mx.zeros((_IMG_SEQ, 3)),
    )


# ---------------------------------------------------------------------------
# The main regression test
# ---------------------------------------------------------------------------


def test_cfg_branches_reconstruct_from_own_residual() -> None:
    """After a seed step both branch residuals are independently correct; after
    a skip step each branch reconstructs from its OWN residual.

    Arithmetic derivation (all shapes (1, TEXT_SEQ+IMG_SEQ, DIM)):
    ---------------------------------------------------------------
    Let pm = _POS_FILL = 0.5, nm = _NEG_FILL = -0.25.

    context_embedder doubles: enc_pos = 2*pm = 1.0, enc_neg = 2*nm = -0.5.
    x_embedder returns zeros: body_in = 0.

    _FakeCFGBlock returns (enc*2, hs + mean(enc)):
      pos branch: (2*enc_pos, body_in + mean(enc_pos)) = (2.0_text, 1.0_img)
      neg branch: (2*enc_neg, body_in + mean(enc_neg)) = (-1.0_text, -0.5_img)

    body_in_concat_pos = concat([enc_pos, body_in]) = concat([1.0_text, 0_img])
    body_in_concat_neg = concat([enc_neg, body_in]) = concat([-0.5_text, 0_img])

    body_out_pos after single-block pass-through = concat([2.0_text, 1.0_img])
    body_out_neg after single-block pass-through = concat([-1.0_text, -0.5_img])

    residual_pos = body_out_pos - body_in_concat_pos
                 = concat([2.0-1.0, 1.0-0]) = 1.0 fill  ← uniform 1.0
    residual_neg = body_out_neg - body_in_concat_neg
                 = concat([-1.0-(-0.5), -0.5-0]) = -0.5 fill  ← uniform -0.5

    Negative control: residual_pos (1.0) ≠ residual_neg (-0.5), so a
    pos/neg swap at the cache-WRITE site would make assertion 1 fail.

    For the skip step (step 1, same inputs):
      body_in_concat_pos + residual_pos = concat([1.0, 0]) + 1.0 = concat([2.0, 1.0])
      noise_pos = concat([2.0, 1.0])[:, _TEXT_SEQ:, ...] = 1.0_img
      body_in_concat_neg + residual_neg = concat([-0.5, 0]) + (-0.5) = concat([-1.0, -0.5])
      noise_neg = concat([-1.0, -0.5])[:, _TEXT_SEQ:, ...] = -0.5_img

    CFG output (guidance=2.0):
      noise_neg + 2.0*(noise_pos - noise_neg) = -0.5 + 2.0*(1.0-(-0.5)) = 2.5

    A pos/neg swap at the cache-READ site would give:
      noise_pos_swapped = body_in_concat_pos + residual_neg → noise = -0.5_img
      noise_neg_swapped = body_in_concat_neg + residual_pos → noise = 1.0_img
      output_swapped = 1.0 + 2.0*(-0.5-1.0) = -2.0  ← differs from 2.5 ✓
    """
    inner = _FakeCFGFlux2Inner()
    handle = _make_cfg_handle(num_inference_steps=2)
    inputs = _make_inputs()

    # ---- Seed step (step 0) ------------------------------------------------
    # gate_step: previous_mod_input is None → should_compute=True,
    # should_update_cache=True → both residuals written.
    _ = flux2_cfg_forward_with_gate(inner, handle, **inputs)
    mx.eval(handle._state.cache.cached_residual)
    mx.eval(handle._state.cache.cached_residual_neg)

    seq_len = _TEXT_SEQ + _IMG_SEQ  # total concatenated sequence length

    # 1. Pin each branch's cached residual against an INDEPENDENTLY derived
    #    expected tensor (from the fake's known arithmetic — NOT read back
    #    from the same state object).
    expected_residual_pos = mx.ones((1, seq_len, _DIM))  # uniform 1.0
    expected_residual_neg = mx.full((1, seq_len, _DIM), -0.5)  # uniform -0.5

    # Negative control: the two expected residuals differ — a swap at the
    # WRITE site would make one of the assertions below fail.
    assert float(mx.mean(expected_residual_pos)) != float(mx.mean(expected_residual_neg)), (
        "Expected residuals must differ so that a pos/neg swap is detectable"
    )

    assert bool(mx.all(mx.abs(handle._state.cache.cached_residual - expected_residual_pos) < 1e-5)), (
        f"cached_residual (pos) mismatch; got mean={float(mx.mean(handle._state.cache.cached_residual)):.4f}, "
        f"expected mean={float(mx.mean(expected_residual_pos)):.4f}"
    )
    assert bool(mx.all(mx.abs(handle._state.cache.cached_residual_neg - expected_residual_neg) < 1e-5)), (
        f"cached_residual_neg (neg) mismatch; got mean={float(mx.mean(handle._state.cache.cached_residual_neg)):.4f}, "
        f"expected mean={float(mx.mean(expected_residual_neg)):.4f}"
    )

    # ---- Skip step (step 1) ------------------------------------------------
    # gate_step with zero-coeff polynomial: new_acc=0 < thresh=1.0 → skipped.
    # Each branch reconstructs: body_out = body_in_concat + own_residual.
    skip_out = flux2_cfg_forward_with_gate(inner, handle, **inputs)
    mx.eval(skip_out)

    # The gate must have actually SKIPPED step 1 — the step-invariant fake
    # makes the compute path produce the same 2.5, so without this assertion
    # an always-compute gate regression would stay green.
    staged = handle._state.stats._staging.decisions
    assert len(staged) == 2, f"expected 2 staged decisions, got {len(staged)}"
    assert staged[0].decision == "computed", f"step 0 should compute, got {staged[0].decision!r}"
    assert staged[1].decision == "skipped", f"step 1 should skip, got {staged[1].decision!r}"

    # 2. The combined CFG output for the skip step must equal 2.5 (derived
    #    analytically above from body_in_concat + own_residual + tail + CFG).
    #    A pos/neg residual swap at the READ site gives -2.0 instead.
    expected_skip_output = mx.full((1, _IMG_SEQ, _DIM), 2.5)
    assert bool(mx.all(mx.abs(skip_out - expected_skip_output) < 1e-4)), (
        f"skip-step output mismatch; got mean={float(mx.mean(skip_out)):.4f}, "
        f"expected 2.5 (a read-site swap would give -2.0)"
    )

    # 3. Sanity: step counter advanced past both steps.
    assert handle._state.cache.step_counter == 2
