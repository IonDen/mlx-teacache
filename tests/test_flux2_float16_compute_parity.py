"""Weight-free parity of the gated FLUX.2 forwards against a tiny REAL mflux Flux2Transformer, under mflux's opt-in
float16 compute (mflux PR 820, `compute_precision`).

The setting lives on the attention and feed-forward modules and the weights are cast in place, so the gated forwards,
which call those modules, must equal the transformer's own `__call__` bit for bit at threshold 0, with the option on
and off, with and without CFG. An mflux without the option skips the float16 cases."""

import mlx.core as mx
import pytest
from mlx.utils import tree_map

from mlx_teacache.variants.flux2_klein_base_4b.integration import (
    flux2_cfg_forward_with_gate,
    flux2_forward_with_gate,
)
from tests._mflux_surface import installed_vcs_commit, require_or_skip
from tests.test_forward_flux2 import _make_handle

pytest.importorskip("mflux")
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer  # noqa: E402

_HAS_FLOAT16_COMPUTE = hasattr(Flux2Transformer, "apply_compute_precision")
_COMPUTE = [
    pytest.param(False, id="model-precision"),
    pytest.param(True, id="float16-compute"),
]

_IMG_TOKENS = 6
_TXT_TOKENS = 4
_TIMESTEP = mx.array([0.5])


def _tiny_transformer(*, float16_compute: bool) -> Flux2Transformer:
    mx.random.seed(0)
    transformer = Flux2Transformer(
        in_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=24,
        timestep_guidance_channels=32,
        axes_dims_rope=(4, 4, 4, 4),
    )
    # Default init is small enough that the residual stream's bfloat16 rounding hides float16-vs-bfloat16 compute in
    # the output; larger block weights let the option show, and both builds get the same weights.
    for blocks in (transformer.transformer_blocks, transformer.single_transformer_blocks):
        for block in blocks:
            block.update(tree_map(lambda w: w * 4.0, block.parameters()))
    transformer.set_dtype(mx.bfloat16)
    if float16_compute:
        require_or_skip(
            _HAS_FLOAT16_COMPUTE,
            vcs_commit=installed_vcs_commit("mflux"),
            feature="float16 compute option (PR 820)",
        )
        # The call mflux's initializer makes last, after weights and LoRA.
        from mflux.models.common.compute_precision import ComputePrecision

        transformer.apply_compute_precision(ComputePrecision(mx.float16))
    return transformer


def _ids(tokens: int) -> mx.array:
    return mx.stack([mx.zeros(tokens), mx.arange(tokens), mx.zeros(tokens), mx.zeros(tokens)], axis=-1)


def _inputs() -> dict[str, mx.array]:
    mx.random.seed(1)
    return {
        "latents": mx.random.normal((1, _IMG_TOKENS, 8)).astype(mx.bfloat16),
        "pos": mx.random.normal((1, _TXT_TOKENS, 24)).astype(mx.bfloat16),
        "neg": mx.random.normal((1, _TXT_TOKENS, 24)).astype(mx.bfloat16),
    }


@pytest.mark.parametrize("float16_compute", _COMPUTE)
def test_gated_forward_at_threshold_zero_equals_the_plain_transformer(float16_compute: bool) -> None:
    """Bug: the gated forward bypasses or re-implements the attention/feed-forward modules, so a TeaCache run on a
    model built with float16 compute is not the model mflux runs."""
    transformer = _tiny_transformer(float16_compute=float16_compute)
    x = _inputs()
    expected = transformer(
        hidden_states=x["latents"],
        encoder_hidden_states=x["pos"],
        timestep=_TIMESTEP,
        img_ids=_ids(_IMG_TOKENS),
        txt_ids=_ids(_TXT_TOKENS),
    )
    actual = flux2_forward_with_gate(
        transformer,
        _make_handle(rel_l1_thresh=0.0),
        hidden_states=x["latents"],
        encoder_hidden_states=x["pos"],
        timestep=_TIMESTEP,
        img_ids=_ids(_IMG_TOKENS),
        txt_ids=_ids(_TXT_TOKENS),
    )
    assert actual.dtype == expected.dtype
    diff = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
    assert mx.array_equal(actual, expected).item(), f"max abs difference {diff}"


@pytest.mark.parametrize("float16_compute", _COMPUTE)
def test_cfg_gated_forward_at_threshold_zero_equals_the_plain_cfg_combine(float16_compute: bool) -> None:
    """Bug: the CFG gated forward differs from mflux's two transformer calls plus `neg + g * (pos - neg)` for either
    branch, with the modules' float16 compute on or off."""
    transformer = _tiny_transformer(float16_compute=float16_compute)
    x = _inputs()
    guidance = 4.0
    pos = transformer(
        hidden_states=x["latents"],
        encoder_hidden_states=x["pos"],
        timestep=_TIMESTEP,
        img_ids=_ids(_IMG_TOKENS),
        txt_ids=_ids(_TXT_TOKENS),
    )
    neg = transformer(
        hidden_states=x["latents"],
        encoder_hidden_states=x["neg"],
        timestep=_TIMESTEP,
        img_ids=_ids(_IMG_TOKENS),
        txt_ids=_ids(_TXT_TOKENS),
    )
    expected = neg + guidance * (pos - neg)
    actual = flux2_cfg_forward_with_gate(
        transformer,
        _make_handle(rel_l1_thresh=0.0),
        hidden_states=x["latents"],
        prompt_embeds=x["pos"],
        text_ids=_ids(_TXT_TOKENS),
        negative_prompt_embeds=x["neg"],
        negative_text_ids=_ids(_TXT_TOKENS),
        guidance=guidance,
        timestep=_TIMESTEP,
        img_ids=_ids(_IMG_TOKENS),
    )
    assert actual.dtype == expected.dtype
    diff = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
    assert mx.array_equal(actual, expected).item(), f"max abs difference {diff}"


def test_float16_compute_changes_the_plain_output_so_the_parity_cases_are_not_vacuous() -> None:
    """Bug: `apply_compute_precision` leaves the tiny transformer unchanged, so the float16-compute parity cases
    compare two identical model-precision runs and would stay green if the gated forward bypassed the option."""
    x = _inputs()
    outs = [
        _tiny_transformer(float16_compute=flag)(
            hidden_states=x["latents"],
            encoder_hidden_states=x["pos"],
            timestep=_TIMESTEP,
            img_ids=_ids(_IMG_TOKENS),
            txt_ids=_ids(_TXT_TOKENS),
        )
        for flag in (False, True)
    ]
    assert not mx.array_equal(outs[0], outs[1]).item()
