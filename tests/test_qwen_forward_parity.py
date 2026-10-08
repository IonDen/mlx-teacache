"""Weight-free parity of the gated Qwen-Image forward against a two-block mflux QwenTransformer with random weights.

The gated forward re-walks QwenTransformer.__call__ (prelude, the blocks, norm_out + proj_out). On a computed step it
must equal the plain transformer bit for bit, for both CFG branches, in bfloat16 and with 4-bit quantized linears. Its
gate signal must be the modulated image input mflux's block 0 computes. mflux 0.22 added the zero_cond_t path for
Qwen-Image-Edit-2511 to that forward and to the block's modulation; it is off for text-to-image, so the copy, which
does not take it, still has to match, and detection must refuse a transformer built with it on. mflux 0.22 also loads
a bias for norm_out, which the copy reaches by calling the module; the bias is set to non-zero values here so a tail
that skipped the module would show."""

import inspect
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_teacache.variants.qwen_image.detect import matches
from mlx_teacache.variants.qwen_image.integration import (
    _InternalHandle,
    _qwen_prelude,
    _qwen_signal_a,
    qwen_forward_with_gate,
)

pytest.importorskip("mflux")
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer  # noqa: E402
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock  # noqa: E402

_HEIGHT = _WIDTH = 64  # 4 x 4 latent patches
_TOKENS = (_HEIGHT // 16) * (_WIDTH // 16)
_PRECISIONS = ["bf16", "q4"]


def _quantizable(_path: str, module: nn.Module) -> bool:
    # mflux quantizes the linears whose input width the group size divides; txt_in (32 wide here) stays bfloat16.
    return isinstance(module, nn.Linear) and module.weight.shape[-1] % 64 == 0


def _tiny_transformer(precision: str = "bf16", **overrides: object) -> QwenTransformer:
    mx.random.seed(0)
    # Random weights. attention_head_dim stays 128: the RoPE axes are fixed at [16, 56, 56].
    transformer = QwenTransformer(
        num_layers=2, attention_head_dim=128, num_attention_heads=1, joint_attention_dim=32, **overrides
    )
    has_norm_bias = "bias" in transformer.norm_out.linear
    if has_norm_bias:
        transformer.norm_out.linear.bias = mx.random.normal(transformer.norm_out.linear.bias.shape)
    # Loaded weights are bfloat16.
    transformer.set_dtype(mx.bfloat16)
    if precision == "q4":
        nn.quantize(transformer, group_size=64, bits=4, class_predicate=_quantizable)
        assert isinstance(transformer.norm_out.linear, nn.QuantizedLinear)
        assert ("bias" in transformer.norm_out.linear) == has_norm_bias
    return transformer


def _config() -> SimpleNamespace:
    sigmas = mx.array([1.0, 0.75, 0.5, 0.25, 0.0])
    return SimpleNamespace(
        height=_HEIGHT,
        width=_WIDTH,
        num_inference_steps=4,
        scheduler=SimpleNamespace(sigmas=sigmas, timesteps=sigmas * 1000),
    )


def _inputs() -> tuple[mx.array, mx.array, mx.array, mx.array]:
    mx.random.seed(1)
    latents = mx.random.normal((1, _TOKENS, 64)).astype(mx.bfloat16)
    prompt_pos = mx.random.normal((1, 5, 32)).astype(mx.bfloat16)
    prompt_neg = mx.random.normal((1, 5, 32)).astype(mx.bfloat16)
    mask = mx.ones((1, 5))
    return latents, prompt_pos, prompt_neg, mask


def _handle(rel_l1_thresh: float) -> _InternalHandle:
    # Step 1 computes through the gate's seed guard (step 0 is forced and caches nothing) and writes the cache;
    # the constant coefficients would only matter from a third step pair on.
    handle = _InternalHandle(
        rel_l1_thresh=rel_l1_thresh,
        coefficients=(0.0, 0.0, 0.0, 0.0, 1.0),
        skip_first_n_steps=1,
        skip_last_n_steps=1,
    )
    handle._gen_ctx.token = 1
    handle._gen_ctx.active_num_steps = 4
    return handle


def _assert_bit_equal(actual: mx.array, expected: mx.array) -> None:
    # Loaded models run in bfloat16; a non-finite reference would compare equal to an equally broken copy.
    assert expected.dtype == mx.bfloat16
    assert mx.all(mx.isfinite(expected)).item()
    assert actual.dtype == expected.dtype
    diff = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
    assert mx.array_equal(actual, expected).item(), f"max abs difference {diff}"


# fast-path: threshold 0, one step that always computes. gate-path: threshold 0.3 over two steps, a forced compute
# (step 0) and then a compute the gate decides (step 1), which also writes the cache.
@pytest.mark.parametrize("precision", _PRECISIONS)
@pytest.mark.parametrize(
    ("rel_l1_thresh", "expected_decisions"),
    [(0.0, ["computed"]), (0.3, ["forced", "computed"])],
    ids=["fast-path", "gate-path-forced-then-gated-compute"],
)
def test_computed_steps_equal_the_plain_transformer_for_both_branches(
    rel_l1_thresh: float, expected_decisions: list[str], precision: str
) -> None:
    """Bug: the gated forward's prelude, block walk or tail drifts from mflux's QwenTransformer.__call__ (a new
    argument the copy does not pass, a tail that bypasses norm_out or reads a linear's weight directly), so a
    computed TeaCache step is not the model mflux runs."""
    transformer = _tiny_transformer(precision)
    config = _config()
    latents, prompt_pos, prompt_neg, mask = _inputs()
    handle = _handle(rel_l1_thresh)
    for t in range(len(expected_decisions)):
        # mflux calls the transformer for the positive, then the negative prompt
        for prompt in (prompt_pos, prompt_neg):
            expected = transformer(
                t=t,
                config=config,
                hidden_states=latents,
                encoder_hidden_states=prompt,
                encoder_hidden_states_mask=mask,
            )
            actual = qwen_forward_with_gate(
                transformer,
                handle,
                t=t,
                config=config,
                hidden_states=latents,
                encoder_hidden_states=prompt,
                encoder_hidden_states_mask=mask,
            )
            _assert_bit_equal(actual, expected)
    # One decision per step pair, each a compute: the comparisons above are not against a replayed residual.
    stats = handle._state.stats
    stats.finalize_last_generation(num_inference_steps=len(expected_decisions), cfg_was_active=True)
    assert stats.last_generation is not None
    assert [d.decision for d in stats.last_generation.decisions] == expected_decisions


@pytest.mark.parametrize("precision", _PRECISIONS)
def test_gate_signal_is_block_zeros_modulated_image_input(
    monkeypatch: pytest.MonkeyPatch, precision: str
) -> None:
    """Bug: the gate signal is not what mflux's block 0 feeds its attention (img_norm1 dropped, the second
    modulation chunk taken instead of the first, or the raw img_in output), so the calibrated polynomial reads a
    different quantity from the one it was fitted on."""
    transformer = _tiny_transformer(precision)
    config = _config()
    latents, prompt_pos, _prompt_neg, mask = _inputs()
    modulated: list[mx.array] = []
    original = QwenTransformerBlock._modulate

    def _spy(*args: object, **kwargs: object) -> tuple[mx.array, mx.array]:
        out: tuple[mx.array, mx.array] = original(*args, **kwargs)
        modulated.append(out[0])
        return out

    monkeypatch.setattr(QwenTransformerBlock, "_modulate", staticmethod(_spy))
    transformer(
        t=0,
        config=config,
        hidden_states=latents,
        encoder_hidden_states=prompt_pos,
        encoder_hidden_states_mask=mask,
    )
    monkeypatch.undo()
    # mflux's first _modulate call is block 0's image stream, before attention.
    signal = _qwen_signal_a(transformer, _qwen_prelude(transformer, 0, config, latents))
    _assert_bit_equal(signal, modulated[0])


def _qwen_flux(transformer: object) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(aliases=["qwen-image", "qwen"], model_name=None), transformer=transformer
    )


def test_detection_refuses_mfluxs_zero_cond_t_transformer() -> None:
    """Bug: mflux renames or moves the zero_cond_t attribute, so detection stops seeing it and TeaCache patches a
    Qwen-Image-Edit-2511 transformer whose reference-image path its forward does not implement."""
    if "zero_cond_t" not in inspect.signature(QwenTransformer.__init__).parameters:
        pytest.skip("this mflux has no zero_cond_t transformer option (added in 0.22)")
    assert matches(_qwen_flux(_tiny_transformer(zero_cond_t=True))) is False


def test_detection_accepts_mfluxs_default_transformer() -> None:
    """Bug: detection refuses on the attribute's presence rather than its value, so every text-to-image Qwen-Image
    transformer (mflux 0.22 always sets zero_cond_t, to False) is refused."""
    assert matches(_qwen_flux(_tiny_transformer())) is True
