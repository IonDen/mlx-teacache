"""Weight-free parity of the gated Z-Image forwards against a tiny REAL mflux ZImageTransformer.

mflux builds after 0.21.0 run the hidden stream in bfloat16 (mflux PR 803) and cast t_emb through
`ZImageTransformer.stream_t_emb`. At threshold 0 the gated forward runs every layer, so it must equal the plain
transformer bit for bit in both stream modes, with and without CFG. A released mflux without `stream_t_emb`
skips these tests; they run on mflux main and on the first release that carries the change."""

import mlx.core as mx
import pytest

from mlx_teacache.variants.z_image_base.integration import (
    zimage_cfg_forward_with_gate,
    zimage_forward_with_gate,
)
from tests.test_forward_z_image_fake import _handle

pytest.importorskip("mflux")
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(ZImageTransformer, "stream_t_emb"),
    reason="this mflux has no bfloat16 Z-Image stream (stream_t_emb)",
)

_DIM = 64
_SIGMAS = mx.array([0.5])


def _tiny_transformer(*, float32: bool) -> ZImageTransformer:
    mx.random.seed(0)
    transformer = ZImageTransformer(
        dim=_DIM,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=2,
        cap_feat_dim=32,
        axes_dims=[8, 12, 12],
        axes_lens=[64, 32, 32],
    )
    # Loaded weights are bfloat16; the RoPE tables are not parameters and stay float32, as in a real load.
    transformer.set_dtype(mx.bfloat16)
    transformer.set_float32(float32)
    return transformer


def _inputs() -> tuple[mx.array, mx.array, mx.array]:
    mx.random.seed(1)
    latents = mx.random.normal((16, 1, 8, 8)).astype(mx.bfloat16)
    cap_pos = mx.random.normal((7, 32)).astype(mx.bfloat16)
    cap_neg = mx.random.normal((7, 32)).astype(mx.bfloat16)
    return latents, cap_pos, cap_neg


@pytest.mark.parametrize("float32", [False, True], ids=["bf16-stream", "float32-stream"])
def test_gated_forward_at_threshold_zero_equals_the_plain_transformer(float32: bool) -> None:
    """Bug: the gated forward differs from mflux in dtype or arithmetic on the hidden stream (a float32 t_emb on the
    bfloat16 stream, a cast applied in float32 mode), so a TeaCache run is not the model mflux runs."""
    transformer = _tiny_transformer(float32=float32)
    latents, cap_pos, _ = _inputs()
    timestep = mx.array([0.5])
    expected = transformer(x=latents, timestep=timestep, sigmas=_SIGMAS, cap_feats=cap_pos)
    actual = zimage_forward_with_gate(
        transformer,
        _handle(rel_l1_thresh=0.0),
        latents=latents,
        timestep=timestep,
        sigmas=_SIGMAS,
        cap_feats=cap_pos,
    )
    assert actual.dtype == expected.dtype
    diff = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
    assert mx.array_equal(actual, expected).item(), f"max abs difference {diff}"


@pytest.mark.parametrize("float32", [False, True], ids=["bf16-stream", "float32-stream"])
def test_cfg_gated_forward_at_threshold_zero_equals_the_plain_cfg_combine(float32: bool) -> None:
    """Bug: the CFG gated forward differs from mflux's two transformer calls plus `pos + g * (pos - neg)` in dtype
    or arithmetic, for either branch."""
    transformer = _tiny_transformer(float32=float32)
    latents, cap_pos, cap_neg = _inputs()
    timestep = mx.array([0.5])
    guidance = 4.0
    pos = transformer(x=latents, timestep=timestep, sigmas=_SIGMAS, cap_feats=cap_pos)
    neg = transformer(x=latents, timestep=timestep, sigmas=_SIGMAS, cap_feats=cap_neg)
    expected = pos + guidance * (pos - neg)
    actual = zimage_cfg_forward_with_gate(
        transformer,
        _handle(rel_l1_thresh=0.0),
        latents=latents,
        timestep=timestep,
        sigmas=_SIGMAS,
        cap_feats_pos=cap_pos,
        cap_feats_neg=cap_neg,
        guidance=guidance,
    )
    assert actual.dtype == expected.dtype
    diff = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
    assert mx.array_equal(actual, expected).item(), f"max abs difference {diff}"
