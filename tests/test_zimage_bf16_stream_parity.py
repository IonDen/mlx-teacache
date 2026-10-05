"""Weight-free parity of the gated Z-Image forwards against a tiny REAL mflux ZImageTransformer.

mflux builds after 0.21.0 run the hidden stream in bfloat16 (mflux PR 803) and cast t_emb through
`ZImageTransformer.stream_t_emb`. At threshold 0 the gated forward runs every layer, so it must equal the plain
transformer bit for bit in both stream modes, with and without CFG. A released mflux without `stream_t_emb`
skips these tests; they run on mflux main and on the first release that carries the change.

mflux PR 820 adds an opt-in float16 compute for the attention and feed-forward layers (`compute_precision`). The
setting lives on those modules and the weights are cast in place, so the gated forward reaches it only by calling the
modules. Each case also runs with that option, applied through mflux's own `apply_compute_precision`, and skips on an
mflux without it."""

import mlx.core as mx
import pytest

from mlx_teacache.variants.z_image_base.integration import (
    zimage_cfg_forward_with_gate,
    zimage_forward_with_gate,
)
from tests._mflux_surface import installed_vcs_commit, require_or_skip
from tests.test_forward_z_image_fake import _handle

pytest.importorskip("mflux")
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer  # noqa: E402


@pytest.fixture(autouse=True)
def _require_bf16_stream() -> None:
    require_or_skip(
        hasattr(ZImageTransformer, "stream_t_emb"),
        vcs_commit=installed_vcs_commit("mflux"),
        feature="bfloat16 Z-Image stream (stream_t_emb)",
    )


_HAS_FLOAT16_COMPUTE = hasattr(ZImageTransformer, "apply_compute_precision")
_COMPUTE = [
    pytest.param(False, id="model-precision"),
    pytest.param(True, id="float16-compute"),
]

_DIM = 64
_SIGMAS = mx.array([0.5])


def _tiny_transformer(*, float32: bool, float16_compute: bool = False) -> ZImageTransformer:
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


def _inputs() -> tuple[mx.array, mx.array, mx.array]:
    mx.random.seed(1)
    latents = mx.random.normal((16, 1, 8, 8)).astype(mx.bfloat16)
    cap_pos = mx.random.normal((7, 32)).astype(mx.bfloat16)
    cap_neg = mx.random.normal((7, 32)).astype(mx.bfloat16)
    return latents, cap_pos, cap_neg


@pytest.mark.parametrize("float16_compute", _COMPUTE)
@pytest.mark.parametrize("float32", [False, True], ids=["bf16-stream", "float32-stream"])
def test_gated_forward_at_threshold_zero_equals_the_plain_transformer(
    float32: bool, float16_compute: bool
) -> None:
    """Bug: the gated forward differs from mflux in dtype or arithmetic on the hidden stream (a float32 t_emb on the
    bfloat16 stream, a cast applied in float32 mode, a bypass of the modules' float16 compute), so a TeaCache run is
    not the model mflux runs."""
    transformer = _tiny_transformer(float32=float32, float16_compute=float16_compute)
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


@pytest.mark.parametrize("float16_compute", _COMPUTE)
@pytest.mark.parametrize("float32", [False, True], ids=["bf16-stream", "float32-stream"])
def test_cfg_gated_forward_at_threshold_zero_equals_the_plain_cfg_combine(
    float32: bool, float16_compute: bool
) -> None:
    """Bug: the CFG gated forward differs from mflux's two transformer calls plus `pos + g * (pos - neg)` in dtype
    or arithmetic, for either branch, with the modules' float16 compute on or off."""
    transformer = _tiny_transformer(float32=float32, float16_compute=float16_compute)
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


def test_float16_compute_changes_the_plain_output_so_the_parity_cases_are_not_vacuous() -> None:
    """Bug: `apply_compute_precision` leaves the tiny transformer unchanged, so the float16-compute parity cases
    above compare two identical model-precision runs and would stay green if the gated forward bypassed the option."""
    latents, cap_pos, _ = _inputs()
    timestep = mx.array([0.5])
    plain = _tiny_transformer(float32=False)(x=latents, timestep=timestep, sigmas=_SIGMAS, cap_feats=cap_pos)
    half = _tiny_transformer(float32=False, float16_compute=True)(
        x=latents, timestep=timestep, sigmas=_SIGMAS, cap_feats=cap_pos
    )
    assert not mx.array_equal(plain, half).item()
