"""Z-Image gated forwards against a synthetic ZImageTransformer.

The fake keeps the whole prelude an identity (patchify returns the latents and
caption features as-is, embedders and pad masks are no-ops, no refiner layers),
so `unified_in` is exactly `concat([latents, cap_feats])` along the sequence.
Main layers 0 and 2 add the constants 1 and 100. Layer 1 adds 10 plus the mean of
the caption slice of its input, so the body residual depends on the caption:
with caption features +7 layer 1 sees a caption mean of 8 and the residual
`main_out - unified_in` is 1 + 18 + 100 = 119 everywhere; with -7 it sees -6 and
the residual is 1 + 4 + 100 = 105. Layer 0's output h1 differs from unified_in by
1. The tail is identity plus the vanilla negation, so a branch's noise is
`-main_out[:x_len]`.

These pin the residual base of both forwards (the cache stores
`main_out - unified_in`, a skip rebuilds `unified_in + cached`, never from h1),
and, because the two CFG branches now differ, the CFG forward's branch wiring:
the negative branch must skip on its own cached residual and the combine must
start from the positive noise. Real-weights numerical parity lives in
tests/test_parity_z_image.py.
"""

from typing import Any

import mlx.core as mx

from mlx_teacache.variants.z_image_base.integration import (
    _InternalHandle,
    _zimage_t_emb,
    zimage_cfg_forward_with_gate,
    zimage_forward_with_gate,
)

_X_SEQ = 3
_CAP_SEQ = 2
_DIM = 4
_KEY = "2-1"


def _identity(x: mx.array, *_: Any) -> mx.array:
    return x


class _AddLayer:
    def __init__(self, c: float) -> None:
        self.c = c

    def __call__(self, *, x: mx.array, attn_mask: Any, freqs_cis: Any, t_emb: Any) -> mx.array:
        return x + self.c


class _AddCaptionMeanLayer(_AddLayer):
    """Adds c plus the mean of the caption rows of the stream, so a branch's residual
    depends on its caption and the positive and negative branches differ."""

    def __call__(self, *, x: mx.array, attn_mask: Any, freqs_cis: Any, t_emb: Any) -> mx.array:
        return x + self.c + mx.mean(x[:, _X_SEQ:, :])


class _FakeZImageTransformer:
    patch_size = 2
    f_patch_size = 1
    t_scale = 1000.0
    out_channels = _DIM

    def __init__(self) -> None:
        self.all_x_embedder = {_KEY: _identity}
        self.all_final_layer = {_KEY: _identity}
        self.x_pad_token = mx.zeros((_DIM,))
        self.cap_pad_token = mx.zeros((_DIM,))
        self.cap_embedder = [_identity, _identity]
        self.noise_refiner: list[Any] = []
        self.context_refiner: list[Any] = []
        self.layers = [_AddLayer(1.0), _AddCaptionMeanLayer(10.0), _AddLayer(100.0)]

    def t_embedder(self, t: mx.array) -> mx.array:
        return t

    def rope_embedder(self, pos_ids: mx.array) -> mx.array:
        return mx.zeros((int(pos_ids.shape[0]), 2))

    @staticmethod
    def _patchify(
        *, image: mx.array, cap_feats: mx.array, patch_size: int, f_patch_size: int
    ) -> tuple[Any, ...]:
        return (
            image,
            cap_feats,
            None,  # x_size, only forwarded to _unpatchify
            mx.zeros((_X_SEQ,)),
            mx.zeros((_CAP_SEQ,)),
            mx.zeros((_X_SEQ,), dtype=mx.bool_),
            mx.zeros((_CAP_SEQ,), dtype=mx.bool_),
        )

    @staticmethod
    def _unpatchify(
        *, x: mx.array, size: Any, patch_size: int, f_patch_size: int, out_channels: int
    ) -> mx.array:
        return x


def _handle(rel_l1_thresh: float = 0.5) -> _InternalHandle:
    handle = _InternalHandle(
        rel_l1_thresh=rel_l1_thresh,
        coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),  # poly→0 ⇒ every gated step after the seed skips
        skip_first_n_steps=0,
        skip_last_n_steps=1,
    )
    handle._gen_ctx.active_num_steps = 4
    return handle


def _full(shape: tuple[int, ...], value: float) -> mx.array:
    return mx.full(shape, value)


def _decisions(handle: _InternalHandle) -> list[str]:
    return [d.decision for d in handle._state.stats._staging.decisions]


def test_non_cfg_forward_caches_body_residual_and_rebuilds_from_unified_in():
    """Bug caught: zimage_forward_with_gate rebuilding a skip as `h1 + cached` instead of
    `pre.unified_in + cached`, or caching `main_out - h1` instead of `main_out - pre.unified_in`.

    Seed at latents 1.0 caches residual 119; the forced skip at latents 2.0 must give
    noise -(2 + 119) = -121 (h1 as the skip base gives -122, h1 as the residual base
    gives a residual of 118 and -120)."""
    transformer = _FakeZImageTransformer()
    handle = _handle()

    def step(latent_value: float) -> mx.array:
        return zimage_forward_with_gate(
            transformer,
            handle,
            latents=_full((_X_SEQ, _DIM), latent_value),
            timestep=mx.array([0.5]),
            sigmas=mx.array([1.0, 0.5, 0.0]),
            cap_feats=_full((_CAP_SEQ, _DIM), 7.0),
        )

    seed_out = step(1.0)
    assert mx.array_equal(
        handle._state.cache.cached_residual, _full((1, _X_SEQ + _CAP_SEQ, _DIM), 119.0)
    ).item()
    skip_out = step(2.0)

    assert _decisions(handle) == ["computed", "skipped"]
    assert mx.array_equal(seed_out, _full((_X_SEQ, _DIM), -120.0)).item()
    assert mx.array_equal(skip_out, _full((_X_SEQ, _DIM), -121.0)).item()


def test_cfg_forward_caches_both_body_residuals_and_rebuilds_from_unified_in():
    """Bug caught, in zimage_cfg_forward_with_gate: the negative branch skipping on the
    positive `cached_residual` instead of `cached_residual_neg`; the combine starting from
    `noise_neg` instead of `noise_pos`; a skip rebuilt as `h1_pos + cached`; a residual
    cached as `main_out_{pos,neg} - h1_{pos,neg}`.

    Residuals are 119 (pos, caption +7) and 105 (neg, caption -7). Seed at latents 1.0:
    noises -120 / -106, combine -120 + 4 * (-120 + 106) = -176. Skip at latents 2.0:
    -121 / -107, combine -177. The neg skip on the pos residual gives -121 (the
    branches cancel); a combine from noise_neg gives -162 on the seed; h1_pos as the
    skip base gives -182; an h1-based pos or neg residual gives -172 or -181."""
    transformer = _FakeZImageTransformer()
    handle = _handle()

    def step(latent_value: float) -> mx.array:
        return zimage_cfg_forward_with_gate(
            transformer,
            handle,
            latents=_full((_X_SEQ, _DIM), latent_value),
            timestep=mx.array([0.5]),
            sigmas=mx.array([1.0, 0.5, 0.0]),
            cap_feats_pos=_full((_CAP_SEQ, _DIM), 7.0),
            cap_feats_neg=_full((_CAP_SEQ, _DIM), -7.0),
            guidance=4.0,
        )

    seed_out = step(1.0)
    residual_shape = (1, _X_SEQ + _CAP_SEQ, _DIM)
    assert mx.array_equal(handle._state.cache.cached_residual, _full(residual_shape, 119.0)).item()
    assert mx.array_equal(handle._state.cache.cached_residual_neg, _full(residual_shape, 105.0)).item()
    skip_out = step(2.0)

    assert _decisions(handle) == ["computed", "skipped"]
    assert mx.array_equal(seed_out, _full((_X_SEQ, _DIM), -176.0)).item()
    assert mx.array_equal(skip_out, _full((_X_SEQ, _DIM), -177.0)).item()


def test_cfg_forward_with_gating_off_combines_from_the_positive_noise():
    """Bug caught: the threshold-0 path of zimage_cfg_forward_with_gate combining as
    `noise_neg + g * (pos - neg)` instead of `noise_pos + g * (pos - neg)`.

    Both bodies run in full: noises -120 (pos) and -106 (neg), combine
    -120 + 4 * (-14) = -176; starting from noise_neg gives -162."""
    transformer = _FakeZImageTransformer()
    handle = _handle(rel_l1_thresh=0.0)
    out = zimage_cfg_forward_with_gate(
        transformer,
        handle,
        latents=_full((_X_SEQ, _DIM), 1.0),
        timestep=mx.array([0.5]),
        sigmas=mx.array([1.0, 0.5, 0.0]),
        cap_feats_pos=_full((_CAP_SEQ, _DIM), 7.0),
        cap_feats_neg=_full((_CAP_SEQ, _DIM), -7.0),
        guidance=4.0,
    )
    assert _decisions(handle) == ["computed"]
    assert mx.array_equal(out, _full((_X_SEQ, _DIM), -176.0)).item()


class _Bf16StreamFake(_FakeZImageTransformer):
    """Mirrors mflux's bfloat16 stream: a `stream_t_emb` staticmethod plus a `float32` flag."""

    def __init__(self, *, float32: bool) -> None:
        super().__init__()
        self.float32 = float32

    @staticmethod
    def stream_t_emb(t_emb: mx.array, float32: bool) -> mx.array:
        return t_emb if float32 else t_emb.astype(mx.bfloat16)


_SIGMAS = mx.array([1.0, 0.5, 0.0])


def test_t_emb_is_bfloat16_when_the_transformer_has_a_bf16_stream():
    """Bug: _zimage_t_emb skips mflux's stream_t_emb, so a TeaCache run keeps the float32 stream
    while plain mflux runs bfloat16."""
    t_emb = _zimage_t_emb(_Bf16StreamFake(float32=False), mx.array([0.5]), _SIGMAS)
    assert t_emb.dtype == mx.bfloat16


def test_t_emb_stays_float32_when_the_transformer_asks_for_float32():
    """Bug: the float32 flag is ignored (the cast is applied unconditionally), so mflux's float32 mode
    silently turns bfloat16 under TeaCache."""
    t_emb = _zimage_t_emb(_Bf16StreamFake(float32=True), mx.array([0.5]), _SIGMAS)
    assert t_emb.dtype == mx.float32


def test_t_emb_is_unchanged_on_a_transformer_without_stream_t_emb():
    """Bug: the cast is applied without feature detection, changing the dtype on mflux 0.17.5 to 0.21.0."""
    t_emb = _zimage_t_emb(_FakeZImageTransformer(), mx.array([0.5]), _SIGMAS)
    assert t_emb.dtype == mx.float32


def test_gated_forward_hands_the_layers_a_bfloat16_t_emb_on_a_bf16_stream():
    """Bug: the cast is made in _zimage_t_emb's caller path only for one forward, or not at all,
    so the main layers receive a float32 t_emb that plain mflux would give them as bfloat16."""
    seen: list[Any] = []

    class _Spy(_AddLayer):
        def __call__(self, *, x: mx.array, attn_mask: Any, freqs_cis: Any, t_emb: Any) -> mx.array:
            seen.append(t_emb.dtype)
            return super().__call__(x=x, attn_mask=attn_mask, freqs_cis=freqs_cis, t_emb=t_emb)

    transformer = _Bf16StreamFake(float32=False)
    transformer.layers = [_Spy(1.0)]
    zimage_forward_with_gate(
        transformer,
        _handle(),
        latents=_full((_X_SEQ, _DIM), 1.0),
        timestep=mx.array([0.5]),
        sigmas=_SIGMAS,
        cap_feats=_full((_CAP_SEQ, _DIM), 7.0),
    )
    assert seen == [mx.bfloat16]


def test_cfg_gated_forward_hands_the_layers_a_bfloat16_t_emb_on_a_bf16_stream():
    """Bug: zimage_cfg_forward_with_gate builds t_emb without _zimage_t_emb (or casts only for one branch), so
    CFG runs hand the main layers a float32 t_emb that plain mflux would give them as bfloat16."""
    seen: list[Any] = []

    class _Spy(_AddLayer):
        def __call__(self, *, x: mx.array, attn_mask: Any, freqs_cis: Any, t_emb: Any) -> mx.array:
            seen.append(t_emb.dtype)
            return super().__call__(x=x, attn_mask=attn_mask, freqs_cis=freqs_cis, t_emb=t_emb)

    transformer = _Bf16StreamFake(float32=False)
    transformer.layers = [_Spy(1.0)]
    zimage_cfg_forward_with_gate(
        transformer,
        _handle(),
        latents=_full((_X_SEQ, _DIM), 1.0),
        timestep=mx.array([0.5]),
        sigmas=_SIGMAS,
        cap_feats_pos=_full((_CAP_SEQ, _DIM), 7.0),
        cap_feats_neg=_full((_CAP_SEQ, _DIM), -7.0),
        guidance=4.0,
    )
    assert seen == [mx.bfloat16, mx.bfloat16]
