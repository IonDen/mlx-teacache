"""Z-Image gated forwards against a synthetic ZImageTransformer.

The fake keeps the whole prelude an identity (patchify returns the latents and
caption features as-is, embedders and pad masks are no-ops, no refiner layers),
so `unified_in` is exactly `concat([latents, cap_feats])`. Main layer i adds a
constant c_i = 1, 10, 100, so layer 0's output h1 differs from unified_in by 1
and the full body residual `main_out - unified_in` is 111 everywhere. The tail
is identity plus the vanilla negation, so a branch's noise is
`-main_out[:x_len]`.

These pin the residual base of both forwards: the cache must store
`main_out - unified_in`, and a skip must rebuild `unified_in + cached`, never
from h1. Real-weights numerical parity lives in tests/test_parity_z_image.py.
"""

from typing import Any

import mlx.core as mx

from mlx_teacache.variants.z_image_base.integration import (
    _InternalHandle,
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
        self.layers = [_AddLayer(1.0), _AddLayer(10.0), _AddLayer(100.0)]

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


def _handle() -> _InternalHandle:
    handle = _InternalHandle(
        rel_l1_thresh=0.5,
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
    """Bug caught: z_image_base/integration.py:292 `pre.unified_in + cached` -> `h1 + cached`,
    or :286 `main_out - pre.unified_in` -> `main_out - h1`, stays green today.

    Seed at latents 1.0 caches residual 111; the forced skip at latents 2.0 must give
    noise -(2 + 111) = -113 (h1 as the skip base gives -114, h1 as the residual base
    gives a residual of 110 and -112)."""
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
        handle._state.cache.cached_residual, _full((1, _X_SEQ + _CAP_SEQ, _DIM), 111.0)
    ).item()
    skip_out = step(2.0)

    assert _decisions(handle) == ["computed", "skipped"]
    assert mx.array_equal(seed_out, _full((_X_SEQ, _DIM), -112.0)).item()
    assert mx.array_equal(skip_out, _full((_X_SEQ, _DIM), -113.0)).item()


def test_cfg_forward_caches_both_body_residuals_and_rebuilds_from_unified_in():
    """Bug caught: z_image_base/integration.py:382 `pre_pos.unified_in + cached` -> `h1_pos + cached`,
    or :373/:374 `main_out_{pos,neg} - pre_{pos,neg}.unified_in` -> `- h1_{pos,neg}`, stays green today.

    Both branches share the latents, so with correct residuals (111 each) the two
    noises match and the CFG combine pos + 4 * (pos - neg) reduces to the positive
    noise: -(2 + 111) = -113 on the skip step. Any residual or skip-base swap on one
    branch shifts that branch by 1 and the combine by 4 or 5."""
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
    expected_residual = _full((1, _X_SEQ + _CAP_SEQ, _DIM), 111.0)
    assert mx.array_equal(handle._state.cache.cached_residual, expected_residual).item()
    assert mx.array_equal(handle._state.cache.cached_residual_neg, expected_residual).item()
    skip_out = step(2.0)

    assert _decisions(handle) == ["computed", "skipped"]
    assert mx.array_equal(seed_out, _full((_X_SEQ, _DIM), -112.0)).item()
    assert mx.array_equal(skip_out, _full((_X_SEQ, _DIM), -113.0)).item()
