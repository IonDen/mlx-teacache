# tests/test_forward_last_timestep.py
"""The FLUX.2 and Z-Image forwards keep ``TeaCacheState.last_timestep`` current.

``last_timestep`` is a public field of ``TeaCacheState`` (exported through
``mlx_teacache.cache``); callers read it from ``handle`` state after a step to
see which timestep the forward last saw. Every forward path writes it: the
threshold-zero fast path, a computed gated step and a skipped gated step, with
and without CFG.

The FLUX.2 cases use the synthetic inners from the FLUX.2 forward tests. The
Z-Image cases replace the module's prelude, timestep-embedding and tail helpers
(which need a real ZImageTransformer) with tiny stand-ins, so the real gated
forward and the real gate run on small tensors.
"""

from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest

from mlx_teacache._kernel.cache import TeaCacheState
from mlx_teacache._kernel.stats import TeaCacheStats
from mlx_teacache.variants.flux2_klein_base_4b.integration import (
    flux2_cfg_forward_with_gate,
    flux2_forward_with_gate,
)
from tests.test_cfg_branch_independence import _FakeCFGFlux2Inner, _make_inputs
from tests.test_forward_flux2 import _FakeFlux2Inner

# Calling the forwards lazily imports mflux's ModelConfig.
pytestmark = pytest.mark.mflux

_ZERO_COEFFS = (0.0, 0.0, 0.0, 0.0, 0.0)  # p(x) = 0: every step after the seed skips


def _handle(*, rel_l1_thresh: float, steps: int = 4) -> Any:
    return SimpleNamespace(
        rel_l1_thresh=rel_l1_thresh,
        coefficients=_ZERO_COEFFS,
        calibrated_range=None,
        skip_first_n_steps=0,
        skip_last_n_steps=0,
        _state=SimpleNamespace(cache=TeaCacheState(), stats=TeaCacheStats()),
        _gen_ctx=SimpleNamespace(active_num_steps=steps),
    )


def _flux2_step(handle: Any, timestep: float) -> None:
    flux2_forward_with_gate(
        _FakeFlux2Inner(),
        handle,
        hidden_states=mx.zeros((1, 4, 4)),
        encoder_hidden_states=mx.zeros((1, 2, 4)),
        timestep=mx.array([timestep]),
        img_ids=mx.zeros((4, 3)),
        txt_ids=mx.zeros((2, 3)),
    )


def _flux2_cfg_step(handle: Any, timestep: float) -> None:
    inputs = _make_inputs()
    inputs["timestep"] = mx.array([timestep])
    flux2_cfg_forward_with_gate(_FakeCFGFlux2Inner(), handle, **inputs)


def _kinds(handle: Any) -> list[str]:
    return [d.decision for d in handle._state.stats._staging.decisions]


def test_flux2_fast_path_records_last_timestep() -> None:
    """Bug caught: the threshold-zero FLUX.2 path stops writing last_timestep."""
    handle = _handle(rel_l1_thresh=0.0)
    _flux2_step(handle, 700.0)
    assert handle._state.cache.last_timestep == 700.0


def test_flux2_gated_path_records_last_timestep_on_compute_and_skip() -> None:
    """Bug caught: the gated FLUX.2 path writes last_timestep only on computed
    steps (or not at all), so after a skipped step it names the previous one."""
    handle = _handle(rel_l1_thresh=1.0)
    _flux2_step(handle, 900.0)
    assert handle._state.cache.last_timestep == 900.0
    _flux2_step(handle, 800.0)
    assert _kinds(handle) == ["computed", "skipped"]
    assert handle._state.cache.last_timestep == 800.0


def test_flux2_cfg_records_last_timestep_on_fast_compute_and_skip() -> None:
    """Bug caught: a CFG FLUX.2 path (fast, computed or skipped) stops writing
    last_timestep."""
    fast = _handle(rel_l1_thresh=0.0)
    _flux2_cfg_step(fast, 600.0)
    assert fast._state.cache.last_timestep == 600.0

    gated = _handle(rel_l1_thresh=1.0)
    _flux2_cfg_step(gated, 500.0)
    assert gated._state.cache.last_timestep == 500.0
    _flux2_cfg_step(gated, 400.0)
    assert _kinds(gated) == ["computed", "skipped"]
    assert gated._state.cache.last_timestep == 400.0


# ---------- Z-Image ----------


class _ZLayer:
    def __call__(self, *, x: mx.array, **_kw: Any) -> mx.array:
        return x + 1.0


@pytest.fixture
def zimage(monkeypatch: pytest.MonkeyPatch) -> Any:
    from mlx_teacache.variants.z_image_base import integration as zi

    def prelude(transformer: Any, latents: mx.array, t_emb: mx.array, cap_feats: mx.array) -> Any:
        return zi._Prelude(
            unified_in=mx.zeros((1, 3, 4)),
            freqs_cis=mx.zeros((3, 2)),
            attn_mask=mx.ones((1, 3), dtype=mx.bool_),
            x_len=2,
            x_size=None,
        )

    monkeypatch.setattr(zi, "_zimage_t_emb", lambda transformer, timestep, sigmas: mx.zeros((1, 4)))
    monkeypatch.setattr(zi, "_zimage_prelude", prelude)
    monkeypatch.setattr(
        zi, "_zimage_tail", lambda transformer, main_out, t_emb, pre: main_out[:, : pre.x_len]
    )
    return zi


def _z_transformer() -> Any:
    return SimpleNamespace(layers=[_ZLayer(), _ZLayer()])


def _z_step(zi: Any, handle: Any, timestep: float) -> None:
    zi.zimage_forward_with_gate(
        _z_transformer(),
        handle,
        latents=mx.zeros((1, 4)),
        timestep=mx.array([timestep]),
        sigmas=mx.zeros((4,)),
        cap_feats=mx.zeros((1, 4)),
    )


def _z_cfg_step(zi: Any, handle: Any, timestep: float) -> None:
    zi.zimage_cfg_forward_with_gate(
        _z_transformer(),
        handle,
        latents=mx.zeros((1, 4)),
        timestep=mx.array([timestep]),
        sigmas=mx.zeros((4,)),
        cap_feats_pos=mx.zeros((1, 4)),
        cap_feats_neg=mx.zeros((1, 4)),
        guidance=4.0,
    )


def test_zimage_records_last_timestep_on_fast_compute_and_skip(zimage: Any) -> None:
    """Bug caught: a non-CFG Z-Image path (fast, computed or skipped) stops
    writing last_timestep."""
    fast = _handle(rel_l1_thresh=0.0)
    _z_step(zimage, fast, 0.75)
    assert fast._state.cache.last_timestep == 0.75

    gated = _handle(rel_l1_thresh=1.0)
    _z_step(zimage, gated, 0.5)
    assert gated._state.cache.last_timestep == 0.5
    _z_step(zimage, gated, 0.25)
    assert _kinds(gated) == ["computed", "skipped"]
    assert gated._state.cache.last_timestep == 0.25


def test_zimage_cfg_records_last_timestep_on_fast_compute_and_skip(zimage: Any) -> None:
    """Bug caught: a CFG Z-Image path (fast, computed or skipped) stops writing
    last_timestep."""
    fast = _handle(rel_l1_thresh=0.0)
    _z_cfg_step(zimage, fast, 0.75)
    assert fast._state.cache.last_timestep == 0.75

    gated = _handle(rel_l1_thresh=1.0)
    _z_cfg_step(zimage, gated, 0.5)
    assert gated._state.cache.last_timestep == 0.5
    _z_cfg_step(zimage, gated, 0.25)
    assert _kinds(gated) == ["computed", "skipped"]
    assert gated._state.cache.last_timestep == 0.25
