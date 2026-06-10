# tests/test_flux2_predict.py
"""Smoke tests for make_teacache_predict_factory. Uses fake transformers to
avoid loading real FLUX.2 weights. Deep parity tests use a real Flux2Klein
4b in tests/test_parity_flux2.py (Task 26)."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_teacache.cache import TeaCacheState
from mlx_teacache.errors import (
    InvalidStepWindowError,
    MissingGenerationContextError,
)
from mlx_teacache.integrations.mflux.lifecycle import GenerationContext
from mlx_teacache.stats import TeaCacheStats
from mlx_teacache.variants.flux2_klein_base_4b.integration import make_teacache_predict_factory

# NOTE: this file runs in the pure-core (no-mflux) CI lane. The current tests
# all raise before reaching flux2_forward_with_gate, which does a function-local
# `from mflux...` import. A future test that drives predict() past the
# validation gates would ImportError in that lane — mark it mflux if so.


@dataclass
class _FakeHandle:
    variant_id: str = "flux2-klein-4b"
    rel_l1_thresh: float = 0.0  # always-compute, threshold-zero short-circuit
    coefficients: tuple = (0.0, 0.0, 0.0, 0.0, 0.0)
    skip_first_n_steps: int = 1
    skip_last_n_steps: int = 1
    _gen_ctx: GenerationContext = field(default_factory=GenerationContext)
    _state: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(
            stats=TeaCacheStats(),
            cache=TeaCacheState(),
        )
    )


def _set_ctx_ready(handle, num_steps):
    handle._gen_ctx.token += 1
    handle._gen_ctx.active_num_steps = num_steps
    handle._gen_ctx.consumed_at_token = None


def test_factory_returns_callable_per_generation():
    handle = _FakeHandle()
    factory = make_teacache_predict_factory(handle)
    transformer = SimpleNamespace(__call__=lambda *a, **kw: mx.ones((1, 2, 4)))
    p1 = factory(transformer)
    p2 = factory(transformer)
    assert p1 is not p2  # fresh closure per generation


def test_missing_context_before_first_call_raises():
    handle = _FakeHandle()
    factory = make_teacache_predict_factory(handle)

    def transformer(**kw):
        return mx.ones((1, 2, 4))

    predict = factory(transformer)
    with pytest.raises(MissingGenerationContextError):
        predict(
            latents=mx.ones((1, 2, 4)),
            latent_ids=mx.zeros((1, 2, 3)),
            prompt_embeds=mx.ones((1, 8, 16)),
            text_ids=mx.zeros((1, 8, 3)),
            negative_prompt_embeds=None,
            negative_text_ids=None,
            guidance=1.0,
            timestep=mx.array(1.0),
        )


def test_invalid_step_window_raised_on_first_non_cfg_step():
    handle = _FakeHandle(skip_first_n_steps=1, skip_last_n_steps=1)
    _set_ctx_ready(handle, num_steps=2)  # 1+1 >= 2 ⇒ invalid
    factory = make_teacache_predict_factory(handle)

    def transformer(**kw):
        return mx.ones((1, 2, 4))

    predict = factory(transformer)
    # Non-CFG call ⇒ validation fires here.
    with pytest.raises(InvalidStepWindowError):
        predict(
            latents=mx.ones((1, 2, 4)),
            latent_ids=mx.zeros((1, 2, 3)),
            prompt_embeds=mx.ones((1, 8, 16)),
            text_ids=mx.zeros((1, 8, 3)),
            negative_prompt_embeds=None,
            negative_text_ids=None,
            guidance=1.0,
            timestep=mx.array(1.0),
        )


def test_cfg_path_enforces_skip_window_validation():
    """v0.4.1+ behavior: skip-window validation runs on the FIRST gated
    call regardless of CFG. Pre-v0.4.1 deferred validation to a non-CFG
    step (and an all-CFG generation never reached one), which silently
    skipped the check. The new contract makes the gate fire eagerly so
    misconfigured runs fail fast.

    Regression guard: with skip_first=1 + skip_last=1 and active_num_steps=2,
    the sum (2) is not strictly less than active_num_steps (2), so the
    first CFG call must raise InvalidStepWindowError before reaching the
    transformer."""
    from mlx_teacache.errors import InvalidStepWindowError

    handle = _FakeHandle(skip_first_n_steps=1, skip_last_n_steps=1)
    _set_ctx_ready(handle, num_steps=2)
    factory = make_teacache_predict_factory(handle)

    def transformer(**kw):
        return mx.zeros((1, 2, 4))

    predict = factory(transformer)
    with pytest.raises(InvalidStepWindowError):
        predict(
            latents=mx.zeros((1, 2, 4)),
            latent_ids=mx.zeros((1, 2, 3)),
            prompt_embeds=mx.zeros((1, 8, 16)),
            text_ids=mx.zeros((1, 8, 3)),
            negative_prompt_embeds=mx.zeros((1, 8, 16)),
            negative_text_ids=mx.zeros((1, 8, 3)),
            guidance=3.5,
            timestep=mx.array(1.0),
        )
