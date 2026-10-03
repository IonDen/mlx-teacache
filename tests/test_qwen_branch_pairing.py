"""Qwen-Image CFG two-call branch-pairing. Pure-core: no mflux, no weights.

Qwen calls the transformer twice per denoising step (positive then negative).
CfgBranchPairer threads one shared gate decision + branch parity across the two
calls and resets on a fresh generation (lifecycle gen-ctx token change).
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

import mlx_teacache.variants.qwen_image.integration as integ
from mlx_teacache.variants.qwen_image.integration import _InternalHandle, qwen_forward_with_gate
from mlx_teacache.variants.qwen_image.pairing import CfgBranchPairer


def test_fresh_token_starts_positive() -> None:
    p = CfgBranchPairer()
    p.advance()  # move off the default (branch_idx -> 1) so the reset is non-trivial
    assert not p.is_positive()
    p.on_generation_token(1)  # a fresh token must reset to positive
    assert p.is_positive()


def test_alternates_within_one_generation() -> None:
    p = CfgBranchPairer()
    p.on_generation_token(1)
    assert p.is_positive()  # step 0 positive
    p.advance()
    p.on_generation_token(1)  # same token → no reset
    assert not p.is_positive()  # step 0 negative
    p.advance()
    p.on_generation_token(1)
    assert p.is_positive()  # step 1 positive


def test_new_token_resets_after_midpair_interrupt() -> None:
    p = CfgBranchPairer()
    p.on_generation_token(1)
    p.advance()  # interrupt lands here: positive done, negative pending
    p.on_generation_token(2)  # fresh generation
    assert p.is_positive()  # reset, not stuck on negative
    assert p.shared_decision is None


def test_shared_decision_survives_to_negative_call() -> None:
    p = CfgBranchPairer()
    p.on_generation_token(1)
    p.shared_decision = "DECISION"
    p.advance()
    p.on_generation_token(1)
    assert p.shared_decision == "DECISION"


def test_new_token_clears_stale_shared_decision() -> None:
    p = CfgBranchPairer()
    p.on_generation_token(1)
    p.shared_decision = "OLD"
    p.on_generation_token(2)
    assert p.shared_decision is None


def test_registry_discovers_qwen_image() -> None:
    from mlx_teacache.variants import _REGISTRY

    assert "qwen-image" in _REGISTRY
    assert _REGISTRY["qwen-image"]["META"]["variant_id"] == "qwen-image"


def _patch_physics(monkeypatch, *, signal_value: float) -> dict:
    calls = {"prelude": 0, "signal_a": 0, "run_body": 0, "tail": 0}

    class _Pre:
        def __init__(self, hidden_states: mx.array) -> None:
            # h_in is the call's own hidden_states, so a skip step's reconstruction
            # is observably built on the CURRENT step's input, not a replayed one.
            self.h_in = hidden_states
            self.text_embeddings = mx.zeros((1, 8))

    def fake_prelude(inner, t, config, hidden_states):  # noqa: ANN001
        calls["prelude"] += 1
        return _Pre(hidden_states)

    def fake_signal_a(inner, pre):  # noqa: ANN001
        calls["signal_a"] += 1
        return mx.full((1, 4, 8), signal_value)

    def fake_run_body(inner, pre, *a, **k):  # noqa: ANN001, ANN002, ANN003
        # Branch-dependent body: a negative-valued encoder_hidden_states marks the
        # negative (unconditional) call and gets residual 2; any other call gets 1.
        # Distinct residuals make a positive/negative cache swap observable.
        calls["run_body"] += 1
        negative = bool(mx.any(k["encoder_hidden_states"] < 0).item())
        return pre.h_in + mx.full((1, 4, 8), 2.0 if negative else 1.0)

    def fake_tail(inner, body_out, pre, *a, **k):  # noqa: ANN001, ANN002, ANN003
        calls["tail"] += 1
        return body_out

    monkeypatch.setattr(integ, "_qwen_prelude", fake_prelude)
    monkeypatch.setattr(integ, "_qwen_signal_a", fake_signal_a)
    monkeypatch.setattr(integ, "_qwen_run_body", fake_run_body)
    monkeypatch.setattr(integ, "_qwen_tail", fake_tail)
    return calls


def _orch_handle(thresh: float) -> _InternalHandle:
    h = _InternalHandle(
        rel_l1_thresh=thresh,
        coefficients=(0.0, 0.0, 0.0, 0.0, 1.0),  # poly→1.0/step ⇒ accumulate fast ⇒ compute
        skip_first_n_steps=1,
        skip_last_n_steps=1,
    )
    h._gen_ctx.token = 1
    h._gen_ctx.active_num_steps = 4
    return h


def _orch_call(inner, handle, *, t, num_inference_steps=4):  # noqa: ANN001
    return qwen_forward_with_gate(
        inner,
        handle,
        t=t,
        config=SimpleNamespace(num_inference_steps=num_inference_steps),
        hidden_states=mx.zeros((1, 4, 8)),
        encoder_hidden_states=mx.zeros((1, 2, 8)),
        encoder_hidden_states_mask=mx.ones((1, 2)),
        qwen_image_ids=None,
        cond_image_grid=None,
    )


def test_two_calls_record_once_and_advance_step_counter_once(monkeypatch) -> None:  # noqa: ANN001
    calls = _patch_physics(monkeypatch, signal_value=0.5)
    inner = SimpleNamespace()
    handle = _orch_handle(thresh=0.20)
    _orch_call(inner, handle, t=0)  # positive
    _orch_call(inner, handle, t=0)  # negative
    assert handle._state.cache.step_counter == 1
    assert len(handle._state.stats._staging.decisions) == 1
    assert calls["run_body"] == 2


def test_gate_runs_on_positive_only(monkeypatch) -> None:  # noqa: ANN001
    calls = _patch_physics(monkeypatch, signal_value=0.5)
    inner = SimpleNamespace()
    handle = _orch_handle(thresh=0.20)
    _orch_call(inner, handle, t=0)  # positive: signal_a computed
    _orch_call(inner, handle, t=0)  # negative: NO new signal_a
    assert calls["signal_a"] == 1


def test_skip_step_reconstructs_from_cache_without_running_body(monkeypatch) -> None:  # noqa: ANN001
    calls = _patch_physics(monkeypatch, signal_value=0.5)
    inner = SimpleNamespace()
    h = _InternalHandle(
        rel_l1_thresh=0.20,
        coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),  # poly→0 ⇒ accumulator never grows ⇒ skip after seed
        skip_first_n_steps=1,
        skip_last_n_steps=1,
    )
    h._gen_ctx.token = 1
    h._gen_ctx.active_num_steps = 6

    def call(t):  # noqa: ANN001, ANN202
        return qwen_forward_with_gate(
            inner,
            h,
            t=t,
            config=SimpleNamespace(num_inference_steps=6),
            hidden_states=mx.zeros((1, 4, 8)),
            encoder_hidden_states=mx.zeros((1, 2, 8)),
            encoder_hidden_states_mask=mx.ones((1, 2)),
            qwen_image_ids=None,
            cond_image_grid=None,
        )

    # step 0 (forced) + step 1 (seed compute): 4 body runs, cache seeded.
    for t in (0, 0, 1, 1):
        call(t)
    assert calls["run_body"] == 4
    assert h._state.cache.cached_residual is not None
    assert h._state.cache.cached_residual_neg is not None

    # step 2 (eligible, predicted=0 ⇒ skip): NO new body runs; reconstruct from cache.
    out_pos = call(2)
    out_neg = call(2)
    assert calls["run_body"] == 4  # body NOT run on the skip step
    # fake_run_body returns pre.h_in + ones ⇒ residual = ones ⇒ reconstruction = zeros + ones = ones.
    assert bool(mx.array_equal(out_pos, mx.ones((1, 4, 8))))
    assert bool(mx.array_equal(out_neg, mx.ones((1, 4, 8))))


def test_negative_branch_reuses_its_own_residual_on_skip(monkeypatch) -> None:  # noqa: ANN001
    """Bug caught: the Qwen forward rebuilding a skipped negative call with `+ state.cached_residual`
    instead of `+ state.cached_residual_neg`.

    The seed step caches residual 1 on the positive call and residual 2 on the
    negative call; the forced skip on step 1 must rebuild each branch from its
    own residual on top of that step's h_in."""
    calls = _patch_physics(monkeypatch, signal_value=0.5)
    inner = SimpleNamespace()
    h = _InternalHandle(
        rel_l1_thresh=0.5,
        coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),  # poly→0 ⇒ step 1 skips after the step-0 seed
        skip_first_n_steps=0,
        skip_last_n_steps=1,
    )
    h._gen_ctx.token = 1
    h._gen_ctx.active_num_steps = 4

    def call(t: int, *, hidden: float, negative: bool) -> mx.array:
        return qwen_forward_with_gate(
            inner,
            h,
            t=t,
            config=SimpleNamespace(num_inference_steps=4),
            hidden_states=mx.full((1, 4, 8), hidden),
            encoder_hidden_states=mx.full((1, 2, 8), -1.0 if negative else 1.0),
            encoder_hidden_states_mask=mx.ones((1, 2)),
            qwen_image_ids=None,
            cond_image_grid=None,
        )

    # Step 0 seeds both branches (positive residual 1, negative residual 2).
    call(0, hidden=10.0, negative=False)
    call(0, hidden=20.0, negative=True)
    assert calls["run_body"] == 2

    # Step 1 is a forced skip; each branch has its own, new h_in.
    h_in_pos = mx.full((1, 4, 8), 3.0)
    h_in_neg = mx.full((1, 4, 8), 5.0)
    out_pos_skip = call(1, hidden=3.0, negative=False)
    out_neg_skip = call(1, hidden=5.0, negative=True)
    assert calls["run_body"] == 2  # the body did not run on the skip step
    assert [d.decision for d in h._state.stats._staging.decisions] == ["computed", "skipped"]
    assert mx.array_equal(out_neg_skip, h_in_neg + 2).item()
    assert mx.array_equal(out_pos_skip, h_in_pos + 1).item()


def test_fast_path_marks_cfg_active(monkeypatch) -> None:  # noqa: ANN001
    _patch_physics(monkeypatch, signal_value=0.5)
    inner = SimpleNamespace()
    handle = _orch_handle(thresh=0.0)
    _orch_call(inner, handle, t=0)  # positive
    assert handle._state.stats._staging.cfg_was_active is True


def test_overwide_skip_window_raises(monkeypatch) -> None:  # noqa: ANN001
    from mlx_teacache.errors import InvalidStepWindowError

    _patch_physics(monkeypatch, signal_value=0.5)
    inner = SimpleNamespace()
    handle = _InternalHandle(
        rel_l1_thresh=0.20,
        coefficients=(0.0, 0.0, 0.0, 0.0, 1.0),
        skip_first_n_steps=3,
        skip_last_n_steps=3,
    )
    handle._gen_ctx.token = 1
    handle._gen_ctx.active_num_steps = 4  # 3 + 3 >= 4 → invalid
    # Bug: the Qwen call site drops nominal_num_inference_steps, so an img2img refusal never shows the 20-step schedule.
    with pytest.raises(InvalidStepWindowError, match=r"active_num_steps=4, nominal_num_inference_steps=20"):
        _orch_call(inner, handle, t=0, num_inference_steps=20)


def test_fast_path_thresh_zero_advances_once_no_cache(monkeypatch) -> None:  # noqa: ANN001
    calls = _patch_physics(monkeypatch, signal_value=0.5)
    inner = SimpleNamespace()
    handle = _orch_handle(thresh=0.0)
    _orch_call(inner, handle, t=0)
    _orch_call(inner, handle, t=0)
    assert handle._state.cache.step_counter == 1
    assert handle._state.cache.cached_residual is None
    assert len(handle._state.stats._staging.decisions) == 1
    assert calls["run_body"] == 2


def test_interrupt_midpair_then_fresh_generation_clears_stale_residual(monkeypatch) -> None:  # noqa: ANN001
    """Interrupt between the positive and negative calls leaves a positive residual
    cached but no negative residual (asymmetric). The next generation's lifecycle
    reset (reset_for_new_generation) + the pairer's token-reset must clear that
    stale state so it cannot leak into gen 2 — the documented interrupt heal."""
    _patch_physics(monkeypatch, signal_value=0.5)
    inner = SimpleNamespace()
    handle = _InternalHandle(
        rel_l1_thresh=0.20,
        coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),  # poly→0; the first eligible step seeds + caches
        skip_first_n_steps=0,
        skip_last_n_steps=0,
    )
    handle._gen_ctx.token = 1
    handle._gen_ctx.active_num_steps = 4

    def call(t: int):  # noqa: ANN202
        return qwen_forward_with_gate(
            inner,
            handle,
            t=t,
            config=SimpleNamespace(num_inference_steps=4),
            hidden_states=mx.zeros((1, 4, 8)),
            encoder_hidden_states=mx.zeros((1, 2, 8)),
            encoder_hidden_states_mask=mx.ones((1, 2)),
            qwen_image_ids=None,
            cond_image_grid=None,
        )

    call(0)  # gen 1, step 0 positive: seed compute caches the POSITIVE residual
    # INTERRUPT here — the negative call never runs.
    assert handle._state.cache.cached_residual is not None
    assert handle._state.cache.cached_residual_neg is None  # asymmetric interrupt state

    # Fresh generation: what the lifecycle's call_before_loop does.
    handle._gen_ctx.token = 2
    handle._state.cache.reset_for_new_generation(num_steps=4)
    assert handle._state.cache.cached_residual is None  # stale positive residual cleared
    assert handle._state.cache.cached_residual_neg is None

    call(0)  # gen 2, step 0 positive: re-seeds cleanly on the new token
    assert handle._pairer.last_seen_token == 2
    assert handle._state.cache.cached_residual is not None  # gen-2 fresh seed, not the stale one
