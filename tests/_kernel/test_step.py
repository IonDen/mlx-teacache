"""The per-step skip policy every forward shares: the threshold-zero fast path, the
gate-to-stats mapping, the signal shape check, the once-per-generation window check,
the residual write and the reconstruction of a skipped body output."""

import mlx.core as mx
import pytest

from mlx_teacache._kernel.cache import TeaCacheState
from mlx_teacache._kernel.gate import GateDecision, GateKind
from mlx_teacache._kernel.stats import StepDecision
from mlx_teacache._kernel.step import (
    cache_residuals,
    check_signal_shape,
    fast_path_decision,
    is_disabled,
    reconstruct,
    step_decision,
    validate_window_once,
)
from mlx_teacache.errors import InternalStateError, InvalidStepWindowError, TransformerShapeError


def _decision(kind: GateKind, *, should_compute: bool, should_update_cache: bool) -> GateDecision:
    return GateDecision(
        kind=kind,
        should_compute=should_compute,
        should_update_cache=should_update_cache,
        rel_l1=None,
        predicted_distance=None,
        accumulated_distance=0.0,
    )


_SKIPPED = _decision("skipped", should_compute=False, should_update_cache=False)
_SEED = _decision("computed", should_compute=True, should_update_cache=True)
_FORCED = _decision("forced", should_compute=True, should_update_cache=False)


# --- is_disabled -----------------------------------------------------------


def test_is_disabled_at_and_below_zero() -> None:
    """bug caught: `< 0.0` makes threshold 0 build the gating tensors, losing byte parity at 0."""
    assert is_disabled(0.0) is True
    assert is_disabled(-0.1) is True
    assert is_disabled(1e-9) is False


# --- fast_path_decision ----------------------------------------------------


def test_fast_path_decision_uses_the_current_counter_and_accumulator() -> None:
    """bug caught: the record carries 0.0 for the accumulator, or `step_counter + 1`."""
    state = TeaCacheState(step_counter=3, accumulated_distance=0.25)
    got = fast_path_decision(state, timestep=481.5)
    assert got == StepDecision(
        step_idx=3, timestep=481.5, rel_l1=None, accumulated_distance=0.25, decision="computed"
    )
    # the forward advances the counter itself, after the tail
    assert state.step_counter == 3


# --- step_decision ---------------------------------------------------------


def test_step_decision_copies_the_gate_fields() -> None:
    """bug caught: rel_l1 and accumulated_distance swapped, or the kind not copied."""
    decision = GateDecision(
        kind="skipped",
        should_compute=False,
        should_update_cache=False,
        rel_l1=0.05,
        predicted_distance=0.1,
        accumulated_distance=0.3,
    )
    got = step_decision(decision, step_idx=7, timestep=123.0)
    assert got == StepDecision(
        step_idx=7, timestep=123.0, rel_l1=0.05, accumulated_distance=0.3, decision="skipped"
    )


# --- check_signal_shape ----------------------------------------------------


def test_check_signal_shape_names_previous_then_current() -> None:
    """bug caught: expected and actual swapped, or the wrong step index reported."""
    state = TeaCacheState(step_counter=5, previous_mod_input=mx.zeros((1, 4, 8)))
    with pytest.raises(TransformerShapeError, match=r"step_idx=5 expected=\(1, 4, 8\) actual=\(1, 4, 9\)"):
        check_signal_shape(state, mx.zeros((1, 4, 9)))


def test_check_signal_shape_accepts_no_previous_or_an_equal_shape() -> None:
    """bug caught: the first step (no anchor yet) or an unchanged shape raises."""
    check_signal_shape(TeaCacheState(), mx.zeros((1, 4, 8)))
    check_signal_shape(TeaCacheState(previous_mod_input=mx.ones((1, 4, 8))), mx.zeros((1, 4, 8)))


# --- validate_window_once --------------------------------------------------


def test_validate_window_once_refuses_a_short_window_without_latching() -> None:
    """bug caught: the check is skipped, or the latch is set before the check passes."""
    state = TeaCacheState()
    with pytest.raises(InvalidStepWindowError, match=r"active_num_steps=2 "):
        validate_window_once(state, active_num_steps=2, skip_first_n_steps=1, skip_last_n_steps=1)
    assert state.skip_window_validated is False


def test_validate_window_once_validates_then_latches() -> None:
    """bug caught: validation repeats on every step, or never latches."""
    state = TeaCacheState()
    validate_window_once(state, active_num_steps=3, skip_first_n_steps=1, skip_last_n_steps=1)
    assert state.skip_window_validated is True
    # latched: a window that would be refused is not checked again this generation
    validate_window_once(state, active_num_steps=2, skip_first_n_steps=1, skip_last_n_steps=1)
    assert state.skip_window_validated is True


def test_first_step_only_skips_after_step_zero() -> None:
    """bug caught: `first_step_only` ignored, or it also suppresses the check on step zero."""
    late = TeaCacheState(step_counter=1)
    validate_window_once(
        late, active_num_steps=2, skip_first_n_steps=1, skip_last_n_steps=1, first_step_only=True
    )
    assert late.skip_window_validated is False

    first = TeaCacheState(step_counter=0)
    with pytest.raises(InvalidStepWindowError):
        validate_window_once(
            first, active_num_steps=2, skip_first_n_steps=1, skip_last_n_steps=1, first_step_only=True
        )


def test_without_first_step_only_the_check_runs_on_a_later_step() -> None:
    """bug caught: the step-zero condition applied by default, so FLUX.2 and Z-Image
    (first gated call may not be step zero) stop validating."""
    state = TeaCacheState(step_counter=1)
    with pytest.raises(InvalidStepWindowError):
        validate_window_once(state, active_num_steps=2, skip_first_n_steps=1, skip_last_n_steps=1)


def test_validate_window_once_reports_the_nominal_count() -> None:
    """bug caught: the nominal schedule FLUX.1 and Qwen pass is dropped from the message."""
    with pytest.raises(InvalidStepWindowError, match=r"active_num_steps=2, nominal_num_inference_steps=4 "):
        validate_window_once(
            TeaCacheState(),
            active_num_steps=2,
            skip_first_n_steps=1,
            skip_last_n_steps=1,
            nominal_num_inference_steps=4,
        )


def test_validate_window_once_passes_the_skip_counts_through() -> None:
    """bug caught: the window is checked with the defaults instead of the handle's counts."""
    state = TeaCacheState()
    with pytest.raises(InvalidStepWindowError, match=r"skip_first=3, skip_last=2, active_num_steps=5 "):
        validate_window_once(state, active_num_steps=5, skip_first_n_steps=3, skip_last_n_steps=2)


# --- reconstruct -----------------------------------------------------------


def _state_with_residuals() -> TeaCacheState:
    return TeaCacheState(
        cached_residual=mx.array([1.0, 2.0]),
        cached_residual_neg=mx.array([10.0, 20.0]),
    )


def test_reconstruct_adds_the_requested_branch() -> None:
    """bug caught: the negative branch reads the positive residual (or the reverse)."""
    state = _state_with_residuals()
    body_in = mx.array([0.5, 0.25])
    assert reconstruct(body_in, state, _SKIPPED).tolist() == [1.5, 2.25]
    assert reconstruct(body_in, state, _SKIPPED, branch="pos").tolist() == [1.5, 2.25]
    assert reconstruct(body_in, state, _SKIPPED, branch="neg").tolist() == [10.5, 20.25]


@pytest.mark.parametrize("branch", ["pos", "neg"])
def test_reconstruct_without_a_residual_raises(branch: str) -> None:
    """bug caught: a skip with no residual returns `body_in` unchanged, or checks the other branch."""
    state = _state_with_residuals()
    if branch == "pos":
        state.cached_residual = None
    else:
        state.cached_residual_neg = None
    with pytest.raises(InternalStateError, match=f"branch {branch!r}"):
        reconstruct(mx.zeros((2,)), state, _SKIPPED, branch=branch)  # type: ignore[arg-type]


def test_the_missing_residual_message_differs_only_by_branch() -> None:
    """bug caught: the two branches carry diverging copies of the message."""
    messages = {}
    for branch in ("pos", "neg"):
        state = TeaCacheState()
        with pytest.raises(InternalStateError) as info:
            reconstruct(mx.zeros((2,)), state, _SKIPPED, branch=branch)  # type: ignore[arg-type]
        messages[branch] = str(info.value)
    assert messages["pos"].replace("'pos'", "'neg'") == messages["neg"]
    assert messages["pos"] != messages["neg"]


@pytest.mark.parametrize("decision", [_SEED, _FORCED], ids=["computed", "forced"])
def test_reconstruct_refuses_a_compute_decision(decision: GateDecision) -> None:
    """bug caught: a computed step's fresh body output is replaced by the stale residual."""
    with pytest.raises(InternalStateError):
        reconstruct(mx.zeros((2,)), _state_with_residuals(), decision)


# --- cache_residuals -------------------------------------------------------


def test_cache_residuals_writes_only_when_asked() -> None:
    """bug caught: a forced step (or a skip) writes its residual and poisons the cache."""
    body_out = mx.array([3.0, 5.0])
    body_in = mx.array([1.0, 1.0])
    for decision in (_FORCED, _SKIPPED):
        state = TeaCacheState()
        cache_residuals(state, decision, pos=(body_out, body_in), neg=(body_out, body_in))
        assert state.cached_residual is None
        assert state.cached_residual_neg is None


def test_cache_residuals_stores_out_minus_in_for_the_given_branch_only() -> None:
    """bug caught: `in - out` stored, or a missing branch overwritten."""
    keep = mx.array([9.0, 9.0])
    state = TeaCacheState(cached_residual_neg=keep)
    cache_residuals(state, _SEED, pos=(mx.array([3.0, 5.0]), mx.array([1.0, 1.0])))
    assert state.cached_residual is not None
    assert state.cached_residual.tolist() == [2.0, 4.0]
    assert state.cached_residual_neg is keep


class _SpyState(TeaCacheState):
    """A real TeaCacheState that records each store_residuals call before storing."""

    def __init__(self) -> None:
        super().__init__()
        self.store_calls: list[tuple[bool, bool]] = []

    def store_residuals(self, *, pos: mx.array | None = None, neg: mx.array | None = None) -> None:
        self.store_calls.append((pos is not None, neg is not None))
        super().store_residuals(pos=pos, neg=neg)


def test_cache_residuals_stores_both_branches_in_one_call() -> None:
    """bug caught: one store per branch, so a CFG step pays two host syncs instead of one."""
    state = _SpyState()
    cache_residuals(
        state,
        _SEED,
        pos=(mx.array([3.0, 5.0]), mx.array([1.0, 1.0])),
        neg=(mx.array([7.0, 9.0]), mx.array([1.0, 2.0])),
    )
    assert state.store_calls == [(True, True)]
    assert state.cached_residual is not None
    assert state.cached_residual_neg is not None
    assert state.cached_residual.tolist() == [2.0, 4.0]
    assert state.cached_residual_neg.tolist() == [6.0, 7.0]
