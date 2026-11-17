"""The per-step skip policy shared by every forward.

Each helper replaces one expression of a forward at the same position, so the graph a
forward builds is unchanged: the threshold-zero fast path, the gate-to-stats mapping,
the gating-signal shape check, the once-per-generation step-window check, the residual
write and the reconstruction of a skipped body output.

Pure MLX: no mflux; imports only `_kernel` and `errors`."""

from typing import Literal

import mlx.core as mx

from mlx_teacache._kernel.cache import TeaCacheState
from mlx_teacache._kernel.gate import GateDecision
from mlx_teacache._kernel.stats import StepDecision
from mlx_teacache._kernel.window import check_step_window
from mlx_teacache.errors import InternalStateError, TransformerShapeError

Branch = Literal["pos", "neg"]

_MISSING_RESIDUAL = (
    "the cached residual for branch {branch!r} is None on a skipped step; the gate never "
    "skips before a computed step seeds the cache, so this is a bug in mlx-teacache. "
    "Please file an issue."
)


def is_disabled(rel_l1_thresh: float) -> bool:
    """True when no step can skip, so the forward runs without building any gating tensor."""
    return rel_l1_thresh <= 0.0


def fast_path_decision(state: TeaCacheState, *, timestep: float) -> StepDecision:
    """The record of a step run on the threshold-zero fast path."""
    return StepDecision(
        step_idx=state.step_counter,
        timestep=timestep,
        rel_l1=None,
        accumulated_distance=state.accumulated_distance,
        decision="computed",
    )


def step_decision(decision: GateDecision, *, step_idx: int, timestep: float) -> StepDecision:
    """The record of a gated step."""
    return StepDecision(
        step_idx=step_idx,
        timestep=timestep,
        rel_l1=decision.rel_l1,
        accumulated_distance=decision.accumulated_distance,
        decision=decision.kind,
    )


def check_signal_shape(state: TeaCacheState, signal: mx.array) -> None:
    """Raise TransformerShapeError when the gating signal's shape differs from the previous step's."""
    previous = state.previous_mod_input
    if previous is not None and signal.shape != previous.shape:
        raise TransformerShapeError(
            step_idx=state.step_counter,
            expected=previous.shape,
            actual=signal.shape,
        )


def validate_window_once(
    state: TeaCacheState,
    *,
    active_num_steps: int,
    skip_first_n_steps: int,
    skip_last_n_steps: int,
    nominal_num_inference_steps: int | None = None,
    first_step_only: bool = False,
) -> None:
    """Check the step window once per generation, then latch. With `first_step_only`, only a
    call on the generation's first step checks."""
    if state.skip_window_validated:
        return
    if first_step_only and state.step_counter != 0:
        return
    check_step_window(
        active_num_steps,
        skip_first_n_steps=skip_first_n_steps,
        skip_last_n_steps=skip_last_n_steps,
        nominal_num_inference_steps=nominal_num_inference_steps,
    )
    state.skip_window_validated = True


def cache_residuals(
    state: TeaCacheState,
    decision: GateDecision,
    *,
    pos: tuple[mx.array, mx.array] | None = None,
    neg: tuple[mx.array, mx.array] | None = None,
) -> None:
    """Store `body_out - body_in` for each `(body_out, body_in)` branch given, when the decision
    updates the cache. Both branches are stored together, so a CFG step evaluates once."""
    if not decision.should_update_cache:
        return
    pos_residual = None if pos is None else pos[0] - pos[1]
    neg_residual = None if neg is None else neg[0] - neg[1]
    state.store_residuals(pos=pos_residual, neg=neg_residual)


def reconstruct(
    body_in: mx.array,
    state: TeaCacheState,
    decision: GateDecision,
    *,
    branch: Branch = "pos",
) -> mx.array:
    """The body output of a skipped step: `body_in` plus the branch's cached residual."""
    if decision.should_compute:
        raise InternalStateError(
            f"reconstruct was called for a {decision.kind!r} step, which runs the body; "
            "this is a bug in mlx-teacache. Please file an issue."
        )
    residual = state.cached_residual if branch == "pos" else state.cached_residual_neg
    if residual is None:
        raise InternalStateError(_MISSING_RESIDUAL.format(branch=branch))
    return body_in + residual
