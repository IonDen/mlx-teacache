# src/mlx_teacache/_kernel/gate.py
"""Canonical home for the gate primitives (extracted in v0.6.0).

Pure-math gating module. No mflux imports; only mlx.core for tensor ops.
Returns a structured GateDecision so the caller can drive both the compute
path AND the cache-update path explicitly."""

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

GateKind = Literal["computed", "forced", "skipped", "numerical-miss"]


@dataclass(frozen=True)
class GateDecision:
    kind: GateKind
    should_compute: bool
    should_update_cache: bool
    rel_l1: float | None
    predicted_distance: float | None
    accumulated_distance: float


def poly_eval(coeffs: tuple[float, float, float, float, float], x: float) -> float:
    """Evaluate a 4th-degree polynomial in [c4, c3, c2, c1, c0] order (highest first)."""
    c4, c3, c2, c1, c0 = coeffs
    return ((((c4 * x) + c3) * x + c2) * x + c1) * x + c0


def mean_abs_rel_l1(current: mx.array, previous: mx.array) -> float:
    """Mean absolute relative L1 distance: mean(|current - previous|) / mean(|previous|).
    Guards against division by zero with a small epsilon."""
    num = float(mx.mean(mx.abs(current - previous)))
    denom = float(mx.mean(mx.abs(previous)))
    return num / max(denom, 1e-12)


def _all_finite(t: mx.array) -> bool:
    return bool(mx.all(mx.isfinite(t)))


def gate_step(  # type: ignore[no-untyped-def]
    state,
    *,
    rel_l1_thresh: float,
    coefficients: tuple[float, float, float, float, float],
    skip_first: int,
    skip_last: int,
    num_steps: int,
    step_idx: int,
    mod_in: mx.array,
) -> GateDecision:
    """Return a structured decision for one denoising step.

    `state` is a mlx_teacache.cache.TeaCacheState (duck-typed here to keep the
    gate module pure / circular-import-free)."""
    # Hard short-circuit: threshold <= 0 ⇒ always compute, never cache.
    # At non-positive threshold no future step can ever be skipped, so the
    # cache can never be consumed. Setting should_update_cache=False avoids
    # building cached_residual = body_out - body_in, which would otherwise
    # keep the body/tail intermediates alive past the tail and perturb
    # Metal in-place buffer donation. Guards against negative polynomial
    # outputs and arbitrary custom coefficients.
    if rel_l1_thresh <= 0.0:
        return GateDecision(
            kind="computed",
            should_compute=True,
            should_update_cache=False,
            rel_l1=None,
            predicted_distance=None,
            accumulated_distance=state.accumulated_distance,
        )

    # Forced windows: full forward, but DO NOT update the cache so a forced
    # output doesn't poison the cache for a later threshold-gated step.
    if step_idx < skip_first or step_idx >= num_steps - skip_last:
        return GateDecision(
            kind="forced",
            should_compute=True,
            should_update_cache=False,
            rel_l1=None,
            predicted_distance=None,
            accumulated_distance=state.accumulated_distance,
        )

    # Numerical safety: non-finite mod_in. Compute (recover), but NEVER cache.
    if not _all_finite(mod_in):
        return GateDecision(
            kind="numerical-miss",
            should_compute=True,
            should_update_cache=False,
            rel_l1=None,
            predicted_distance=None,
            accumulated_distance=state.accumulated_distance,
        )

    # First eligible step with no cache yet: compute and cache (seeds the cache).
    if state.previous_mod_input is None:
        return GateDecision(
            kind="computed",
            should_compute=True,
            should_update_cache=True,
            rel_l1=None,
            predicted_distance=None,
            accumulated_distance=state.accumulated_distance,
        )

    # Compute rel_l1 and predicted distance; clamp at 0 so the accumulator is
    # monotonic non-decreasing within a generation. The `max(0.0, ...)` clamp is an
    # INTENTIONAL divergence from upstream ali-vilab TeaCache (which uses the raw
    # polynomial): it keeps the accumulator monotonic for origin-constrained FLUX.2
    # fits and arbitrary user coefficients whose polynomial can dip negative near
    # the origin. Don't remove it to "match upstream".
    rel_l1 = mean_abs_rel_l1(mod_in, state.previous_mod_input)
    predicted = max(0.0, poly_eval(coefficients, rel_l1))
    new_acc = state.accumulated_distance + predicted

    if new_acc < rel_l1_thresh:
        state.accumulated_distance = new_acc
        return GateDecision(
            kind="skipped",
            should_compute=False,
            should_update_cache=False,
            rel_l1=rel_l1,
            predicted_distance=predicted,
            accumulated_distance=new_acc,
        )

    state.accumulated_distance = 0.0
    return GateDecision(
        kind="computed",
        should_compute=True,
        should_update_cache=True,
        rel_l1=rel_l1,
        predicted_distance=predicted,
        accumulated_distance=0.0,
    )
