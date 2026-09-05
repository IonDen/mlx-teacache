# src/mlx_teacache/_kernel/gate.py
"""Canonical home for the gate primitives (extracted in v0.6.0).

Pure-math gating module. No mflux imports; only mlx.core for tensor ops.
Returns a structured GateDecision so the caller can drive both the compute
path AND the cache-update path explicitly."""

import math
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

GateKind = Literal["computed", "forced", "skipped", "numerical-miss"]

MAX_CONSECUTIVE_SKIPS = 8
"""Runaway-skip guard, not a tuning knob (intentional divergence from
upstream ali-vilab TeaCache, like the max(0,·) clamp below). The
origin-constrained in-repo fits are positive for small deltas but cross
zero at large ones (base-4b x≈0.24, z-image x≈0.29, qwen x≈0.78), past the
range they were calibrated on; there the max(0,·) clamp turns a large,
real change into a predicted change of zero, so the accumulator stops
advancing and a stale residual could be reused without bound at a
user-raised threshold. Observed max streaks at per-variant default
thresholds are recorded in docs/calibration.md; raise this constant only
from a re-measured run."""


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

    The element-wise difference stays in the inputs' dtype (bf16 in every shipped
    variant); the two reductions run in float32. ``mx.mean`` on a bf16 array
    returns a bf16 scalar, and that rounding is not systematic (it grows with the
    element count), so a polynomial calibrated at one resolution does not absorb
    it. The cast is a free pass over data the reduction already reads. Guards
    against division by zero with a small epsilon."""
    num = float(mx.mean(mx.abs(current - previous).astype(mx.float32)))
    denom = float(mx.mean(mx.abs(previous).astype(mx.float32)))
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
    # output doesn't poison the cache for a later threshold-gated step. In the
    # LEADING window the anchor does advance (upstream-faithful): the signal is
    # real, and the first gated step must measure a consecutive delta against
    # it. In the TRAILING window nothing gated follows, so the anchor is left
    # alone (writing it would only cost a host sync per trailing step).
    if step_idx < skip_first or step_idx >= num_steps - skip_last:
        if step_idx < skip_first and _all_finite(mod_in):
            state.previous_mod_input = mod_in
        return GateDecision(
            kind="forced",
            should_compute=True,
            should_update_cache=False,
            rel_l1=None,
            predicted_distance=None,
            accumulated_distance=state.accumulated_distance,
        )

    # Numerical safety: non-finite mod_in. Compute (recover), but NEVER cache —
    # and drop the residual cached BEFORE the miss, so the next finite step
    # re-seeds (compute + cache) instead of skipping on a residual that is now
    # >= 2 diffusion steps stale, judged against a pre-miss anchor. The anchor
    # itself cannot advance to a non-finite value; it stays where it was, and
    # the re-seed refreshes it. Accumulator and streak restart with the cache.
    if not _all_finite(mod_in):
        state.cached_residual = None
        state.cached_residual_neg = None
        state.accumulated_distance = 0.0
        state.consecutive_skips = 0
        return GateDecision(
            kind="numerical-miss",
            should_compute=True,
            should_update_cache=False,
            rel_l1=None,
            predicted_distance=None,
            accumulated_distance=0.0,
        )

    # Seed / re-seed: no anchor yet, OR an anchor without a cached residual
    # (possible now that the anchor advances on forced steps while the cache
    # does not). A skip without a residual to reuse must never be issued.
    if state.previous_mod_input is None or state.cached_residual is None:
        state.previous_mod_input = mod_in
        state.consecutive_skips = 0
        return GateDecision(
            kind="computed",
            should_compute=True,
            should_update_cache=True,
            rel_l1=None,
            predicted_distance=None,
            accumulated_distance=state.accumulated_distance,
        )

    rel_l1 = mean_abs_rel_l1(mod_in, state.previous_mod_input)
    # Option A anchoring (upstream-faithful, v0.10.0): advance on EVERY gated
    # step so rel_l1 is always the consecutive delta d(M_t, M_{t-1}) the
    # degree-4 polynomials were calibrated on. Before v0.10.0 the anchor
    # advanced only on computed steps, feeding cumulative drift into a
    # consecutive-delta calibration (a documented deviation, corrected here).
    state.previous_mod_input = mod_in
    predicted_raw = poly_eval(coefficients, rel_l1) if math.isfinite(rel_l1) else math.nan
    if not math.isfinite(predicted_raw):
        # A finite signal whose reduction overflowed (inf / inf = nan), or a
        # polynomial that blew up. Python's max(0.0, nan) is 0.0, which would
        # freeze the accumulator and let the gate skip until the runaway cap.
        # Treat it like a non-finite input: compute, drop the residual, restart
        # accumulator and streak. The anchor stays advanced (mod_in is finite).
        state.cached_residual = None
        state.cached_residual_neg = None
        state.accumulated_distance = 0.0
        state.consecutive_skips = 0
        return GateDecision(
            kind="numerical-miss",
            should_compute=True,
            should_update_cache=False,
            rel_l1=rel_l1,
            predicted_distance=None,
            accumulated_distance=0.0,
        )
    # Clamp at 0 so the accumulator is monotonic non-decreasing within a
    # generation. The `max(0.0, ...)` clamp is an INTENTIONAL divergence from
    # upstream ali-vilab TeaCache (which uses the raw polynomial): it keeps
    # the accumulator monotonic for origin-constrained FLUX.2 fits and
    # arbitrary user coefficients whose polynomial can dip negative near the
    # origin. Don't remove it to "match upstream".
    predicted = max(0.0, predicted_raw)
    new_acc = state.accumulated_distance + predicted

    if new_acc < rel_l1_thresh and state.consecutive_skips < MAX_CONSECUTIVE_SKIPS:
        state.accumulated_distance = new_acc
        state.consecutive_skips += 1
        return GateDecision(
            kind="skipped",
            should_compute=False,
            should_update_cache=False,
            rel_l1=rel_l1,
            predicted_distance=predicted,
            accumulated_distance=new_acc,
        )

    state.accumulated_distance = 0.0
    state.consecutive_skips = 0
    return GateDecision(
        kind="computed",
        should_compute=True,
        should_update_cache=True,
        rel_l1=rel_l1,
        predicted_distance=predicted,
        accumulated_distance=0.0,
    )
