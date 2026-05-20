# tests/_kernel/test_gate_equivalence.py
"""Phase A discipline: _kernel.gate.gate_step must behave identically to
the v0.5.x src/mlx_teacache/gate.gate_step across every decision branch."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_teacache import gate as legacy_gate
from mlx_teacache._kernel import gate as kernel_gate
from mlx_teacache.cache import TeaCacheState
from mlx_teacache.variants.flux1_dev.config import COEFFICIENTS as _UPSTREAM_FLUX_COEFFS


@pytest.mark.parametrize(
    "rel_l1_thresh,skip_first,skip_last,num_steps,step_idx,seed_prev",
    [
        (0.0, 1, 1, 25, 5, True),  # threshold short-circuit
        (-0.1, 1, 1, 25, 5, True),
        (0.2, 1, 1, 25, 0, True),  # forced (skip_first)
        (0.2, 1, 1, 25, 24, True),  # forced (skip_last edge)
        (0.2, 1, 1, 25, 5, False),  # first eligible, no previous_mod_input
        (0.2, 1, 1, 25, 5, True),  # gated middle step
    ],
)
def test_gate_step_equivalence(rel_l1_thresh, skip_first, skip_last, num_steps, step_idx, seed_prev):
    mod_in = mx.array([[1.0, 2.0, 3.0]])
    prev = mx.array([[0.9, 2.1, 3.05]]) if seed_prev else None

    legacy_state = TeaCacheState(previous_mod_input=prev, accumulated_distance=0.1)
    kernel_state = TeaCacheState(previous_mod_input=prev, accumulated_distance=0.1)

    legacy_decision = legacy_gate.gate_step(
        legacy_state,
        rel_l1_thresh=rel_l1_thresh,
        coefficients=_UPSTREAM_FLUX_COEFFS,
        skip_first=skip_first,
        skip_last=skip_last,
        num_steps=num_steps,
        step_idx=step_idx,
        mod_in=mod_in,
    )
    kernel_decision = kernel_gate.gate_step(
        kernel_state,
        rel_l1_thresh=rel_l1_thresh,
        coefficients=_UPSTREAM_FLUX_COEFFS,
        skip_first=skip_first,
        skip_last=skip_last,
        num_steps=num_steps,
        step_idx=step_idx,
        mod_in=mod_in,
    )
    assert legacy_decision == kernel_decision
    assert legacy_state.accumulated_distance == kernel_state.accumulated_distance


def test_poly_eval_equivalence():
    for x in [0.0, 0.01, 0.1, 0.5, 1.0]:
        assert legacy_gate.poly_eval(_UPSTREAM_FLUX_COEFFS, x) == kernel_gate.poly_eval(
            _UPSTREAM_FLUX_COEFFS, x
        )


def test_mean_abs_rel_l1_equivalence():
    curr = mx.array([1.0, 2.0, 3.0])
    prev = mx.array([0.9, 2.1, 3.05])
    assert legacy_gate.mean_abs_rel_l1(curr, prev) == kernel_gate.mean_abs_rel_l1(curr, prev)
