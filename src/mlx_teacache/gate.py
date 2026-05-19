# src/mlx_teacache/gate.py
"""Compatibility shim. Canonical home is `mlx_teacache._kernel.gate`."""

from mlx_teacache._kernel.gate import (
    GateDecision,
    GateKind,
    gate_step,
    mean_abs_rel_l1,
    poly_eval,
)

__all__ = ["GateDecision", "GateKind", "gate_step", "mean_abs_rel_l1", "poly_eval"]
