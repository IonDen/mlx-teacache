# tests/_kernel/test_gate_equivalence.py
"""`mlx_teacache.gate` is a pure re-export shim of `mlx_teacache._kernel.gate`.

The legacy import path must resolve to the *same* objects as the kernel — not
a re-implementation that could silently drift. This file pins only that shim
identity; the behavioral branch coverage of `gate_step` (threshold short-circuit,
forced windows, skip/compute decisions, NaN handling) lives in `tests/test_gate.py`.

A divergent definition in the shim (e.g. `gate.py` growing its own `def gate_step`)
turns the identity assertion red.
"""


def test_shim_re_exports_gate_symbols_by_identity() -> None:
    from mlx_teacache import gate as legacy_gate
    from mlx_teacache._kernel import gate as kernel_gate

    assert legacy_gate.gate_step is kernel_gate.gate_step
    assert legacy_gate.poly_eval is kernel_gate.poly_eval
    assert legacy_gate.mean_abs_rel_l1 is kernel_gate.mean_abs_rel_l1
    assert legacy_gate.GateDecision is kernel_gate.GateDecision
    assert legacy_gate.GateKind is kernel_gate.GateKind
