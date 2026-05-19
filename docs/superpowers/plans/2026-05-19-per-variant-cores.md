# v0.6.0 — per-variant cores + shared algorithmic kernel implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md`
**Audit:** `docs/superpowers/notes/2026-05-19-per-variant-cores-plan-audit.md`

**Goal:** Restructure `src/mlx_teacache/` from "shared everything, per-variant config" to per-variant assemblies + a shared algorithmic kernel. Public API surface unchanged at every v0.5.x import path; internal layout becomes per-variant directories under `src/mlx_teacache/variants/` plus pure-algorithm primitives under `src/mlx_teacache/_kernel/`.

**Architecture:** Six variants get directories with `config.py` (META + coefficients + recipes, mflux-free), `detect.py` (`matches(flux)`, mflux-free), and `integration.py` (forward wrapper + `VariantPatch`, mflux-touching, lazy-imported). Pure-algorithm primitives (gate, cache state, stats) live under `_kernel/`. Shared mflux machinery (the generation-lifecycle callback that owns commit/discard) stays at `src/mlx_teacache/integrations/mflux/lifecycle.py` and is imported by each variant's integration. `TeaCacheHandle` is variant-agnostic; per-variant teardown is captured into a `VariantPatch` (rollback + finalizer callback lists) returned by each variant's `apply()`.

**Tech Stack:** Python 3.11+, MLX, mflux 0.17.x (lazy-imported), pytest 8 + hypothesis, ruff, `mypy --strict`, `uv` for dev, `hatchling` + `hatch-vcs` for build.

---

## Audit-corrected execution discipline — READ BEFORE STARTING

Per `docs/superpowers/notes/2026-05-19-per-variant-cores-plan-audit.md`, the plan MUST follow these rules. Every task below is written with these baked in; do not deviate.

1. **Extract verbatim, then refactor.** `_kernel/gate.py`, `_kernel/cache.py`, and `_kernel/stats.py` start as byte-for-byte copies of the current `src/mlx_teacache/gate.py`, `cache.py`, `stats.py`. Field names, function signatures, decision kinds, clamp behavior, accumulator-reset semantics — all preserved. New helper seams are extracted only after equivalence tests prove parity.
2. **Real file paths only.** The v0.5.x tree has: `src/mlx_teacache/{gate,cache,stats,coefficients,api,errors,__init__,_version}.py` and `src/mlx_teacache/integrations/mflux/{detect,flux1,flux2,forward,lifecycle,__init__}.py`. There is no top-level `state.py` or `lifecycle.py`. Cleanup tasks delete files that actually exist.
3. **Preserve the public `apply_teacache` signature.** `apply_teacache(flux, *, rel_l1_thresh=..., coefficients=None, skip_first_n_steps=1, skip_last_n_steps=1) -> TeaCacheHandle`. All four explicit keyword params survive. The dispatcher accepts them; each variant's `apply()` accepts them. Tests gate on exact parameter names, not `**kwargs`.
4. **Never spell out coefficient tuples in the plan.** Workers copy from `src/mlx_teacache/coefficients.py` (constants `_UPSTREAM_FLUX_COEFFS`, `_FLUX2_KLEIN_4B_COEFFS`, `_FLUX2_KLEIN_9B_COEFFS`, `_FLUX2_KLEIN_BASE_4B_COEFFS`) when writing each variant's `config.py`. An identity test asserts the variant's `COEFFICIENTS` is bit-equal to the legacy registry tuple before the legacy registry is deleted.
5. **Integration ports are byte-for-byte first.** Variant `integration.py` copies `flux1_forward_with_gate()` / `flux2_forward_with_gate()` / `flux2_cfg_forward_with_gate()` and `ProxyFlux1Transformer` from `integrations/mflux/{forward,flux1,flux2}.py` exactly as they are. Do not invent new helper boundaries during the port.
6. **Stats commit/discard stays attached to the mflux generation lifecycle.** `VariantPatch` owns teardown of mutations + callback unsubscription only. `finalize_last_generation()` is called from `wrap_generate_image`'s try/finally in `integrations/mflux/lifecycle.py`, NOT from `handle.restore()`. Failed/interrupted generations must continue to leave no public stats trace.
7. **Shared mflux machinery stays shared.** `src/mlx_teacache/integrations/mflux/lifecycle.py` (the `GenerationContextCallback` + `wrap_generate_image`) is not duplicated per variant. Each variant's `integration.py` imports it. The duplication discipline of the spec applies to the family-forward code, not to the mflux-callback plumbing.

---

## Phase A — `_kernel/` extraction (verbatim)

Goal: move existing pure-algorithm modules into `_kernel/` byte-for-byte. Each move pairs with a compatibility shim at the old path. Equivalence tests prove the move did not change behavior.

### Task 1: Scaffold `_kernel/` + no-mflux import guard

**Files:**
- Create: `src/mlx_teacache/_kernel/__init__.py`
- Create: `tests/_kernel/__init__.py`
- Create: `tests/_kernel/test_kernel_no_mflux_import.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/_kernel/test_kernel_no_mflux_import.py
"""Pure-algorithm primitives have no business pulling mflux. Walks every
module under _kernel/ in a simulated no-mflux environment and confirms
they import cleanly."""
from __future__ import annotations

import importlib
import pkgutil


def test_kernel_subtree_imports_without_mflux(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "mflux", None)

    import mlx_teacache._kernel as kernel_pkg
    importlib.reload(kernel_pkg)
    for _, name, _ in pkgutil.walk_packages(kernel_pkg.__path__, kernel_pkg.__name__ + "."):
        importlib.import_module(name)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/_kernel/test_kernel_no_mflux_import.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_teacache._kernel'`.

- [ ] **Step 3: Create the empty package**

```python
# src/mlx_teacache/_kernel/__init__.py
"""Pure-algorithm primitives. No mflux imports anywhere in this subtree.
See docs/superpowers/specs/2026-05-19-per-variant-cores-design.md.
"""
```

```python
# tests/_kernel/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/_kernel/test_kernel_no_mflux_import.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mlx_teacache/_kernel/ tests/_kernel/
git commit -m "feat(_kernel): scaffold pure-algorithm package + no-mflux guard"
```

---

### Task 2: `_kernel.gate` — verbatim extraction + shim + equivalence test

**Files:**
- Create: `src/mlx_teacache/_kernel/gate.py` (byte-for-byte copy of v0.5.x `src/mlx_teacache/gate.py`)
- Modify: `src/mlx_teacache/gate.py` (becomes a re-export shim)
- Create: `tests/_kernel/test_gate_equivalence.py`

The v0.5.x `gate.py` defines `GateDecision`, `GateKind`, `poly_eval`, `mean_abs_rel_l1`, `_all_finite`, `gate_step`. The polynomial is evaluated against per-step `rel_l1` (NOT against an accumulator), the predicted distance is clamped at zero, added to `state.accumulated_distance`, and the accumulator is reset to 0.0 on a compute. Every short-circuit branch (threshold ≤ 0, forced window, numerical-miss, cache-seeding) preserved.

- [ ] **Step 1: Inspect the source**

```bash
wc -l src/mlx_teacache/gate.py
grep -n "^def \|^class \|^GateKind\|^@dataclass" src/mlx_teacache/gate.py
```

Expected: ~135 lines; `GateKind` Literal, `GateDecision` dataclass, `poly_eval`, `mean_abs_rel_l1`, `_all_finite`, `gate_step` functions.

- [ ] **Step 2: Write the equivalence test (FAILS before extraction)**

```python
# tests/_kernel/test_gate_equivalence.py
"""Phase A discipline: _kernel.gate.gate_step must behave identically to
the v0.5.x src/mlx_teacache/gate.gate_step across every decision branch."""
from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_teacache import gate as legacy_gate
from mlx_teacache._kernel import gate as kernel_gate
from mlx_teacache.cache import TeaCacheState
from mlx_teacache.coefficients import _UPSTREAM_FLUX_COEFFS


@pytest.mark.parametrize("rel_l1_thresh,skip_first,skip_last,num_steps,step_idx,seed_prev", [
    (0.0, 1, 1, 25, 5, True),       # threshold short-circuit
    (-0.1, 1, 1, 25, 5, True),
    (0.2, 1, 1, 25, 0, True),       # forced (skip_first)
    (0.2, 1, 1, 25, 24, True),      # forced (skip_last edge)
    (0.2, 1, 1, 25, 5, False),      # first eligible, no previous_mod_input
    (0.2, 1, 1, 25, 5, True),       # gated middle step
])
def test_gate_step_equivalence(rel_l1_thresh, skip_first, skip_last, num_steps, step_idx, seed_prev):
    mod_in = mx.array([[1.0, 2.0, 3.0]])
    prev = mx.array([[0.9, 2.1, 3.05]]) if seed_prev else None

    legacy_state = TeaCacheState(previous_mod_input=prev, accumulated_distance=0.1)
    kernel_state = TeaCacheState(previous_mod_input=prev, accumulated_distance=0.1)

    legacy_decision = legacy_gate.gate_step(
        legacy_state, rel_l1_thresh=rel_l1_thresh, coefficients=_UPSTREAM_FLUX_COEFFS,
        skip_first=skip_first, skip_last=skip_last, num_steps=num_steps,
        step_idx=step_idx, mod_in=mod_in,
    )
    kernel_decision = kernel_gate.gate_step(
        kernel_state, rel_l1_thresh=rel_l1_thresh, coefficients=_UPSTREAM_FLUX_COEFFS,
        skip_first=skip_first, skip_last=skip_last, num_steps=num_steps,
        step_idx=step_idx, mod_in=mod_in,
    )
    assert legacy_decision == kernel_decision
    assert legacy_state.accumulated_distance == kernel_state.accumulated_distance


def test_poly_eval_equivalence():
    for x in [0.0, 0.01, 0.1, 0.5, 1.0]:
        assert legacy_gate.poly_eval(_UPSTREAM_FLUX_COEFFS, x) == kernel_gate.poly_eval(_UPSTREAM_FLUX_COEFFS, x)


def test_mean_abs_rel_l1_equivalence():
    curr = mx.array([1.0, 2.0, 3.0])
    prev = mx.array([0.9, 2.1, 3.05])
    assert legacy_gate.mean_abs_rel_l1(curr, prev) == kernel_gate.mean_abs_rel_l1(curr, prev)
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/_kernel/test_gate_equivalence.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_teacache._kernel.gate'`.

- [ ] **Step 4: Copy the gate module byte-for-byte**

```bash
cp src/mlx_teacache/gate.py src/mlx_teacache/_kernel/gate.py
```

Edit only the module-level docstring header in `_kernel/gate.py` to note "canonical home for the gate primitives (extracted in v0.6.0)". Leave every other line — every function body, every decision branch — exactly as it is in the legacy file.

- [ ] **Step 5: Convert the old path to a re-export shim**

```python
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
```

- [ ] **Step 6: Run equivalence + no-mflux guard tests**

```bash
uv run pytest tests/_kernel/test_gate_equivalence.py tests/_kernel/test_kernel_no_mflux_import.py -v
```

Expected: PASS. Existing `tests/test_gate.py` (v0.5.x tests) must also pass via the shim — run them too:

```bash
uv run pytest tests/test_gate.py -v
```

- [ ] **Step 7: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/gate.py src/mlx_teacache/_kernel/gate.py tests/_kernel/test_gate_equivalence.py
uv run mypy src/mlx_teacache/_kernel/gate.py
```

Expected: green.

- [ ] **Step 8: Commit**

```bash
git add src/mlx_teacache/gate.py src/mlx_teacache/_kernel/gate.py tests/_kernel/test_gate_equivalence.py
git commit -m "feat(_kernel/gate): verbatim extraction from src/mlx_teacache/gate.py (+ shim)"
```

---

### Task 3: `_kernel.cache` — verbatim extraction + shim + equivalence

**Files:**
- Create: `src/mlx_teacache/_kernel/cache.py` (byte-for-byte copy of v0.5.x `src/mlx_teacache/cache.py`)
- Modify: `src/mlx_teacache/cache.py` (becomes a re-export shim)
- Create: `tests/_kernel/test_cache_equivalence.py`

The v0.5.x `cache.py` defines `TeaCacheState` with the EXACT fields: `step_counter: int`, `previous_mod_input: mx.array | None`, `cached_residual: mx.array | None`, `cached_residual_neg: mx.array | None`, `accumulated_distance: float`, `last_timestep: float | None`, `skip_window_validated: bool`, `num_steps: int | None`. Plus `reset_for_new_generation(*, num_steps: int) -> None`. Preserve all fields and the reset signature.

- [ ] **Step 1: Inspect the source**

```bash
wc -l src/mlx_teacache/cache.py
grep -n "^class \|^def \|^    [a-z_]*:" src/mlx_teacache/cache.py
```

Expected: ~55 lines; fields enumerated above.

- [ ] **Step 2: Write equivalence test**

```python
# tests/_kernel/test_cache_equivalence.py
"""Phase A discipline: _kernel.cache.TeaCacheState is the same dataclass
as v0.5.x src/mlx_teacache/cache.TeaCacheState (post-shim)."""
from __future__ import annotations

import dataclasses

import mlx.core as mx


def test_field_set_matches_v05():
    from mlx_teacache._kernel.cache import TeaCacheState

    expected_field_names = {
        "step_counter", "previous_mod_input",
        "cached_residual", "cached_residual_neg",
        "accumulated_distance", "last_timestep",
        "skip_window_validated", "num_steps",
    }
    actual = {f.name for f in dataclasses.fields(TeaCacheState)}
    assert actual == expected_field_names


def test_reset_signature_takes_num_steps():
    from mlx_teacache._kernel.cache import TeaCacheState

    s = TeaCacheState()
    s.step_counter = 5
    s.accumulated_distance = 1.0
    s.cached_residual = mx.array([1.0])
    s.cached_residual_neg = mx.array([2.0])
    s.skip_window_validated = True
    s.num_steps = 8
    s.last_timestep = 0.5
    s.previous_mod_input = mx.array([0.1])

    s.reset_for_new_generation(num_steps=12)

    assert s.step_counter == 0
    assert s.previous_mod_input is None
    assert s.cached_residual is None
    assert s.cached_residual_neg is None
    assert s.accumulated_distance == 0.0
    assert s.last_timestep is None
    assert s.skip_window_validated is False
    assert s.num_steps == 12


def test_shim_re_exports_state_identity():
    """Old import path still works."""
    from mlx_teacache._kernel.cache import TeaCacheState as KernelState
    from mlx_teacache.cache import TeaCacheState as LegacyState
    assert LegacyState is KernelState
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/_kernel/test_cache_equivalence.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Copy `cache.py` byte-for-byte**

```bash
cp src/mlx_teacache/cache.py src/mlx_teacache/_kernel/cache.py
```

Edit only the module docstring.

- [ ] **Step 5: Convert the old path to a re-export shim**

```python
# src/mlx_teacache/cache.py
"""Compatibility shim. Canonical home is `mlx_teacache._kernel.cache`."""
from mlx_teacache._kernel.cache import TeaCacheState

__all__ = ["TeaCacheState"]
```

- [ ] **Step 6: Run equivalence + no-mflux tests + existing cache tests**

```bash
uv run pytest tests/_kernel/test_cache_equivalence.py tests/_kernel/test_kernel_no_mflux_import.py -v
uv run pytest tests/ -k "cache or state" -v
```

Expected: all pass.

- [ ] **Step 7: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/cache.py src/mlx_teacache/_kernel/cache.py
uv run mypy src/mlx_teacache/_kernel/cache.py
```

Expected: green.

- [ ] **Step 8: Commit**

```bash
git add src/mlx_teacache/cache.py src/mlx_teacache/_kernel/cache.py tests/_kernel/test_cache_equivalence.py
git commit -m "feat(_kernel/cache): verbatim extraction from src/mlx_teacache/cache.py (+ shim)"
```

---

### Task 4: `_kernel.stats` — verbatim extraction + shim + equivalence

**Files:**
- Create: `src/mlx_teacache/_kernel/stats.py` (byte-for-byte copy of v0.5.x `src/mlx_teacache/stats.py`)
- Modify: `src/mlx_teacache/stats.py` (becomes a re-export shim)
- Create: `tests/_kernel/test_stats_equivalence.py`

The v0.5.x `stats.py` defines: `StatsFrozenError`, `Decision` Literal (5 values including the deprecated `"cfg-fallback"`), `StepDecision` (frozen, fields: `step_idx, timestep, rel_l1, accumulated_distance, decision`), `GenerationStats` (frozen, fields: `num_steps, cfg_was_active, decisions`), `_Staging` (mutable), `TeaCacheStats` (mutable with `generations, computed_count, forced_count, skipped_count, numerical_miss_count, cfg_fallback_steps, last_generation, _staging, _frozen` fields; properties `total_active_steps, total_steps_seen, speedup_estimate`; methods `record, finalize_last_generation(num_inference_steps, cfg_was_active), discard_current_generation, _freeze`). Preserve every field name and the commit/discard semantics.

- [ ] **Step 1: Inspect the source**

```bash
wc -l src/mlx_teacache/stats.py
grep -n "^class \|^def \|^@dataclass\|^Decision" src/mlx_teacache/stats.py
```

Expected: ~172 lines.

- [ ] **Step 2: Write equivalence test**

```python
# tests/_kernel/test_stats_equivalence.py
"""Phase A discipline: _kernel.stats preserves v0.5.x stats contract.
Field names, decision values, commit/discard semantics."""
from __future__ import annotations


def test_step_decision_fields_match_v05():
    import dataclasses

    from mlx_teacache._kernel.stats import StepDecision

    actual = {f.name for f in dataclasses.fields(StepDecision)}
    assert actual == {"step_idx", "timestep", "rel_l1", "accumulated_distance", "decision"}


def test_generation_stats_fields_match_v05():
    import dataclasses

    from mlx_teacache._kernel.stats import GenerationStats

    actual = {f.name for f in dataclasses.fields(GenerationStats)}
    assert actual == {"num_steps", "cfg_was_active", "decisions"}


def test_decision_literal_includes_cfg_fallback():
    """v0.4.1 deprecated cfg-fallback but kept the Literal value; v0.6.0 preserves it."""
    import typing

    from mlx_teacache._kernel.stats import Decision

    assert "cfg-fallback" in typing.get_args(Decision)


def test_teacachestats_public_counter_fields():
    from mlx_teacache._kernel.stats import TeaCacheStats

    s = TeaCacheStats()
    assert s.generations == 0
    assert s.computed_count == 0
    assert s.forced_count == 0
    assert s.skipped_count == 0
    assert s.numerical_miss_count == 0
    assert s.cfg_fallback_steps == 0
    assert s.last_generation is None
    assert s.speedup_estimate == 1.0


def test_failed_generation_leaves_no_public_trace():
    """The commit/discard contract: record() touches staging only;
    discard_current_generation() drops staging; public counters
    are unchanged."""
    from mlx_teacache._kernel.stats import StepDecision, TeaCacheStats

    s = TeaCacheStats()
    s.record(StepDecision(step_idx=0, timestep=1.0, rel_l1=None,
                          accumulated_distance=0.0, decision="computed"))
    s.discard_current_generation()
    assert s.computed_count == 0
    assert s.generations == 0
    assert s.last_generation is None


def test_finalize_commits_to_public_counters():
    from mlx_teacache._kernel.stats import StepDecision, TeaCacheStats

    s = TeaCacheStats()
    for i in range(4):
        s.record(StepDecision(step_idx=i, timestep=float(i), rel_l1=None,
                              accumulated_distance=0.0, decision="computed"))
    s.finalize_last_generation(num_inference_steps=4, cfg_was_active=False)
    assert s.computed_count == 4
    assert s.generations == 1
    assert s.last_generation is not None
    assert s.last_generation.num_steps == 4
    assert s.last_generation.cfg_was_active is False


def test_shim_re_exports_identity():
    from mlx_teacache._kernel.stats import TeaCacheStats as KS
    from mlx_teacache.stats import TeaCacheStats as LS
    assert LS is KS
    from mlx_teacache._kernel.stats import StatsFrozenError as KSE
    from mlx_teacache.stats import StatsFrozenError as LSE
    assert LSE is KSE
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/_kernel/test_stats_equivalence.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Copy `stats.py` byte-for-byte**

```bash
cp src/mlx_teacache/stats.py src/mlx_teacache/_kernel/stats.py
```

Edit only the module docstring header.

- [ ] **Step 5: Convert the old path to a re-export shim**

```python
# src/mlx_teacache/stats.py
"""Compatibility shim. Canonical home is `mlx_teacache._kernel.stats`."""
from mlx_teacache._kernel.stats import (
    Decision,
    GenerationStats,
    StatsFrozenError,
    StepDecision,
    TeaCacheStats,
)

__all__ = ["Decision", "GenerationStats", "StatsFrozenError", "StepDecision", "TeaCacheStats"]
```

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/_kernel/test_stats_equivalence.py tests/_kernel/test_kernel_no_mflux_import.py -v
uv run pytest tests/ -k "stats" -v
```

Expected: all pass.

- [ ] **Step 7: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/stats.py src/mlx_teacache/_kernel/stats.py
uv run mypy src/mlx_teacache/_kernel/stats.py
```

Expected: green.

- [ ] **Step 8: Commit**

```bash
git add src/mlx_teacache/stats.py src/mlx_teacache/_kernel/stats.py tests/_kernel/test_stats_equivalence.py
git commit -m "feat(_kernel/stats): verbatim extraction preserving v0.5.x commit/discard contract"
```

---

### Task 5: `_kernel.coefficients` — `Provenance` only (registry stays in variants)

**Files:**
- Create: `src/mlx_teacache/_kernel/coefficients.py` (holds `Provenance` dataclass only — the legacy `_REGISTRY` dict is NOT moved; it becomes per-variant)
- Modify: `src/mlx_teacache/coefficients.py` (re-exports `Provenance` + keeps legacy `_REGISTRY` for now; legacy registry deleted after all variants have config.py)
- Create: `tests/_kernel/test_coefficients_provenance.py`

Read `src/mlx_teacache/coefficients.py` first. It contains:
- 4 coefficient tuples: `_UPSTREAM_FLUX_COEFFS`, `_FLUX2_KLEIN_4B_COEFFS`, `_FLUX2_KLEIN_9B_COEFFS`, `_FLUX2_KLEIN_BASE_4B_COEFFS`
- `Provenance` dataclass
- `_REGISTRY` dict mapping variant_id → (coefficients, Provenance)
- `load_builtin(variant_id)`, `validate_custom(coeffs)` functions

In v0.6.0, the registry moves to per-variant `config.py` files (one variant per file, no central registry). `Provenance` stays as a shared type in `_kernel/`. The coefficient tuples stay in `coefficients.py` until Task 12-17 migrate them into per-variant `config.py` files; the tuples are NOT deleted until ALL variants reference them from their own config.

- [ ] **Step 1: Inspect the source**

```bash
grep -n "^def \|^class \|^_[A-Z]" src/mlx_teacache/coefficients.py
```

- [ ] **Step 2: Write the failing test**

```python
# tests/_kernel/test_coefficients_provenance.py
def test_provenance_is_accessible_from_kernel():
    from mlx_teacache._kernel.coefficients import Provenance
    p = Provenance.for_user_supplied()
    assert p.source == "user"


def test_shim_re_exports_provenance():
    from mlx_teacache._kernel.coefficients import Provenance as KP
    from mlx_teacache.coefficients import Provenance as LP
    assert LP is KP
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/_kernel/test_coefficients_provenance.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Create `_kernel/coefficients.py`**

```python
# src/mlx_teacache/_kernel/coefficients.py
"""Provenance dataclass. Lives here because it is a pure data type
shared across variants.

Coefficient tuples themselves live in per-variant config.py files
under src/mlx_teacache/variants/<name>/. The legacy _REGISTRY mapping
in src/mlx_teacache/coefficients.py is deleted after all variants
have migrated."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Provenance:
    """COPY VERBATIM from src/mlx_teacache/coefficients.py::Provenance.
    Read the legacy module first and reproduce the field list, defaults,
    and the classmethod `for_user_supplied()` exactly."""
    # Worker: copy the @dataclass body from src/mlx_teacache/coefficients.py.
```

The worker reads `src/mlx_teacache/coefficients.py`, finds the `Provenance` dataclass body, and copies it into `_kernel/coefficients.py` byte-for-byte. Field names, defaults, methods preserved.

- [ ] **Step 5: Modify the old `coefficients.py` to re-export `Provenance` (registry kept for now)**

Edit `src/mlx_teacache/coefficients.py`:
- Replace the `Provenance` definition with `from mlx_teacache._kernel.coefficients import Provenance`.
- Keep the four coefficient tuples (`_UPSTREAM_FLUX_COEFFS`, etc.) unchanged.
- Keep `_REGISTRY`, `load_builtin`, `validate_custom` unchanged — they will be removed in Task 18 after every variant has its own config.py.

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/_kernel/test_coefficients_provenance.py tests/_kernel/test_kernel_no_mflux_import.py -v
uv run pytest tests/test_coefficients.py -v
```

Expected: all pass.

- [ ] **Step 7: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/coefficients.py src/mlx_teacache/_kernel/coefficients.py
uv run mypy src/mlx_teacache/_kernel/coefficients.py
```

Expected: green.

- [ ] **Step 8: Commit**

```bash
git add src/mlx_teacache/coefficients.py src/mlx_teacache/_kernel/coefficients.py tests/_kernel/test_coefficients_provenance.py
git commit -m "feat(_kernel/coefficients): Provenance moved; legacy registry kept temporarily"
```

---

## Phase B — `TeaCacheHandle` + `VariantPatch` contract

Goal: introduce the variant-agnostic handle + patch contract. Stats commit/discard stays in the mflux lifecycle (unchanged); the handle only owns mutation rollback + callback unsubscribe.

### Task 6: `mlx_teacache.handle` — `TeaCacheHandle` + `VariantPatch`

**Files:**
- Create: `src/mlx_teacache/handle.py`
- Create: `tests/test_handle.py`

Read `src/mlx_teacache/api.py` first — today's `TeaCacheHandle` is defined inline in `api.py` (around line 39) with `_HandleState`. The new module extracts that class as-is (preserving `.stats`, `.provenance`, `.rel_l1_thresh`, `.restore()`), and adds `VariantPatch` as the teardown contract. Stats finalization is NOT moved into `restore()`; it stays in the mflux lifecycle wrapper.

- [ ] **Step 1: Inspect the current handle**

```bash
sed -n '30,110p' src/mlx_teacache/api.py
```

Read what `_HandleState` carries and what `TeaCacheHandle.restore()` does today. The new handle in `handle.py` must preserve:
- The same public attributes (at minimum `.stats`, `.provenance`, `.rel_l1_thresh`).
- The same context-manager protocol.
- `.restore()` runs the per-variant teardown.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_handle.py
"""TeaCacheHandle + VariantPatch contract.

Critical property (audit F2/F3): handle does NOT call stats finalize.
Stats commit/discard stays in the mflux lifecycle wrapper.
"""
from __future__ import annotations

import pytest


def test_handle_runs_rollbacks_in_reverse_install_order():
    from mlx_teacache._kernel.stats import TeaCacheStats
    from mlx_teacache.handle import TeaCacheHandle, VariantPatch

    log: list[str] = []
    patch = VariantPatch(
        rollbacks=[lambda: log.append("r1"), lambda: log.append("r2")],
        finalizers=[],
    )
    stats = TeaCacheStats()
    h = TeaCacheHandle(patch=patch, stats=stats,
                      provenance=_dummy_provenance(), rel_l1_thresh=0.2)
    h.restore()
    assert log == ["r2", "r1"]


def test_handle_restore_is_idempotent():
    from mlx_teacache._kernel.stats import TeaCacheStats
    from mlx_teacache.handle import TeaCacheHandle, VariantPatch

    counter = {"n": 0}
    patch = VariantPatch(rollbacks=[lambda: counter.update(n=counter["n"] + 1)], finalizers=[])
    h = TeaCacheHandle(patch=patch, stats=TeaCacheStats(),
                      provenance=_dummy_provenance(), rel_l1_thresh=0.2)
    h.restore()
    h.restore()
    assert counter["n"] == 1


def test_handle_does_not_finalize_stats():
    """Audit F2: stats commit stays in mflux lifecycle. Handle restore must
    not call finalize_last_generation; that's the lifecycle's job in
    integrations/mflux/lifecycle.py:wrap_generate_image."""
    from mlx_teacache._kernel.stats import StepDecision, TeaCacheStats
    from mlx_teacache.handle import TeaCacheHandle, VariantPatch

    stats = TeaCacheStats()
    stats.record(StepDecision(step_idx=0, timestep=1.0, rel_l1=None,
                              accumulated_distance=0.0, decision="computed"))
    # Staging has 1 entry; public counters still 0.
    h = TeaCacheHandle(patch=VariantPatch(), stats=stats,
                      provenance=_dummy_provenance(), rel_l1_thresh=0.2)
    h.restore()
    # Public counters unchanged — restore did NOT commit.
    assert stats.computed_count == 0
    assert stats.generations == 0


def test_handle_has_no_variant_branches():
    """Audit F3: TeaCacheHandle is variant-agnostic. Static-grep check."""
    import inspect

    from mlx_teacache import handle as handle_module

    source = inspect.getsource(handle_module)
    code = "\n".join(ln for ln in source.splitlines()
                     if not ln.lstrip().startswith("#")).lower()
    for bad in ("flux1", "flux2", "klein"):
        assert bad not in code, f"handle.py must not mention {bad!r}"


def _dummy_provenance():
    from mlx_teacache._kernel.coefficients import Provenance
    return Provenance(source="builtin")
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/test_handle.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_teacache.handle'`.

- [ ] **Step 4: Implement `handle.py`**

```python
# src/mlx_teacache/handle.py
"""Variant-agnostic context-manager handle.

`apply_teacache(flux)` returns a `TeaCacheHandle`. Variants build the
handle in their `apply()` and pass a `VariantPatch` describing how to
undo their mutations + unsubscribe their callbacks. The handle does NOT
own stats finalization — that's the mflux lifecycle wrapper's job (see
src/mlx_teacache/integrations/mflux/lifecycle.py::wrap_generate_image).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.stats import TeaCacheStats


@dataclass
class VariantPatch:
    """Teardown contract returned by each variant's apply().

    rollbacks: undo callables in install order (handle runs them in reverse).
    finalizers: callables that run after rollbacks (e.g., callback unsubscribe).

    Stats commit/discard is NOT in this list. The mflux lifecycle owns that.
    """
    rollbacks: list[Callable[[], None]] = field(default_factory=list)
    finalizers: list[Callable[[], None]] = field(default_factory=list)


class TeaCacheHandle:
    """Context-manager handle returned by apply_teacache."""

    def __init__(
        self,
        *,
        patch: VariantPatch,
        stats: TeaCacheStats,
        provenance: Provenance,
        rel_l1_thresh: float,
    ) -> None:
        self._patch = patch
        self.stats = stats
        self.provenance = provenance
        self.rel_l1_thresh = rel_l1_thresh
        self._torn_down = False

    def __enter__(self) -> TeaCacheHandle:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.restore()

    def restore(self) -> None:
        if self._torn_down:
            return
        for rollback in reversed(self._patch.rollbacks):
            rollback()
        for finalize in self._patch.finalizers:
            finalize()
        # Freeze stats — same contract as v0.5.x TeaCacheHandle.restore().
        if hasattr(self.stats, "_freeze"):
            self.stats._freeze()
        self._torn_down = True
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_handle.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/handle.py tests/test_handle.py
uv run mypy src/mlx_teacache/handle.py
```

Expected: green.

- [ ] **Step 7: Commit**

```bash
git add src/mlx_teacache/handle.py tests/test_handle.py
git commit -m "feat(handle): TeaCacheHandle + VariantPatch (variant-agnostic, no stats finalize)"
```

---

## Phase C — Variant scaffold + per-variant cores

### Task 7: `variants/` registry (metadata + matches only; integration lazy)

**Files:**
- Create: `src/mlx_teacache/variants/__init__.py`
- Create: `tests/variants/__init__.py`
- Create: `tests/variants/test_registry.py`

- [ ] **Step 1: Write failing test**

```python
# tests/variants/test_registry.py
"""Registry walks variants/ at import time. config.py + detect.py load
eagerly. integration.py is lazy (loaded on first dispatch). The walker
must be mflux-free at import time."""
from __future__ import annotations


def test_registry_is_a_mapping():
    from mlx_teacache.variants import _REGISTRY
    assert isinstance(_REGISTRY, dict)


def test_registry_entries_have_required_shape():
    from mlx_teacache.variants import _REGISTRY
    for entry in _REGISTRY.values():
        assert "META" in entry
        assert "matches" in entry
        assert "load_integration" in entry
        assert callable(entry["matches"])
        assert callable(entry["load_integration"])


def test_registry_keys_match_meta_variant_ids():
    from mlx_teacache.variants import _REGISTRY
    for variant_id, entry in _REGISTRY.items():
        assert entry["META"]["variant_id"] == variant_id
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/variants/test_registry.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the walker**

```python
# src/mlx_teacache/variants/__init__.py
"""Variant registry. Walks every subpackage of variants/ at import time
to populate _REGISTRY with (META, matches, load_integration) entries.

config.py + detect.py are imported eagerly (they must be mflux-free).
integration.py is loaded lazily via load_integration() — apply_teacache
calls it after detect picks the winning variant. This is the contract
that keeps `import mlx_teacache` working without the [mflux] extra.
"""
from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any, TypedDict


class _RegistryEntry(TypedDict):
    META: dict[str, Any]
    matches: Callable[[object], bool]
    load_integration: Callable[[], Callable[..., Any]]


_REGISTRY: dict[str, _RegistryEntry] = {}


def _make_lazy_loader(module_name: str) -> Callable[[], Callable[..., Any]]:
    def _load() -> Callable[..., Any]:
        integration = importlib.import_module(f"{module_name}.integration")
        return integration.apply
    return _load


def _build_registry() -> None:
    package = importlib.import_module(__name__)
    for _, subname, ispkg in pkgutil.iter_modules(package.__path__):
        if not ispkg:
            continue
        full = f"{__name__}.{subname}"
        config = importlib.import_module(f"{full}.config")
        detect = importlib.import_module(f"{full}.detect")
        meta: dict[str, Any] = config.META
        variant_id = meta["variant_id"]
        _REGISTRY[variant_id] = _RegistryEntry(
            META=meta,
            matches=detect.matches,
            load_integration=_make_lazy_loader(full),
        )


_build_registry()

__all__ = ["_REGISTRY"]
```

```python
# tests/variants/__init__.py
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/variants/test_registry.py -v
```

Expected: 3 passed (registry is empty — no variants yet).

- [ ] **Step 5: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/variants/__init__.py tests/variants/
uv run mypy src/mlx_teacache/variants/__init__.py
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/mlx_teacache/variants/ tests/variants/
git commit -m "feat(variants): registry walker (config + detect eager, integration lazy)"
```

---

### Task 8: `variants/flux1_dev/` — config + detect (mflux-free) + verbatim integration port

**Files:**
- Create: `src/mlx_teacache/variants/flux1_dev/{__init__,config,detect,integration}.py`
- Create: `tests/variants/flux1_dev/{__init__,test_detect,test_integration_smoke}.py`

Read first:
- `src/mlx_teacache/coefficients.py` — note the `_UPSTREAM_FLUX_COEFFS` tuple. The worker COPIES this tuple into `flux1_dev/config.py::COEFFICIENTS` (no literal in this plan).
- `src/mlx_teacache/integrations/mflux/flux1.py` — `ProxyFlux1Transformer`. Copied byte-for-byte into `flux1_dev/integration.py`.
- `src/mlx_teacache/integrations/mflux/forward.py` lines 30-260ish — `_step_decision_from_gate`, `_flux1_extract_mod_input`, `_flux1_run_body`, `flux1_forward_with_gate`. Copied byte-for-byte.
- `src/mlx_teacache/integrations/mflux/lifecycle.py` — STAYS shared; the integration imports `GenerationContextCallback` and `wrap_generate_image` from it.
- `src/mlx_teacache/api.py` lines 138-315 — how the v0.5.x facade resolves coefficients, builds the handle, attaches callbacks. The variant's `apply()` mirrors that logic for FLUX.1 dev specifically.

- [ ] **Step 1: Write tests**

```python
# tests/variants/flux1_dev/test_detect.py
from __future__ import annotations


class _FC:
    def __init__(self, aliases: list[str]) -> None:
        self.aliases = aliases
        self.model_name = "fake/flux1-dev"


class _FakeFlux1:
    def __init__(self, aliases: list[str]) -> None:
        self.model_config = _FC(aliases)


def test_meta_variant_id():
    from mlx_teacache.variants.flux1_dev.config import META
    assert META["variant_id"] == "flux1-dev"
    assert META["non_distilled"] is True


def test_coefficients_match_v05_registry():
    """Audit F4 guard: the variant's COEFFICIENTS must equal the v0.5.x
    registry entry. This catches transcription errors before the legacy
    registry is removed in Task 18."""
    from mlx_teacache.coefficients import _UPSTREAM_FLUX_COEFFS
    from mlx_teacache.variants.flux1_dev.config import COEFFICIENTS
    assert COEFFICIENTS == _UPSTREAM_FLUX_COEFFS


def test_matches_dev_alias():
    from mlx_teacache.variants.flux1_dev.detect import matches
    assert matches(_FakeFlux1(["dev"])) is True


def test_does_not_match_schnell():
    from mlx_teacache.variants.flux1_dev.detect import matches
    assert matches(_FakeFlux1(["schnell"])) is False
```

```python
# tests/variants/flux1_dev/test_integration_smoke.py
"""Real-weight smoke test for flux1-dev integration. Skipped without
mflux installed."""
import pytest


def test_apply_returns_handle_and_restores_pristine():
    pytest.importorskip("mflux")
    from mflux.models.flux.variants.txt2img.flux import Flux1

    flux = Flux1.from_name("dev", quantize=4)
    flux.freeze()
    transformer_before = flux.transformer

    from mlx_teacache.handle import TeaCacheHandle
    from mlx_teacache.variants.flux1_dev.integration import apply

    handle = apply(flux)
    assert isinstance(handle, TeaCacheHandle)
    # transformer was wrapped during apply
    assert flux.transformer is not transformer_before
    handle.restore()
    # original transformer is back
    assert flux.transformer is transformer_before
```

```python
# tests/variants/flux1_dev/__init__.py
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/variants/flux1_dev/test_detect.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `__init__.py`, `config.py`, `detect.py`**

```python
# src/mlx_teacache/variants/flux1_dev/__init__.py
from .config import META
from .detect import matches
__all__ = ["META", "matches"]
```

```python
# src/mlx_teacache/variants/flux1_dev/config.py
"""FLUX.1 dev configuration. mflux-free."""
from __future__ import annotations

from typing import Any

# Worker: COPY the tuple from src/mlx_teacache/coefficients.py::_UPSTREAM_FLUX_COEFFS
# Do not type out the values from memory or from this plan. Open the source
# and copy line by line.
from mlx_teacache.coefficients import _UPSTREAM_FLUX_COEFFS as _LEGACY_COEFFS
COEFFICIENTS: tuple[float, float, float, float, float] = _LEGACY_COEFFS

DEFAULT_THRESH: float = 0.20

RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 25, "guidance": 3.5},
}

LICENSE: str = "FLUX.1-dev Non-Commercial License"

META: dict[str, Any] = {
    "variant_id": "flux1-dev",
    "display_name": "FLUX.1 dev",
    "hf_model_id": "black-forest-labs/FLUX.1-dev",
    "non_distilled": True,
    "memory_cap_hint_gb": None,
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/black-forest-labs/FLUX.1-dev",
}
```

Note: the variant's `config.py` imports `_UPSTREAM_FLUX_COEFFS` from the legacy `coefficients.py` for the v0.6.0 transition only. After Task 18 (legacy-registry deletion) the variant inlines the tuple as a module-level constant. The transitional import keeps the variant's coefficients identity-equal to the legacy registry, which is what `test_coefficients_match_v05_registry` checks.

```python
# src/mlx_teacache/variants/flux1_dev/detect.py
"""mflux-free detector."""
from __future__ import annotations


def matches(flux: object) -> bool:
    model_config = getattr(flux, "model_config", None)
    if model_config is None:
        return False
    aliases = getattr(model_config, "aliases", None) or []
    if "dev" not in aliases:
        return False
    return type(flux).__name__ == "Flux1"
```

- [ ] **Step 4: Run detect + config tests**

```bash
uv run pytest tests/variants/flux1_dev/test_detect.py tests/variants/test_registry.py -v
```

Expected: all pass.

- [ ] **Step 5: Implement `integration.py` by byte-for-byte port**

The integration port for flux1-dev contains:

1. A copy of `ProxyFlux1Transformer` from `src/mlx_teacache/integrations/mflux/flux1.py` (81 lines). Copy the class verbatim. Update the imports: the proxy currently imports `gate_step` from `mlx_teacache.gate` and `TeaCacheState` from `mlx_teacache.cache` — those still work via the shims, but for cleanliness update to import from `_kernel`.

2. A copy of the FLUX.1 forward function block from `src/mlx_teacache/integrations/mflux/forward.py` lines 30-260ish (the `_step_decision_from_gate`, `_flux1_extract_mod_input`, `_flux1_run_body`, `flux1_forward_with_gate` group). Copy verbatim.

3. An `apply()` function that mirrors the v0.5.x FLUX.1 path in `src/mlx_teacache/api.py::apply_teacache`:
   - Resolve `rel_l1_thresh` (caller > variant default > package fallback 0.20).
   - Resolve `coefficients` (caller > variant COEFFICIENTS).
   - Build `TeaCacheState`, `TeaCacheStats`.
   - Construct `ProxyFlux1Transformer`, swap onto `flux.transformer`.
   - Register `GenerationContextCallback` via `wrap_generate_image` (imported from `mlx_teacache.integrations.mflux.lifecycle`).
   - Build the `VariantPatch` with rollback callbacks that (a) restore `flux.transformer`, (b) unsubscribe the callback. NO stats finalize in the patch.
   - Return `TeaCacheHandle(patch=patch, stats=stats, provenance=PROVENANCE, rel_l1_thresh=resolved_thresh)`.

```python
# src/mlx_teacache/variants/flux1_dev/integration.py
"""FLUX.1 dev integration. Byte-for-byte port from v0.5.x:
- src/mlx_teacache/integrations/mflux/flux1.py::ProxyFlux1Transformer
- src/mlx_teacache/integrations/mflux/forward.py FLUX.1 forward block
- src/mlx_teacache/api.py::apply_teacache FLUX.1 branch

mflux is imported only inside this module. The package registry loads
this lazily, after detect.matches() wins.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_teacache._kernel.cache import TeaCacheState
from mlx_teacache._kernel.coefficients import Provenance
from mlx_teacache._kernel.gate import GateDecision, gate_step
from mlx_teacache._kernel.stats import StepDecision, TeaCacheStats
from mlx_teacache.handle import TeaCacheHandle, VariantPatch
from mlx_teacache.integrations.mflux.lifecycle import wrap_generate_image  # shared

from .config import COEFFICIENTS, DEFAULT_THRESH, META


_PROVENANCE = Provenance(
    source="builtin",
    revision="upstream-flux-v1",
    calibration_dataset="upstream ali-vilab TeaCache (no in-repo calibration)",
    reference_url="https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py",
)


# ----- PORTED VERBATIM from src/mlx_teacache/integrations/mflux/flux1.py -----
# Worker: open that file, copy `ProxyFlux1Transformer` (and any helpers it
# depends on). Adjust import statements to reference `_kernel.gate`,
# `_kernel.cache`, `_kernel.stats` directly. Do not change logic.

# ----- PORTED VERBATIM from src/mlx_teacache/integrations/mflux/forward.py -----
# Worker: copy the FLUX.1 group:
#   _step_decision_from_gate
#   _flux1_extract_mod_input
#   _flux1_run_body
#   flux1_forward_with_gate
# These are the v0.5.x algorithm. Do not invent new boundaries.


def apply(
    flux: Any,
    *,
    rel_l1_thresh: float | None = None,
    coefficients: tuple[float, float, float, float, float] | None = None,
    skip_first_n_steps: int = 1,
    skip_last_n_steps: int = 1,
) -> TeaCacheHandle:
    """FLUX.1 dev apply. Public-API-equivalent of the FLUX.1 branch of
    v0.5.x apply_teacache."""
    # Worker: copy the resolution logic from src/mlx_teacache/api.py:
    #   1. Resolve rel_l1_thresh (caller > DEFAULT_THRESH).
    #   2. Resolve coefficients (caller > COEFFICIENTS).
    #   3. Validate the skip-window (existing validation in api.py).
    #   4. Build TeaCacheState, TeaCacheStats.
    #   5. Build the ProxyFlux1Transformer; swap flux.transformer.
    #   6. Call wrap_generate_image(flux, handle) per v0.5.x lifecycle.
    #   7. Build VariantPatch: rollback restores transformer + unsubscribes
    #      the GenerationContextCallback. NO stats finalize call.
    #   8. Return TeaCacheHandle(...).
    raise NotImplementedError(
        "PORT: build apply() from src/mlx_teacache/api.py::apply_teacache "
        "FLUX.1 branch + src/mlx_teacache/integrations/mflux/lifecycle.py."
    )
```

The worker is expected to read `api.py` and `lifecycle.py`, then assemble `apply()` with the exact same semantics. The skeleton above shows the structure; the actual body is a faithful translation of the v0.5.x logic.

- [ ] **Step 6: Run integration smoke test (with mflux installed)**

```bash
uv run pytest tests/variants/flux1_dev/test_integration_smoke.py -v
```

Expected: passes if mflux is installed; otherwise skipped.

- [ ] **Step 7: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/variants/flux1_dev/
uv run mypy src/mlx_teacache/variants/flux1_dev/__init__.py \
              src/mlx_teacache/variants/flux1_dev/config.py \
              src/mlx_teacache/variants/flux1_dev/detect.py \
              src/mlx_teacache/variants/flux1_dev/integration.py
```

Expected: green.

- [ ] **Step 8: Commit**

```bash
git add src/mlx_teacache/variants/flux1_dev/ tests/variants/flux1_dev/
git commit -m "feat(variants/flux1_dev): config + detect + verbatim integration port"
```

---

### Task 9: `variants/flux1_schnell/`

Follows the flux1_dev pattern. Differences:
- META: `variant_id = "flux1-schnell"`, `display_name = "FLUX.1 schnell"`, `hf_model_id = "black-forest-labs/FLUX.1-schnell"`, `non_distilled = False`, `license = "Apache-2.0"`.
- RECIPES default: 4 steps, guidance=1.0.
- `COEFFICIENTS` cross-imported from `flux1_dev/config.py` (same FLUX.1 architecture; the v0.5.x registry has the schnell entry pointing at the same tuple).
- `detect.matches` checks `"schnell" in aliases`.
- `integration.py` ports the same FLUX.1 forward code (the same `flux1_forward_with_gate` and `ProxyFlux1Transformer`). Worker copies verbatim from `flux1_dev/integration.py` and adjusts the `_PROVENANCE.revision` to `"upstream-flux-v1-shared"`.

- [ ] **Step 1: Write tests**

```python
# tests/variants/flux1_schnell/test_detect.py
def test_meta():
    from mlx_teacache.variants.flux1_schnell.config import META
    assert META["variant_id"] == "flux1-schnell"
    assert META["non_distilled"] is False
    assert META["recipes"]["default"]["num_inference_steps"] == 4


def test_coefficients_shared_with_dev():
    from mlx_teacache.variants.flux1_dev.config import COEFFICIENTS as DEV
    from mlx_teacache.variants.flux1_schnell.config import COEFFICIENTS as SCHNELL
    assert SCHNELL is DEV


def test_matches_schnell():
    from mlx_teacache.variants.flux1_schnell.detect import matches

    class _FC:
        def __init__(self, a):
            self.aliases = a
            self.model_name = "fake/x"

    class _FakeFlux1:
        def __init__(self, a):
            self.model_config = _FC(a)
    assert matches(_FakeFlux1(["schnell"])) is True
    assert matches(_FakeFlux1(["dev"])) is False
```

- [ ] **Step 2: Run test (fails)**, **Step 3: Implement**, **Step 4: Run test (passes)**, **Step 5: Lint**, **Step 6: Commit**.

(Steps follow the Task 8 pattern; full step-by-step structure is the same.)

```bash
git add src/mlx_teacache/variants/flux1_schnell/ tests/variants/flux1_schnell/
git commit -m "feat(variants/flux1_schnell): no-CFG variant, coefficient cross-import from dev"
```

---

### Task 10: `variants/flux2_klein_4b/`

Distilled FLUX.2 Klein 4B. Mirror the flux2_klein_base_4b pattern (Task 12 below) for the integration port — FLUX.2 family integration uses `flux2_forward_with_gate` (the non-CFG forward, since klein-4b's default schedule is g=1.0).

Differences from flux2_klein_base_4b:
- META: `variant_id = "flux2-klein-4b"`, `non_distilled = False`, `license = "Apache-2.0"`, `hf_model_id = "black-forest-labs/FLUX.2-klein-4B"`.
- RECIPES default: 8 steps, guidance=1.0 (distilled).
- `COEFFICIENTS` copied from `_FLUX2_KLEIN_4B_COEFFS` via transitional import.
- `DEFAULT_THRESH = None` (gate doesn't engage on 8-step distilled; package fallback 0.20 used).
- `integration.py` ports `flux2_forward_with_gate` from `src/mlx_teacache/integrations/mflux/forward.py`. NOT `flux2_cfg_forward_with_gate` — distilled klein-4b at g=1.0 doesn't use CFG.

Test, implement, lint, commit per the Task 8/9 pattern.

```bash
git add src/mlx_teacache/variants/flux2_klein_4b/ tests/variants/flux2_klein_4b/
git commit -m "feat(variants/flux2_klein_4b): distilled FLUX.2 Klein 4B"
```

---

### Task 11: `variants/flux2_klein_9b/`

Distilled FLUX.2 Klein 9B. Mirror Task 10. Differences:
- META: `variant_id = "flux2-klein-9b"`, `license = "FLUX Non-Commercial"`, `hf_model_id = "black-forest-labs/FLUX.2-klein-9B"`.
- `COEFFICIENTS` copied from `_FLUX2_KLEIN_9B_COEFFS` via transitional import.
- `integration.py` uses `flux2_forward_with_gate` (non-CFG; klein-9b default is g=1.0 distilled).

```bash
git add src/mlx_teacache/variants/flux2_klein_9b/ tests/variants/flux2_klein_9b/
git commit -m "feat(variants/flux2_klein_9b): distilled FLUX.2 Klein 9B"
```

---

### Task 12: `variants/flux2_klein_base_4b/`

Non-distilled FLUX.2 Klein base 4B. The canonical FLUX.2 CFG variant. Uses `flux2_cfg_forward_with_gate` from `forward.py` (CFG per-branch, v0.4.1).

- META: `variant_id = "flux2-klein-base-4b"`, `non_distilled = True`, `license = "Apache-2.0"`, `hf_model_id = "black-forest-labs/FLUX.2-klein-base-4B"`.
- RECIPES: `default = {"num_inference_steps": 50, "guidance": 4.0}` (canonical CFG); `low_step = {"num_inference_steps": 25, "guidance": 1.0}` (v0.4.0 row).
- `COEFFICIENTS` copied from `_FLUX2_KLEIN_BASE_4B_COEFFS` via transitional import.
- `DEFAULT_THRESH = 0.17` (per v0.4.0 sweep).
- `integration.py` ports BOTH `flux2_forward_with_gate` (used at g=1.0) AND `flux2_cfg_forward_with_gate` (used at g>1.0) plus the helpers (`_flux2_compute_mod_in`, `_flux2_extract_mod_input`, `_flux2_run_body`, `_flux2_apply_tail_and_combine`). The apply() function selects the forward based on `guidance` at attach time (mirroring v0.5.x logic in `src/mlx_teacache/integrations/mflux/flux2.py::make_teacache_predict_factory`).

Test, implement, lint, commit.

```bash
git add src/mlx_teacache/variants/flux2_klein_base_4b/ tests/variants/flux2_klein_base_4b/
git commit -m "feat(variants/flux2_klein_base_4b): non-distilled, ports both no-CFG + CFG paths"
```

---

### Task 13: `variants/flux2_klein_base_9b/` (coefficient cross-import)

Mirrors Task 12. Differences:
- `variant_id = "flux2-klein-base-9b"`, `license = "FLUX Non-Commercial"`, `hf_model_id = "black-forest-labs/FLUX.2-klein-base-9B"`.
- `memory_cap_hint_gb = 24` (32 GB unified memory headroom).
- `COEFFICIENTS` cross-imported from `flux2_klein_base_4b/config.py` (the intentional reuse pattern; v0.5.0 validated).
- `_PROVENANCE.revision = "in-repo-2026-05-18-reuse-base-4b"`.

Identity test:

```python
# tests/variants/flux2_klein_base_9b/test_shared_coefficients.py
def test_klein_base_9b_reuses_base_4b_coefficients():
    """v0.5.0 validated this reuse (SSIM 0.986). The test catches any
    accidental drift from the intentional reuse."""
    from mlx_teacache.variants.flux2_klein_base_4b.config import COEFFICIENTS as BASE_4B
    from mlx_teacache.variants.flux2_klein_base_9b.config import COEFFICIENTS as BASE_9B
    assert BASE_9B is BASE_4B
```

```bash
git add src/mlx_teacache/variants/flux2_klein_base_9b/ tests/variants/flux2_klein_base_9b/
git commit -m "feat(variants/flux2_klein_base_9b): cross-imports base-4b coefficients (v0.5.0 reuse pattern)"
```

---

### Task 14: Kernel-boundary validation gate (between first two variants)

After Tasks 8 and 12 are done — flux1_dev (no-CFG) + flux2_klein_base_4b (CFG per-branch) — the kernel boundary is exercised by both code paths. Pause and validate before porting the remaining four.

- [ ] **Step 1: Run all variant integration smoke tests**

```bash
uv run pytest tests/variants/flux1_dev/ tests/variants/flux2_klein_base_4b/ -v
```

Expected: all pass with mflux installed.

- [ ] **Step 2: Run the kernel test suite + equivalence tests**

```bash
uv run pytest tests/_kernel/ -v
```

Expected: all pass.

- [ ] **Step 3: Verify no duplicate algorithmic code in variants**

```bash
grep -n "def gate_step\|def poly_eval\|def mean_abs_rel_l1" \
    src/mlx_teacache/variants/flux1_dev/integration.py \
    src/mlx_teacache/variants/flux2_klein_base_4b/integration.py
```

Expected: ZERO matches. Kernel functions must not be redefined in variant code.

- [ ] **Step 4: If the boundary feels wrong, pause and re-cut**

Signals that re-cutting is needed:
- A variant imports from another variant's directory.
- The kernel has variant-specific branches.
- A shared concept (e.g., CFG combination) is duplicated across variants because the kernel doesn't expose enough.

Pause; widen the kernel; re-run.

- [ ] **Step 5: No commit** (verification step).

---

## Phase D — API dispatch (preserve public signature)

### Task 15: Rewrite `api.py` as dispatcher (4-kwarg signature intact)

**Files:**
- Modify: `src/mlx_teacache/api.py`
- Create: `tests/test_api_dispatch.py`
- Create: `tests/test_public_api.py`

Audit F3: `apply_teacache(flux, *, rel_l1_thresh=..., coefficients=None, skip_first_n_steps=1, skip_last_n_steps=1)`. All four explicit keyword params survive. Each variant's `apply()` accepts them.

- [ ] **Step 1: Read the current api.py**

```bash
sed -n '130,200p' src/mlx_teacache/api.py
```

Capture the v0.5.x signature, defaults, validation logic. The new dispatcher preserves all of it but moves the FLUX.1 vs FLUX.2 branching out (into the variant `apply()` functions).

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_api_dispatch.py
"""apply_teacache dispatches via _REGISTRY and preserves the v0.5.x
public signature (audit F3)."""
import inspect

import pytest

from mlx_teacache import apply_teacache
from mlx_teacache.errors import IncompatibleModelError


def test_signature_has_explicit_kwargs():
    """All four v0.5.x public kwargs must survive."""
    sig = inspect.signature(apply_teacache)
    expected = {"flux", "rel_l1_thresh", "coefficients",
                "skip_first_n_steps", "skip_last_n_steps"}
    actual = set(sig.parameters.keys())
    missing = expected - actual
    assert not missing, f"public kwargs missing: {missing}"


def test_skip_first_n_steps_default_is_1():
    sig = inspect.signature(apply_teacache)
    assert sig.parameters["skip_first_n_steps"].default == 1


def test_skip_last_n_steps_default_is_1():
    sig = inspect.signature(apply_teacache)
    assert sig.parameters["skip_last_n_steps"].default == 1


def test_coefficients_default_is_none():
    sig = inspect.signature(apply_teacache)
    assert sig.parameters["coefficients"].default is None


class _FC:
    def __init__(self, a):
        self.aliases = a
        self.model_name = "fake/x"


class _FakeFlux1:
    def __init__(self, a):
        self.model_config = _FC(a)


def test_unknown_variant_raises():
    with pytest.raises(IncompatibleModelError) as exc:
        apply_teacache(_FakeFlux1(["bogus"]))
    msg = str(exc.value)
    assert "flux1-dev" in msg
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_api_dispatch.py -v
```

Expected: signature tests pass against v0.5.x; dispatch test may fail until the new code is in.

- [ ] **Step 4: Rewrite `api.py`**

```python
# src/mlx_teacache/api.py
"""Public entry point. Variant dispatch via _REGISTRY.

The 4-kwarg public signature (rel_l1_thresh, coefficients,
skip_first_n_steps, skip_last_n_steps) is preserved exactly from v0.5.x.
Each variant's apply() accepts all four; the dispatcher forwards them.
"""
from __future__ import annotations

from typing import Any

from mlx_teacache.errors import IncompatibleModelError
from mlx_teacache.handle import TeaCacheHandle
from mlx_teacache.variants import _REGISTRY


def apply_teacache(
    flux: Any,
    *,
    rel_l1_thresh: float | None = None,
    coefficients: tuple[float, float, float, float, float] | None = None,
    skip_first_n_steps: int = 1,
    skip_last_n_steps: int = 1,
) -> TeaCacheHandle:
    """Enable TeaCache step-skipping. Walks the variant registry; the
    first variant whose matches(flux) returns True wins. Loads that
    variant's integration module lazily and dispatches with all four
    public kwargs."""
    for entry in _REGISTRY.values():
        if entry["matches"](flux):
            apply = entry["load_integration"]()
            return apply(
                flux,
                rel_l1_thresh=rel_l1_thresh,
                coefficients=coefficients,
                skip_first_n_steps=skip_first_n_steps,
                skip_last_n_steps=skip_last_n_steps,
            )

    model_config = getattr(flux, "model_config", None)
    model_name = getattr(model_config, "model_name", None)
    raise IncompatibleModelError(
        actual_type=type(flux).__name__,
        actual_model_name=model_name,
        supported=sorted(_REGISTRY.keys()),
    )
```

The handle class itself (`_HandleState`, `TeaCacheHandle`) is removed from `api.py` — those now live in `mlx_teacache/handle.py` (Task 6). The legacy `_remove_callback_by_identity` helper (around line 109) moves to `integrations/mflux/lifecycle.py` if not already there.

- [ ] **Step 5: Run dispatch tests + the full fast suite**

```bash
uv run pytest tests/test_api_dispatch.py -v
uv run pytest tests/ -m "not slow and not network" --deselect tests/test_api.py::test_apply_and_restore_roundtrip
```

Expected: all pass.

- [ ] **Step 6: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/api.py tests/test_api_dispatch.py
uv run mypy src/mlx_teacache/api.py
```

Expected: green.

- [ ] **Step 7: Commit**

```bash
git add src/mlx_teacache/api.py tests/test_api_dispatch.py
git commit -m "feat(api): rewrite as dispatcher; preserve 4-kwarg public signature (audit F3)"
```

---

### Task 16: Full public-API snapshot test

**Files:**
- Create: `tests/test_public_api.py`

Audit F1 + F3 + F4: gate on every documented v0.5.x import path and signature.

- [ ] **Step 1: Write the test**

```python
# tests/test_public_api.py
"""Public API surface snapshot. Locks v0.5.x → v0.6.0 compatibility."""
import inspect
import subprocess
import sys


def test_root_package_exports():
    import mlx_teacache
    for name in [
        "__version__", "apply_teacache", "TeaCacheHandle", "TeaCacheStats",
        "GenerationStats", "StepDecision", "Provenance",
        "TeaCacheError", "AlreadyPatchedError", "CalibrationError",
        "IncompatibleModelError", "InternalStateError", "InvalidStepWindowError",
        "MissingGenerationContextError", "StatsFrozenError",
        "TeaCacheNoBenefitWarning", "TransformerShapeError",
    ]:
        assert hasattr(mlx_teacache, name), f"missing public export: {name}"


def test_stats_submodule_paths():
    from mlx_teacache.stats import GenerationStats, StatsFrozenError, StepDecision, TeaCacheStats
    s = TeaCacheStats()
    assert s.computed_count == 0
    assert s.speedup_estimate == 1.0


def test_coefficients_provenance_path():
    from mlx_teacache.coefficients import Provenance
    assert Provenance.for_user_supplied().source == "user"


def test_gate_module_path():
    from mlx_teacache.gate import gate_step, GateDecision  # noqa: F401


def test_cache_module_path():
    from mlx_teacache.cache import TeaCacheState  # noqa: F401


def test_apply_teacache_signature():
    """All four explicit kwargs must survive (audit F3)."""
    from mlx_teacache import apply_teacache
    sig = inspect.signature(apply_teacache)
    for name in ("rel_l1_thresh", "coefficients",
                 "skip_first_n_steps", "skip_last_n_steps"):
        assert name in sig.parameters, f"public kwarg missing: {name}"


def test_base_import_without_mflux():
    """Audit F4: base-package import must work without [mflux] extra."""
    code = (
        "import sys\n"
        "sys.modules['mflux'] = None\n"
        "import mlx_teacache\n"
        "from mlx_teacache import apply_teacache\n"
        "assert callable(apply_teacache)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert "OK" in result.stdout, f"stderr={result.stderr}"
    assert result.returncode == 0
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/test_public_api.py -v
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_public_api.py
git commit -m "test(public-api): snapshot v0.5.x surface (kwargs + import paths + base-import)"
```

---

## Phase E — Legacy cleanup

### Task 17: Delete legacy forward + per-family integration modules

After Tasks 8-13 (all six variants ported) and Tasks 15-16 (dispatcher + public-api gate green), the legacy forward code in `integrations/mflux/` is unused. Trash it (per CLAUDE.md "Never permanently delete").

**Files to trash:**
- `src/mlx_teacache/integrations/mflux/forward.py` (664 lines)
- `src/mlx_teacache/integrations/mflux/flux1.py`
- `src/mlx_teacache/integrations/mflux/flux2.py`
- `src/mlx_teacache/integrations/mflux/detect.py`

**Files to KEEP** (shared mflux machinery used by every variant):
- `src/mlx_teacache/integrations/mflux/lifecycle.py` (`GenerationContextCallback`, `wrap_generate_image`)
- `src/mlx_teacache/integrations/mflux/__init__.py`

- [ ] **Step 1: Confirm no incoming imports from outside the legacy files**

```bash
grep -rn "from mlx_teacache.integrations.mflux.forward\|from mlx_teacache.integrations.mflux.flux1\|from mlx_teacache.integrations.mflux.flux2\|from mlx_teacache.integrations.mflux.detect" \
    src/ tests/ scripts/
```

Expected: results only from the variant `integration.py` files (and those are byte-for-byte ports — they don't depend on the legacy module surviving). If anything else surfaces, fix it before deletion.

- [ ] **Step 2: Move legacy files to Trash**

```bash
mv src/mlx_teacache/integrations/mflux/forward.py \
   ~/.Trash/mlx_teacache-integrations-mflux-forward-v0.5-legacy-$(date +%Y-%m-%d)
mv src/mlx_teacache/integrations/mflux/flux1.py \
   ~/.Trash/mlx_teacache-integrations-mflux-flux1-v0.5-legacy-$(date +%Y-%m-%d)
mv src/mlx_teacache/integrations/mflux/flux2.py \
   ~/.Trash/mlx_teacache-integrations-mflux-flux2-v0.5-legacy-$(date +%Y-%m-%d)
mv src/mlx_teacache/integrations/mflux/detect.py \
   ~/.Trash/mlx_teacache-integrations-mflux-detect-v0.5-legacy-$(date +%Y-%m-%d)
```

- [ ] **Step 3: Run the full fast suite**

```bash
uv run pytest tests/ -m "not slow and not network" --deselect tests/test_api.py::test_apply_and_restore_roundtrip
```

Expected: all pass.

- [ ] **Step 4: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/
uv run mypy src/mlx_teacache/
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git rm src/mlx_teacache/integrations/mflux/forward.py \
       src/mlx_teacache/integrations/mflux/flux1.py \
       src/mlx_teacache/integrations/mflux/flux2.py \
       src/mlx_teacache/integrations/mflux/detect.py
git commit -m "chore(legacy): remove per-family forward + detect (now in variants/)"
```

---

### Task 18: Delete legacy coefficient registry; switch variants to literal tuples

After every variant's `config.py` has a transitional `from mlx_teacache.coefficients import _LEGACY_NAME as _LEGACY` import (Tasks 8-13), the `_REGISTRY` and per-variant coefficient tuples in `src/mlx_teacache/coefficients.py` can be deleted. First each variant inlines its tuple; then the legacy module shrinks to just the `Provenance` re-export.

- [ ] **Step 1: For each variant, inline the coefficient tuple**

Open each `src/mlx_teacache/variants/<name>/config.py`. Replace the transitional import with the literal tuple values. The values are NOT in this plan — workers copy from the legacy `coefficients.py` at this step.

Example for `flux1_dev/config.py`:

```python
# BEFORE
from mlx_teacache.coefficients import _UPSTREAM_FLUX_COEFFS as _LEGACY_COEFFS
COEFFICIENTS: tuple[float, float, float, float, float] = _LEGACY_COEFFS

# AFTER (worker copies the literal from src/mlx_teacache/coefficients.py)
COEFFICIENTS: tuple[float, float, float, float, float] = (
    # paste lines from src/mlx_teacache/coefficients.py::_UPSTREAM_FLUX_COEFFS exactly
)
```

The identity-test guard `test_coefficients_match_v05_registry` in each variant's test file (Tasks 8-13) catches transcription errors at this step.

- [ ] **Step 2: Run the per-variant identity tests**

```bash
uv run pytest tests/variants/ -k "coefficients_match" -v
```

Expected: every variant's coefficient-identity test passes against the legacy `_REGISTRY` entry.

- [ ] **Step 3: Delete `_REGISTRY` and coefficient tuples from `coefficients.py`**

`src/mlx_teacache/coefficients.py` shrinks to:

```python
# src/mlx_teacache/coefficients.py
"""Compatibility shim. Re-exports Provenance from _kernel.coefficients.

The v0.5.x _REGISTRY and per-variant coefficient tuples moved to
src/mlx_teacache/variants/<name>/config.py.
"""
from mlx_teacache._kernel.coefficients import Provenance

__all__ = ["Provenance"]
```

The `load_builtin`, `validate_custom` functions move to `_kernel/coefficients.py` if any caller in `api.py` or variant code uses them.

- [ ] **Step 4: Run the full fast suite + public-api tests**

```bash
uv run pytest tests/ -m "not slow and not network" \
              --deselect tests/test_api.py::test_apply_and_restore_roundtrip
```

Expected: all pass.

- [ ] **Step 5: Now delete the per-variant identity tests** (they referenced the legacy `_REGISTRY` which no longer exists). Replace with a runtime check inside `apply_teacache` startup, OR rely on the variant's own self-consistency tests.

Actually, simpler: leave the identity tests in place but mark them `xfail` or delete after this task. The audit's intent (F4) is to catch transcription errors during the migration. Once the migration is done, the tests are no longer useful.

- [ ] **Step 6: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/ tests/
uv run mypy src/mlx_teacache/
```

Expected: green.

- [ ] **Step 7: Commit**

```bash
git add src/mlx_teacache/coefficients.py src/mlx_teacache/variants/
git commit -m "chore(legacy): inline per-variant coefficient tuples; coefficients.py shrinks to Provenance shim"
```

---

## Phase F — Bench refactor (subprocess-per-rep)

### Task 19: Refactor `scripts/bench_speedup.py` to subprocess-per-rep

**Files:**
- Modify: `scripts/bench_speedup.py`

The v0.5.1 work folded into v0.6.0. Each (variant, condition, rep) runs in a fresh subprocess. Worker prints `::BENCH_RESULT::<json>` sentinel; orchestrator aggregates. Modeled on `scripts/bench_comparison.py`.

The refactor is a self-contained replacement of `scripts/bench_speedup.py`. See Task 23 in the previous plan draft for the full skeleton; the new version reads variant metadata from `_REGISTRY` and applies `memory_cap_hint_gb` in each worker.

Steps: write the new script, dry-run `--help` to confirm parsing, lint, commit. No tests for the bench harness itself — its output is the JSON report consumed by Phase G validation.

```bash
git add scripts/bench_speedup.py
git commit -m "feat(bench): subprocess-per-rep harness (v0.5.1 work folded into v0.6.0)"
```

---

## Phase G — Validation runs

### Task 20: Fast suite + lint sweep

After the structural work, run the full fast suite and the repo-wide lint/typecheck.

```bash
uv run pytest tests/ -m "not slow and not network" --deselect tests/test_api.py::test_apply_and_restore_roundtrip
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
```

Expected: all green. If anything fails, fix and re-run before continuing.

### Task 21: Three-way bench — klein-base-9b

**Estimated wall-clock: 3-4 hours on M1 Max.** Heavy ML run; per CLAUDE.md "Always state an ETA" + "Memory guardrails for heavy generations on 32 GB".

```bash
uv run python scripts/bench_speedup.py \
    --variant flux2-klein-base-9b \
    --three-way \
    --reps 3 \
    --report _artifacts/v0.6.0_bench_klein_base_9b.json \
    2>&1 | tee /tmp/v0.6.0-bench-klein-base-9b.log
```

Commit the JSON when complete.

### Task 22: Three-way bench — klein-base-4b (v0.4.1 sanity check)

**Estimated wall-clock: 1.5-2 hours on M1 Max.**

```bash
uv run python scripts/bench_speedup.py \
    --variant flux2-klein-base-4b \
    --three-way \
    --reps 3 \
    --report _artifacts/v0.6.0_bench_klein_base_4b.json \
    2>&1 | tee /tmp/v0.6.0-bench-klein-base-4b.log
```

Compare to v0.4.1 (gating 1.16×, compile-avoidance 1.09×, combined 1.26×). Divergence > 5% is a finding.

---

## Phase H — Docs + ship

### Task 23: Generated "Supported models" table

Small script `docs/_generate_supported_models.py` reads `_REGISTRY` and emits a markdown table. README has `<!-- SUPPORTED_MODELS_START -->` / `END` markers; paste the generated table between them.

### Task 24: Per-variant docs under `docs/variants/<name>.md`

One file per variant: license + obligations, recommended recipe, memory cap hint, validation evidence link, quirks.

### Task 25: CHANGELOG v0.6.0 + ROADMAP update

CHANGELOG entry describing the architectural refactor + the measured klein-base-9b three-way numbers from Task 21. ROADMAP: v0.6.0 → Released.

Run `/humanizer` on the new CHANGELOG entry (substantive public-facing prose).

### Task 26: PR + CI + STOP

```bash
git push -u origin feature/v0.6.0-per-variant-cores
gh pr create --title "v0.6.0: per-variant cores + shared algorithmic kernel" --body "$(cat <<'EOF'
[see plan task 26 in the previous draft for full PR body]
EOF
)"
```

Per the release-flow rule: STOP after PR opens. Human merges; tag push to PyPI is a separate explicit authorization.

---

## Self-review

**Spec coverage:** every spec section maps to tasks:
- Architecture → Tasks 1-13
- VariantPatch contract → Task 6
- Compatibility shims (audit F1) → Tasks 2-5
- Public-import-path gate (audit F1) → Task 16
- Base-import-without-mflux (audit F4) → Tasks 7, 16
- `VariantPatch` no variant branches (audit F3 plan-level concern) → Task 6
- Apply-teacache 4-kwarg signature (audit F3) → Task 15
- Verbatim extraction discipline (audit F1) → Tasks 2-4
- Cache/state fields preserved (audit F1) → Task 3
- Stats commit/discard with mflux lifecycle (audit F2) → Task 6 (handle does NOT finalize), Task 17 (lifecycle.py kept shared)
- Real file paths only (audit F5) → Tasks 17 (delete only files that exist), 18
- Integration verbatim port (audit F6) → Tasks 8-13
- Three-way bench → Tasks 19, 21, 22

**Placeholder scan:** every `PORT:` marker explicitly names the source file and lines. No coefficient literals in this plan. No "TBD" outside intentional measurement placeholders.

**Type consistency:** `META`, `matches`, `apply`, `COEFFICIENTS`, `DEFAULT_THRESH`, `RECIPES`, `LICENSE`, `VariantPatch`, `TeaCacheHandle`, `TeaCacheState`, `TeaCacheStats`, `gate_step`, `GateDecision` names match the real v0.5.x source.

**Real-codebase grounding:**
- `src/mlx_teacache/gate.py` exists (135 lines) — Task 2 copies into `_kernel/gate.py`.
- `src/mlx_teacache/cache.py` exists (55 lines) — Task 3 copies into `_kernel/cache.py`.
- `src/mlx_teacache/stats.py` exists (172 lines) — Task 4 copies into `_kernel/stats.py`.
- `src/mlx_teacache/coefficients.py` exists (250 lines) — Task 5 extracts `Provenance`; Task 18 deletes the legacy registry.
- `src/mlx_teacache/integrations/mflux/lifecycle.py` exists (262 lines) — STAYS, shared across variants.
- `src/mlx_teacache/integrations/mflux/forward.py` exists (664 lines) — Task 17 deletes after variants port their copies.
- `src/mlx_teacache/integrations/mflux/{flux1,flux2,detect}.py` exist — Task 17 deletes.
- There is NO top-level `state.py` or `lifecycle.py` — the previous plan draft's references to these were wrong and have been removed.
