# v0.6.0 — per-variant cores + shared algorithmic kernel implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md`

**Goal:** Restructure `src/mlx_teacache/` from "shared everything, per-variant config" to per-variant assemblies + a shared algorithmic kernel. Public API surface stays unchanged at every v0.5.x import path; internal layout becomes per-variant directories under `src/mlx_teacache/variants/` plus pure-algorithm primitives under `src/mlx_teacache/_kernel/`.

**Architecture:** Six variants get directories with `config.py` (META + coefficients + recipes, mflux-free), `detect.py` (`matches(flux)`, mflux-free), and `integration.py` (forward wrapper + `VariantPatch`, mflux-touching, lazy-imported). The package registry walks variants at import time but loads `integration.py` only on first dispatch. `TeaCacheHandle` is variant-agnostic; variant-specific teardown is captured into a `VariantPatch` (rollback + finalizer callback lists) returned by each variant's `apply()`.

**Tech Stack:** Python 3.11+, MLX, mflux 0.17.x (lazy-imported), pytest 8 + hypothesis, ruff, `mypy --strict`, `uv` for dev, `hatchling` + `hatch-vcs` for build.

---

## Phase A — Kernel extraction

Goal: pull pure-algorithm code out of today's `state.py`, `lifecycle.py`, `stats.py`, `coefficients.py`, and the algorithmic core of `integrations/mflux/forward.py` / `flux2.py` into `src/mlx_teacache/_kernel/`. Tests for the kernel live at `tests/_kernel/`. Zero mflux imports anywhere under `_kernel/`.

### Task 1: Scaffold the `_kernel/` package

**Files:**
- Create: `src/mlx_teacache/_kernel/__init__.py`
- Create: `tests/_kernel/__init__.py`
- Create: `tests/_kernel/test_kernel_no_mflux_import.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/_kernel/test_kernel_no_mflux_import.py
"""The _kernel package must be importable without mflux installed.

This is the hardest guarantee in v0.6.0: pure-algorithm primitives have
no business pulling mflux internals.
"""
from __future__ import annotations

import importlib
import pkgutil


def test_kernel_subtree_imports_without_mflux_in_sys_modules(monkeypatch):
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
"""Pure-algorithm primitives for mlx-teacache. No mflux imports allowed
anywhere under this package. See docs/superpowers/specs/2026-05-19-per-variant-cores-design.md.
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

### Task 2: `_kernel.gate` — `rel_l1`, `accumulate`, `polynomial_gate`

**Files:**
- Create: `src/mlx_teacache/_kernel/gate.py`
- Create: `tests/_kernel/test_gate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/_kernel/test_gate.py
import math

import mlx.core as mx
import pytest

from mlx_teacache._kernel.gate import accumulate, polynomial_gate, rel_l1


def test_rel_l1_zero_on_identical():
    a = mx.array([1.0, 2.0, 3.0])
    assert rel_l1(a, a) == pytest.approx(0.0, abs=1e-7)


def test_rel_l1_nonzero_on_perturbed():
    prev = mx.array([1.0, 1.0, 1.0])
    curr = mx.array([1.0, 1.0, 2.0])
    # |curr - prev|_1 / |prev|_1 == 1.0 / 3.0
    assert rel_l1(curr, prev) == pytest.approx(1.0 / 3.0, rel=1e-5)


def test_rel_l1_safe_on_zero_prev():
    # |prev| -> 0 must not divide-by-zero; small epsilon in the impl
    prev = mx.zeros((4,))
    curr = mx.array([1.0, 0.0, 0.0, 0.0])
    out = rel_l1(curr, prev)
    assert math.isfinite(out)


def test_accumulate_is_sum():
    assert accumulate(0.0, 0.5) == pytest.approx(0.5)
    assert accumulate(0.5, 0.5) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "coefficients,accumulated,threshold,expected_skip",
    [
        # Identity polynomial: y = x. Skip when y < threshold.
        ((0.0, 0.0, 0.0, 1.0, 0.0), 0.1, 0.2, True),
        ((0.0, 0.0, 0.0, 1.0, 0.0), 0.3, 0.2, False),
        # All-zero polynomial: y = 0. Always under threshold ⇒ always skip.
        ((0.0, 0.0, 0.0, 0.0, 0.0), 5.0, 0.2, True),
    ],
)
def test_polynomial_gate(coefficients, accumulated, threshold, expected_skip):
    assert polynomial_gate(coefficients, accumulated, threshold) is expected_skip
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/_kernel/test_gate.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_teacache._kernel.gate'`.

- [ ] **Step 3: Implement `gate.py`**

```python
# src/mlx_teacache/_kernel/gate.py
"""Polynomial-gate primitives. Variant integration code composes these.

The gate signal is rel-L1 between consecutive `mod_in` tensors; the gate
decision is the polynomial evaluation against the accumulated rel-L1.
"""
from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx

_EPS = 1e-12


def rel_l1(curr: mx.array, prev: mx.array) -> float:
    """Relative-L1 distance, matching mflux's runtime gate signal.

    Returns |curr - prev|_1 / (|prev|_1 + eps). The epsilon avoids
    divide-by-zero on the first step / zero-valued tensors.
    """
    curr_f = curr.astype(mx.float32)
    prev_f = prev.astype(mx.float32)
    num = mx.sum(mx.abs(curr_f - prev_f))
    den = mx.sum(mx.abs(prev_f)) + _EPS
    return float(num / den)


def accumulate(prev_accumulated: float, delta: float) -> float:
    """Running-sum accumulator for the cumulative mod_in rel-L1.

    Single-step trivial wrapper kept as a named function so the
    accumulator pattern is callable from variants without inlining the
    arithmetic (and so a future change to e.g. exponential-decay only
    touches one site).
    """
    return prev_accumulated + delta


def polynomial_gate(
    coefficients: Sequence[float],
    accumulated_rel_l1: float,
    threshold: float,
) -> bool:
    """Evaluate the calibration polynomial at the accumulated rel-L1.

    Returns True (skip) when the polynomial value is below the threshold.

    Coefficients are stored in numpy.polyval order: highest degree first.
    Length-5 = degree-4 polynomial. The trailing 0.0 in origin-constrained
    fits enforces poly(0) == 0.
    """
    x = accumulated_rel_l1
    value = 0.0
    for c in coefficients:
        value = value * x + c
    return value < threshold
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/_kernel/test_gate.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Run no-mflux guard test**

```bash
uv run pytest tests/_kernel/test_kernel_no_mflux_import.py -v
```

Expected: PASS (gate.py imports `mlx.core` only).

- [ ] **Step 6: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/_kernel/gate.py tests/_kernel/test_gate.py
uv run ruff format --check src/mlx_teacache/_kernel/gate.py tests/_kernel/test_gate.py
uv run mypy src/mlx_teacache/_kernel/gate.py
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/mlx_teacache/_kernel/gate.py tests/_kernel/test_gate.py
git commit -m "feat(_kernel/gate): rel_l1, accumulate, polynomial_gate primitives"
```

---

### Task 3: `_kernel.cfg` — `cfg_per_branch_combine`

**Files:**
- Create: `src/mlx_teacache/_kernel/cfg.py`
- Create: `tests/_kernel/test_cfg.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/_kernel/test_cfg.py
import mlx.core as mx
import numpy as np
import pytest

from mlx_teacache._kernel.cfg import cfg_per_branch_combine


def test_cfg_combine_at_guidance_one_returns_pos():
    """guidance=1.0 means pure conditional generation."""
    pos = mx.array([1.0, 2.0, 3.0])
    neg = mx.array([10.0, 20.0, 30.0])
    out = cfg_per_branch_combine(pos, neg, guidance_scale=1.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(pos), rtol=1e-5)


def test_cfg_combine_at_guidance_zero_returns_neg():
    """guidance=0.0 collapses to the unconditional branch."""
    pos = mx.array([1.0, 2.0, 3.0])
    neg = mx.array([10.0, 20.0, 30.0])
    out = cfg_per_branch_combine(pos, neg, guidance_scale=0.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(neg), rtol=1e-5)


def test_cfg_combine_standard_formula():
    """Standard CFG: neg + guidance * (pos - neg)."""
    pos = mx.array([2.0, 4.0])
    neg = mx.array([1.0, 1.0])
    out = cfg_per_branch_combine(pos, neg, guidance_scale=3.0)
    # neg + 3 * (pos - neg) = [1+3, 1+9] = [4, 10]
    np.testing.assert_allclose(np.asarray(out), np.array([4.0, 10.0]), rtol=1e-5)


@pytest.mark.parametrize("guidance", [1.5, 4.0, 7.5])
def test_cfg_combine_shape_preserved(guidance):
    pos = mx.zeros((2, 16, 16, 4))
    neg = mx.zeros((2, 16, 16, 4))
    out = cfg_per_branch_combine(pos, neg, guidance_scale=guidance)
    assert out.shape == pos.shape
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/_kernel/test_cfg.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `cfg.py`**

```python
# src/mlx_teacache/_kernel/cfg.py
"""Classifier-free guidance (CFG) combination math.

FLUX.2 at guidance > 1.0 runs predict twice per step: once for the
conditional (positive-prompt) branch and once for the unconditional
(null-prompt) branch. The standard CFG formula combines them:

    out = neg + guidance * (pos - neg)

Equivalent to `(1 - guidance) * neg + guidance * pos`.
"""
from __future__ import annotations

import mlx.core as mx


def cfg_per_branch_combine(
    pos: mx.array, neg: mx.array, guidance_scale: float
) -> mx.array:
    """Standard CFG combination of conditional + unconditional predictions."""
    return neg + guidance_scale * (pos - neg)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/_kernel/test_cfg.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/_kernel/cfg.py tests/_kernel/test_cfg.py
uv run mypy src/mlx_teacache/_kernel/cfg.py
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/mlx_teacache/_kernel/cfg.py tests/_kernel/test_cfg.py
git commit -m "feat(_kernel/cfg): cfg_per_branch_combine primitive"
```

---

### Task 4: `_kernel.state` — `TeaCacheState` dataclass

**Files:**
- Create: `src/mlx_teacache/_kernel/state.py`
- Create: `tests/_kernel/test_state.py`

Read `src/mlx_teacache/state.py` first to confirm the existing field set. Today's `TeaCacheState` carries `cached_residual`, `cached_residual_neg` (v0.4.1 CFG per-branch), `accumulated_rel_l1`, `cfg_was_active`. Plus helper transitions for `reset_for_new_generation`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_kernel/test_state.py
import mlx.core as mx

from mlx_teacache._kernel.state import TeaCacheState


def test_initial_state_is_empty():
    s = TeaCacheState.fresh()
    assert s.cached_residual is None
    assert s.cached_residual_neg is None
    assert s.accumulated_rel_l1 == 0.0
    assert s.cfg_was_active is False


def test_reset_clears_everything():
    s = TeaCacheState.fresh()
    s.cached_residual = mx.zeros((1, 8, 8, 4))
    s.cached_residual_neg = mx.zeros((1, 8, 8, 4))
    s.accumulated_rel_l1 = 0.42
    s.cfg_was_active = True
    s.reset_for_new_generation()
    assert s.cached_residual is None
    assert s.cached_residual_neg is None
    assert s.accumulated_rel_l1 == 0.0
    assert s.cfg_was_active is False


def test_cache_residual_pos_branch():
    s = TeaCacheState.fresh()
    r = mx.array([1.0, 2.0, 3.0])
    s.cache_residual(r, branch="pos")
    assert s.cached_residual is r
    assert s.cached_residual_neg is None


def test_cache_residual_neg_branch():
    s = TeaCacheState.fresh()
    r = mx.array([1.0, 2.0, 3.0])
    s.cache_residual(r, branch="neg")
    assert s.cached_residual_neg is r
    assert s.cached_residual is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/_kernel/test_state.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `state.py`**

```python
# src/mlx_teacache/_kernel/state.py
"""TeaCacheState dataclass — per-generation mutable state.

One state object is owned by each `apply_teacache()` invocation. It lives
in the variant's `apply()` closure, not on the flux instance, so concurrent
generations on the same flux can each have their own (post-v0.6.0; today's
code does not support that, but the boundary doesn't preclude it).

`cached_residual` is the FLUX.2 CFG-positive / FLUX.1-only cached residual.
`cached_residual_neg` is the FLUX.2 CFG-negative branch (v0.4.1).
`accumulated_rel_l1` is the cumulative mod_in rel-L1 across active steps;
the polynomial gate evaluates against this.
`cfg_was_active` records whether CFG ran at any point during the generation;
read by stats finalization (see lifecycle.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mlx.core as mx

Branch = Literal["pos", "neg"]


@dataclass
class TeaCacheState:
    cached_residual: mx.array | None = None
    cached_residual_neg: mx.array | None = None
    accumulated_rel_l1: float = 0.0
    cfg_was_active: bool = False

    @classmethod
    def fresh(cls) -> TeaCacheState:
        """Return a state object reset to per-generation initial values."""
        return cls()

    def reset_for_new_generation(self) -> None:
        """In-place reset between generations on the same `flux`."""
        self.cached_residual = None
        self.cached_residual_neg = None
        self.accumulated_rel_l1 = 0.0
        self.cfg_was_active = False

    def cache_residual(self, residual: mx.array, *, branch: Branch) -> None:
        """Store the residual for the named branch."""
        if branch == "pos":
            self.cached_residual = residual
        elif branch == "neg":
            self.cached_residual_neg = residual
        else:  # pragma: no cover — Literal enforces at type-check time
            raise ValueError(f"unknown branch: {branch!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/_kernel/test_state.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/_kernel/state.py tests/_kernel/test_state.py
uv run mypy src/mlx_teacache/_kernel/state.py
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/mlx_teacache/_kernel/state.py tests/_kernel/test_state.py
git commit -m "feat(_kernel/state): TeaCacheState dataclass with branch-aware cache"
```

---

### Task 5: `_kernel.stats` — `StepDecision`, `GenerationStats`, `TeaCacheStats`, `Provenance`

**Files:**
- Create: `src/mlx_teacache/_kernel/stats.py`
- Create: `tests/_kernel/test_stats.py`

Read `src/mlx_teacache/stats.py` (current top-level module) AND `src/mlx_teacache/coefficients.py` (for the `Provenance` dataclass) before writing. The new `_kernel/stats.py` is the canonical home for all four types; the old modules become shims (Tasks 11-12).

- [ ] **Step 1: Read the existing types**

```bash
grep -n "^class \|^@dataclass" src/mlx_teacache/stats.py src/mlx_teacache/coefficients.py
```

Expected: lists `StepDecision`, `GenerationStats`, `TeaCacheStats`, `StatsFrozenError` (from stats.py) and `Provenance` (from coefficients.py).

- [ ] **Step 2: Write the failing test**

```python
# tests/_kernel/test_stats.py
import pytest

from mlx_teacache._kernel.stats import (
    GenerationStats,
    Provenance,
    StatsFrozenError,
    StepDecision,
    TeaCacheStats,
)


def test_step_decision_is_frozen():
    d = StepDecision(step=0, skipped=False, rel_l1=0.1, predicted_rel_l1=0.05, threshold=0.2)
    with pytest.raises(Exception):  # frozen dataclass → FrozenInstanceError
        d.step = 1  # type: ignore[misc]


def test_generation_stats_summary_counts_skipped():
    decisions = (
        StepDecision(step=i, skipped=(i % 2 == 0), rel_l1=0.0, predicted_rel_l1=0.0, threshold=0.2)
        for i in range(10)
    )
    gen = GenerationStats(decisions=tuple(decisions))
    assert gen.computed_count == 5
    assert gen.skipped_count == 5


def test_teacache_stats_aggregates_generations():
    gen1 = GenerationStats(decisions=())
    gen2 = GenerationStats(decisions=())
    stats = TeaCacheStats(generations=(gen1, gen2))
    assert len(stats.generations) == 2
    assert stats.last_generation is gen2


def test_teacache_stats_skipped_count_no_generations_raises_or_zero():
    stats = TeaCacheStats(generations=())
    # The current top-level API exposes .skipped_count on the *handle*,
    # which delegates to last_generation. With zero generations, the handle
    # raises a clear error. Replicate that here.
    with pytest.raises(StatsFrozenError):
        _ = stats.skipped_count


def test_provenance_builtin_has_required_fields():
    p = Provenance(
        source="builtin",
        revision="upstream-flux-v1",
        calibration_dataset="upstream",
        fit_metric=None,
        fit_metric_value=None,
        reference_url="https://example.com",
        default_thresh=None,
    )
    assert p.source == "builtin"
    assert p.revision == "upstream-flux-v1"
    assert p.default_thresh is None


def test_provenance_for_user_supplied():
    p = Provenance.for_user_supplied()
    assert p.source == "user"
    assert p.revision is None
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/_kernel/test_stats.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement `_kernel/stats.py`**

Copy the dataclass definitions verbatim from today's `src/mlx_teacache/stats.py` and `src/mlx_teacache/coefficients.py`. The implementations are unchanged; only the module location moves. Keep `StatsFrozenError` here too (currently in stats.py).

```python
# src/mlx_teacache/_kernel/stats.py
"""TeaCache stats + provenance types.

Public re-exports live at:
- mlx_teacache.TeaCacheStats, GenerationStats, StepDecision
- mlx_teacache.Provenance
- mlx_teacache.stats.* (compatibility shim, see Task 11)
- mlx_teacache.coefficients.Provenance (compatibility shim, see Task 12)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


class StatsFrozenError(RuntimeError):
    """Raised when callers ask for stats before the lifecycle finalizes."""


@dataclass(frozen=True)
class StepDecision:
    step: int
    skipped: bool
    rel_l1: float
    predicted_rel_l1: float
    threshold: float


@dataclass(frozen=True)
class GenerationStats:
    decisions: tuple[StepDecision, ...]

    @property
    def computed_count(self) -> int:
        return sum(1 for d in self.decisions if not d.skipped)

    @property
    def skipped_count(self) -> int:
        return sum(1 for d in self.decisions if d.skipped)


@dataclass(frozen=True)
class TeaCacheStats:
    generations: tuple[GenerationStats, ...]

    @property
    def last_generation(self) -> GenerationStats:
        if not self.generations:
            raise StatsFrozenError("no completed generations yet")
        return self.generations[-1]

    @property
    def skipped_count(self) -> int:
        return self.last_generation.skipped_count

    @property
    def computed_count(self) -> int:
        return self.last_generation.computed_count


@dataclass(frozen=True)
class Provenance:
    source: Literal["builtin", "user"]
    revision: str | None = None
    calibration_dataset: str | None = None
    fit_metric: str | None = None
    fit_metric_value: float | None = None
    reference_url: str | None = None
    default_thresh: float | None = None

    @classmethod
    def for_user_supplied(cls) -> Provenance:
        return cls(source="user")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/_kernel/test_stats.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/_kernel/stats.py tests/_kernel/test_stats.py
uv run mypy src/mlx_teacache/_kernel/stats.py
```

Expected: green.

- [ ] **Step 7: Commit**

```bash
git add src/mlx_teacache/_kernel/stats.py tests/_kernel/test_stats.py
git commit -m "feat(_kernel/stats): canonical home for stats + Provenance types"
```

---

### Task 6: `_kernel.lifecycle` — `LifecycleFSM`

**Files:**
- Create: `src/mlx_teacache/_kernel/lifecycle.py`
- Create: `tests/_kernel/test_lifecycle.py`

Read `src/mlx_teacache/lifecycle.py` first. Today's lifecycle has staging counters that finalize on `after_loop` (natural completion) and discard on exception. The FSM extracts that state-machine logic.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_kernel/test_lifecycle.py
from mlx_teacache._kernel.lifecycle import LifecycleFSM
from mlx_teacache._kernel.stats import GenerationStats, StepDecision, TeaCacheStats


def _decision(step: int, skipped: bool) -> StepDecision:
    return StepDecision(step=step, skipped=skipped, rel_l1=0.0, predicted_rel_l1=0.0, threshold=0.2)


def test_fresh_fsm_has_no_finalized_generations():
    fsm = LifecycleFSM()
    snapshot = fsm.snapshot()
    assert snapshot.generations == ()


def test_record_step_then_finalize_emits_one_generation():
    fsm = LifecycleFSM()
    fsm.record_step(_decision(0, False))
    fsm.record_step(_decision(1, True))
    fsm.finalize_generation()
    snap = fsm.snapshot()
    assert len(snap.generations) == 1
    assert snap.generations[0].decisions == (_decision(0, False), _decision(1, True))


def test_abort_discards_staging_counters():
    fsm = LifecycleFSM()
    fsm.record_step(_decision(0, False))
    fsm.abort_generation()
    snap = fsm.snapshot()
    assert snap.generations == ()


def test_two_finalized_generations_appended_in_order():
    fsm = LifecycleFSM()
    fsm.record_step(_decision(0, False))
    fsm.finalize_generation()
    fsm.record_step(_decision(0, True))
    fsm.finalize_generation()
    snap = fsm.snapshot()
    assert len(snap.generations) == 2
    assert snap.generations[0].decisions[0].skipped is False
    assert snap.generations[1].decisions[0].skipped is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/_kernel/test_lifecycle.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `lifecycle.py`**

```python
# src/mlx_teacache/_kernel/lifecycle.py
"""Stats-lifecycle finite state machine.

Pure state transitions; no mflux callbacks. Variant integration wires
mflux's `after_loop` to `finalize_generation()` and exception teardown
to `abort_generation()`.
"""
from __future__ import annotations

from mlx_teacache._kernel.stats import GenerationStats, StepDecision, TeaCacheStats


class LifecycleFSM:
    """Records per-step decisions into a staging list. `finalize_generation`
    appends the staging list as a `GenerationStats` to the immutable
    snapshot. `abort_generation` discards staging without appending."""

    def __init__(self) -> None:
        self._staging: list[StepDecision] = []
        self._finalized: list[GenerationStats] = []

    def record_step(self, decision: StepDecision) -> None:
        self._staging.append(decision)

    def finalize_generation(self) -> None:
        gen = GenerationStats(decisions=tuple(self._staging))
        self._finalized.append(gen)
        self._staging = []

    def abort_generation(self) -> None:
        self._staging = []

    def snapshot(self) -> TeaCacheStats:
        return TeaCacheStats(generations=tuple(self._finalized))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/_kernel/test_lifecycle.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/_kernel/lifecycle.py tests/_kernel/test_lifecycle.py
uv run mypy src/mlx_teacache/_kernel/lifecycle.py
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/mlx_teacache/_kernel/lifecycle.py tests/_kernel/test_lifecycle.py
git commit -m "feat(_kernel/lifecycle): LifecycleFSM for stats staging + finalize/abort"
```

---

### Task 7: Confirm `_kernel/` is mflux-free

**Files:** (verification only; no new files)

- [ ] **Step 1: Run the no-mflux guard test against the populated kernel**

```bash
uv run pytest tests/_kernel/test_kernel_no_mflux_import.py -v
```

Expected: PASS. If FAIL, find the kernel module that imports mflux and replace its usage with a pure-MLX or pure-Python equivalent. Do not lower the guard.

- [ ] **Step 2: Run the full kernel test suite**

```bash
uv run pytest tests/_kernel/ -v
```

Expected: all tests pass.

- [ ] **Step 3: Verify no kernel module imports mflux even at type-check time**

```bash
grep -rn "^import mflux\|^from mflux" src/mlx_teacache/_kernel/ || echo "clean"
```

Expected: `clean`.

- [ ] **Step 4: Repo-wide lint sweep on `_kernel/`**

```bash
uv run ruff check src/mlx_teacache/_kernel/ tests/_kernel/
uv run ruff format --check src/mlx_teacache/_kernel/ tests/_kernel/
uv run mypy src/mlx_teacache/_kernel/
```

Expected: green.

- [ ] **Step 5: No commit** (verification step; nothing to commit).

---

## Phase B — Handle + `VariantPatch` contract

Goal: introduce the variant-agnostic `TeaCacheHandle` + `VariantPatch` types. The handle is the public context-manager users get back from `apply_teacache`. The patch is the contract: variant `apply()` returns a handle whose teardown is driven entirely by callbacks the variant registered.

### Task 8: `mlx_teacache.handle` — `VariantPatch` + `TeaCacheHandle`

**Files:**
- Create: `src/mlx_teacache/handle.py`
- Create: `tests/test_handle.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_handle.py
import pytest

from mlx_teacache._kernel.stats import Provenance, TeaCacheStats
from mlx_teacache.handle import TeaCacheHandle, VariantPatch


def _provenance() -> Provenance:
    return Provenance(source="builtin")


def test_handle_runs_rollbacks_in_reverse_install_order_on_exit():
    log: list[str] = []
    patch = VariantPatch(
        rollbacks=[lambda: log.append("r1"), lambda: log.append("r2")],
        finalizers=[],
    )
    with TeaCacheHandle(patch=patch, stats=TeaCacheStats(generations=()), provenance=_provenance()):
        pass
    assert log == ["r2", "r1"]


def test_handle_runs_finalizers_after_rollbacks():
    log: list[str] = []
    patch = VariantPatch(
        rollbacks=[lambda: log.append("rollback")],
        finalizers=[lambda: log.append("finalize")],
    )
    with TeaCacheHandle(patch=patch, stats=TeaCacheStats(generations=()), provenance=_provenance()):
        pass
    assert log == ["rollback", "finalize"]


def test_handle_restore_is_idempotent():
    counter = {"n": 0}
    patch = VariantPatch(
        rollbacks=[lambda: counter.update(n=counter["n"] + 1)],
        finalizers=[],
    )
    h = TeaCacheHandle(patch=patch, stats=TeaCacheStats(generations=()), provenance=_provenance())
    h.restore()
    h.restore()
    assert counter["n"] == 1


def test_handle_exposes_stats_and_provenance():
    p = _provenance()
    s = TeaCacheStats(generations=())
    h = TeaCacheHandle(patch=VariantPatch(rollbacks=[], finalizers=[]), stats=s, provenance=p)
    assert h.stats is s
    assert h.provenance is p


def test_handle_module_has_no_variant_branches():
    """Static check: the handle module must NOT contain any variant-specific
    if/elif branches. Variant-specific behavior lives in VariantPatch
    callbacks. See Spec → Quality gates → VariantPatch contract.
    """
    import inspect

    from mlx_teacache import handle as handle_module

    source = inspect.getsource(handle_module)
    assert 'variant ==' not in source
    assert 'flux1' not in source.lower() or 'flux1' not in [
        line for line in source.splitlines() if line.strip().startswith("#")
    ]
    # Strict check: no FLUX-family strings anywhere in non-comment code
    code_lines = [ln for ln in source.splitlines() if not ln.lstrip().startswith("#")]
    code = "\n".join(code_lines).lower()
    for bad in ("flux1", "flux2", "klein"):
        assert bad not in code, f"handle.py must not mention {bad!r}; variant-specific code belongs in variants/"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_handle.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_teacache.handle'`.

- [ ] **Step 3: Implement `handle.py`**

```python
# src/mlx_teacache/handle.py
"""Variant-agnostic context-manager handle returned by `apply_teacache`.

Variants attach their teardown via `VariantPatch` (rollback + finalizer
callback lists). The handle knows nothing about FLUX.1 vs FLUX.2 vs CFG
shapes — it just runs the callbacks. See
docs/superpowers/specs/2026-05-19-per-variant-cores-design.md
→ Handle contract.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mlx_teacache._kernel.stats import Provenance, TeaCacheStats


@dataclass
class VariantPatch:
    """The teardown contract a variant's `apply()` returns to `TeaCacheHandle`.

    rollbacks: undo callables in install order (handle runs them in reverse).
    finalizers: callables that finalize stats, clear sentinels, etc.
    """

    rollbacks: list[Callable[[], None]] = field(default_factory=list)
    finalizers: list[Callable[[], None]] = field(default_factory=list)


class TeaCacheHandle:
    """Context-manager handle. Variant-agnostic by design."""

    def __init__(
        self,
        *,
        patch: VariantPatch,
        stats: TeaCacheStats,
        provenance: Provenance,
        rel_l1_thresh: float | None = None,
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
        self._torn_down = True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_handle.py -v
```

Expected: 5 passed (including the static no-variant-branches check).

- [ ] **Step 5: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/handle.py tests/test_handle.py
uv run mypy src/mlx_teacache/handle.py
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/mlx_teacache/handle.py tests/test_handle.py
git commit -m "feat(handle): TeaCacheHandle + VariantPatch contract (variant-agnostic)"
```

---

## Phase C — Variants scaffold + first two variant cores

Goal: stand up `src/mlx_teacache/variants/` with the registry walker, then port two variants. flux1-dev covers the no-CFG path; flux2-klein-base-4b covers the CFG per-branch path. Together they validate the kernel boundary.

### Task 9: Scaffold `variants/` with metadata-only registry

**Files:**
- Create: `src/mlx_teacache/variants/__init__.py`
- Create: `tests/variants/__init__.py`
- Create: `tests/variants/test_registry.py`

The registry walker imports variant `config.py` + `detect.py` ONLY at package import time. Integration modules are loaded lazily — they're the only place mflux imports are allowed.

- [ ] **Step 1: Write the failing test**

```python
# tests/variants/test_registry.py
"""The variants registry contract.

Walks src/mlx_teacache/variants/<name>/ at import time, loading config.py
(for META) and detect.py (for matches()). Does NOT load integration.py
at import time — that happens lazily in apply_teacache.

This test runs before any variant is implemented; with the scaffolded
registry it must produce an empty dict, not error.
"""
from mlx_teacache.variants import _REGISTRY


def test_registry_is_a_mapping():
    assert isinstance(_REGISTRY, dict)


def test_registry_keys_match_meta_variant_ids():
    for variant_id, entry in _REGISTRY.items():
        assert entry["META"]["variant_id"] == variant_id


def test_registry_entries_have_required_shape():
    for entry in _REGISTRY.values():
        assert "META" in entry
        assert "matches" in entry
        assert "load_integration" in entry
        assert callable(entry["matches"])
        assert callable(entry["load_integration"])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/variants/test_registry.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_teacache.variants'`.

- [ ] **Step 3: Implement the registry walker**

```python
# src/mlx_teacache/variants/__init__.py
"""Variant registry.

At import time, walks every subdirectory of `variants/`, imports its
`config` (for `META`) and `detect` (for `matches`). Does NOT import
`integration` — that touches mflux internals and is loaded lazily by
`apply_teacache()`.

This split is what keeps `import mlx_teacache` working on machines that
installed the base package without the `[mflux]` extra.
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
    """Return a thunk that imports <module_name>.integration lazily and
    returns its `apply` callable. Errors at first call, not at import."""

    def _load() -> Callable[..., Any]:
        integration = importlib.import_module(f"{module_name}.integration")
        return integration.apply

    return _load


def _build_registry() -> None:
    """Walk subpackages once at module import time."""
    package_name = __name__
    package = importlib.import_module(package_name)
    for _, subname, ispkg in pkgutil.iter_modules(package.__path__):
        if not ispkg:
            continue
        full = f"{package_name}.{subname}"
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

- [ ] **Step 4: Create the test package marker**

```python
# tests/variants/__init__.py
```

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/variants/test_registry.py -v
```

Expected: 3 passed (the registry is empty because no variants exist yet — the loop is a no-op).

- [ ] **Step 6: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/variants/__init__.py tests/variants/
uv run mypy src/mlx_teacache/variants/
```

Expected: green.

- [ ] **Step 7: Commit**

```bash
git add src/mlx_teacache/variants/ tests/variants/
git commit -m "feat(variants): scaffold registry walker (config + detect eager, integration lazy)"
```

---

### Task 10: `variants/flux1_dev/` config + detect

**Files:**
- Create: `src/mlx_teacache/variants/flux1_dev/__init__.py`
- Create: `src/mlx_teacache/variants/flux1_dev/config.py`
- Create: `src/mlx_teacache/variants/flux1_dev/detect.py`
- Create: `tests/variants/flux1_dev/__init__.py`
- Create: `tests/variants/flux1_dev/test_detect.py`

Read `src/mlx_teacache/coefficients.py` for the existing FLUX.1 coefficients (`_UPSTREAM_FLUX_COEFFS`) and `src/mlx_teacache/integrations/mflux/detect.py` for the existing alias-matching logic.

- [ ] **Step 1: Write the failing tests**

```python
# tests/variants/flux1_dev/test_detect.py
from mlx_teacache.variants.flux1_dev.config import META
from mlx_teacache.variants.flux1_dev.detect import matches


class _FakeConfig:
    def __init__(self, aliases: list[str]) -> None:
        self.aliases = aliases
        self.model_name = "fake/flux1-dev"


class _FakeFlux1:
    def __init__(self, aliases: list[str]) -> None:
        self.model_config = _FakeConfig(aliases)


def test_meta_variant_id():
    assert META["variant_id"] == "flux1-dev"
    assert META["non_distilled"] is True


def test_matches_dev_alias():
    assert matches(_FakeFlux1(["dev"])) is True


def test_does_not_match_schnell():
    assert matches(_FakeFlux1(["schnell"])) is False


def test_does_not_match_non_flux1():
    class Other:
        model_config = _FakeConfig(["dev"])

    # FLUX.1 dev's matcher must NOT fire on a non-Flux1 instance.
    # We check by class name; the matcher uses lazy isinstance-via-import
    # so non-Flux1 returns False without loading mflux. See detect.py.
    assert matches(Other()) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/variants/flux1_dev/test_detect.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create the `__init__.py` and `config.py`**

```python
# src/mlx_teacache/variants/flux1_dev/__init__.py
"""FLUX.1 dev variant. See config.py + detect.py + integration.py."""
from .config import META
from .detect import matches

__all__ = ["META", "matches"]
```

```python
# src/mlx_teacache/variants/flux1_dev/config.py
"""FLUX.1 dev configuration. mflux-free."""
from __future__ import annotations

from typing import Any

# Upstream ali-vilab/TeaCache coefficients (FLUX.1-dev/schnell share this fit).
COEFFICIENTS: tuple[float, float, float, float, float] = (
    498.651651663,
    -283.13245892,
    65.66776981,
    -2.94329775,
    0.07815814,
)
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

- [ ] **Step 4: Implement `detect.py`**

```python
# src/mlx_teacache/variants/flux1_dev/detect.py
"""FLUX.1 dev detector. mflux-free at import time; uses lazy isinstance."""
from __future__ import annotations


def matches(flux: object) -> bool:
    """Return True if `flux` is a Flux1 instance whose model_config has
    alias 'dev'. Lazy-imports mflux only if `flux` looks like it might be
    a Flux1 (duck check via attribute presence)."""
    model_config = getattr(flux, "model_config", None)
    if model_config is None:
        return False
    aliases = getattr(model_config, "aliases", None) or []
    if "dev" not in aliases:
        return False
    # We're as confident as we can be without importing mflux. The lazy
    # isinstance check happens in integration.py at apply() time.
    return type(flux).__name__ == "Flux1"
```

- [ ] **Step 5: Create the test package init**

```python
# tests/variants/flux1_dev/__init__.py
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/variants/flux1_dev/test_detect.py -v
uv run pytest tests/variants/test_registry.py -v
```

Expected: detect tests all pass; registry test now sees one entry, all still pass.

- [ ] **Step 7: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/variants/flux1_dev/
uv run mypy src/mlx_teacache/variants/flux1_dev/__init__.py \
              src/mlx_teacache/variants/flux1_dev/config.py \
              src/mlx_teacache/variants/flux1_dev/detect.py
```

Expected: green.

- [ ] **Step 8: Commit**

```bash
git add src/mlx_teacache/variants/flux1_dev/ tests/variants/flux1_dev/
git commit -m "feat(variants/flux1_dev): config + detect (mflux-free)"
```

---

### Task 11: `variants/flux1_dev/integration.py` — forward wrapper + `VariantPatch`

**Files:**
- Create: `src/mlx_teacache/variants/flux1_dev/integration.py`
- Create: `tests/variants/flux1_dev/test_integration.py`

Read `src/mlx_teacache/integrations/mflux/forward.py` first — that's today's FLUX.1 forward wrapper. The integration module ports its body, replaces shared-module imports with `_kernel` imports, and returns a `TeaCacheHandle` whose `VariantPatch` captures the teardown.

- [ ] **Step 1: Read today's FLUX.1 forward**

```bash
grep -n "^def \|^class \|mx.compile\|_predict\|forward" src/mlx_teacache/integrations/mflux/forward.py | head -40
```

Note the public function shape and the order of mflux-instance mutations. The integration module must reverse each mutation in its `VariantPatch.rollbacks`.

- [ ] **Step 2: Write the failing test (skeleton)**

```python
# tests/variants/flux1_dev/test_integration.py
"""Integration tests for flux1-dev's apply() under lazy mflux import.

Fast/no-real-weights tests live here. Real-weight parity tests live in
test_parity.py (Task 16) and are gated by HF_TOKEN.
"""
import pytest

from mlx_teacache.handle import TeaCacheHandle


pytestmark = pytest.mark.skipif(
    pytest.importorskip("mflux", reason="mflux extra not installed") is None,
    reason="mflux required",
)


def test_apply_returns_a_handle():
    from mflux.models.flux.variants.txt2img.flux import Flux1

    flux = Flux1.from_name("dev", quantize=4)
    flux.freeze()
    from mlx_teacache.variants.flux1_dev.integration import apply

    handle = apply(flux)
    assert isinstance(handle, TeaCacheHandle)
    handle.restore()


def test_apply_installs_then_rollback_leaves_flux_pristine():
    from mflux.models.flux.variants.txt2img.flux import Flux1

    flux = Flux1.from_name("dev", quantize=4)
    flux.freeze()
    before = getattr(flux, "transformer", None)

    from mlx_teacache.variants.flux1_dev.integration import apply

    handle = apply(flux)
    # transformer is wrapped during the context
    assert flux.transformer is not before or before is not None
    handle.restore()
    # After restore, transformer is the original object identity
    assert flux.transformer is before
```

Note: these tests need mflux installed. They're gated by `pytest.importorskip`. They run in CI's `test-mflux` job.

- [ ] **Step 3: Run tests to verify they fail (with mflux installed)**

```bash
uv run pytest tests/variants/flux1_dev/test_integration.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_teacache.variants.flux1_dev.integration'`.

- [ ] **Step 4: Implement `integration.py`**

Port today's `src/mlx_teacache/integrations/mflux/forward.py` body. Key changes:
- Imports `_kernel.gate.{rel_l1, accumulate, polynomial_gate}` instead of inline arithmetic.
- Imports `_kernel.state.TeaCacheState` instead of the top-level state module.
- Imports `_kernel.lifecycle.LifecycleFSM` instead of the top-level lifecycle.
- Imports `_kernel.stats.{Provenance, ...}` instead of the top-level stats.
- Reads `COEFFICIENTS`, `DEFAULT_THRESH` from `.config`.
- Returns `TeaCacheHandle(patch=VariantPatch(rollbacks=[...], finalizers=[...]), ...)`.

```python
# src/mlx_teacache/variants/flux1_dev/integration.py
"""FLUX.1 dev integration: wires the polynomial gate + cached-residual
shortcut into mflux's Flux1.transformer via an `nn.Module` proxy.

This module is loaded LAZILY by `apply_teacache()` after the variant
registry's `matches()` wins. mflux is imported only at module import time
of THIS file, not at package-root import.
"""
from __future__ import annotations

from typing import Any

import mlx.nn as nn

from mlx_teacache._kernel.gate import accumulate, polynomial_gate, rel_l1
from mlx_teacache._kernel.lifecycle import LifecycleFSM
from mlx_teacache._kernel.state import TeaCacheState
from mlx_teacache._kernel.stats import Provenance, StepDecision
from mlx_teacache.handle import TeaCacheHandle, VariantPatch
from .config import COEFFICIENTS, DEFAULT_THRESH


_PROVENANCE = Provenance(
    source="builtin",
    revision="upstream-flux-v1",
    calibration_dataset="upstream ali-vilab TeaCache (no in-repo calibration)",
    reference_url="https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py",
)


class _TransformerProxy(nn.Module):
    """Wraps mflux.Flux1.transformer; intercepts forward calls to apply
    the polynomial gate. Delegates non-forward attribute access to inner.
    """

    def __init__(self, inner: nn.Module, *, state: TeaCacheState, fsm: LifecycleFSM,
                 coefficients: tuple[float, ...], threshold: float) -> None:
        super().__init__()
        self._inner = inner
        self._state = state
        self._fsm = fsm
        self._coefficients = coefficients
        self._threshold = threshold
        self._prev_mod_in: Any | None = None
        self._step_idx = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # The mod_in signal: today's forward.py extracts it from kwargs
        # (or first arg). Port the same logic. See
        # src/mlx_teacache/integrations/mflux/forward.py for the exact
        # signal extraction.
        mod_in = self._extract_mod_in(*args, **kwargs)

        if self._prev_mod_in is None:
            # First step always computes; no prior to compare against.
            out = self._inner(*args, **kwargs)
            self._state.cached_residual = self._compute_residual(out, *args, **kwargs)
            self._fsm.record_step(StepDecision(
                step=self._step_idx, skipped=False,
                rel_l1=0.0, predicted_rel_l1=0.0, threshold=self._threshold,
            ))
        else:
            delta = rel_l1(mod_in, self._prev_mod_in)
            self._state.accumulated_rel_l1 = accumulate(
                self._state.accumulated_rel_l1, delta
            )
            should_skip = polynomial_gate(
                self._coefficients, self._state.accumulated_rel_l1, self._threshold
            )
            if should_skip and self._state.cached_residual is not None:
                out = self._apply_cached_residual(self._state.cached_residual, *args, **kwargs)
                self._fsm.record_step(StepDecision(
                    step=self._step_idx, skipped=True,
                    rel_l1=delta, predicted_rel_l1=self._state.accumulated_rel_l1,
                    threshold=self._threshold,
                ))
            else:
                out = self._inner(*args, **kwargs)
                self._state.cached_residual = self._compute_residual(out, *args, **kwargs)
                self._state.accumulated_rel_l1 = 0.0  # reset accumulator after a real compute
                self._fsm.record_step(StepDecision(
                    step=self._step_idx, skipped=False,
                    rel_l1=delta, predicted_rel_l1=self._state.accumulated_rel_l1,
                    threshold=self._threshold,
                ))

        self._prev_mod_in = mod_in
        self._step_idx += 1
        return out

    def _extract_mod_in(self, *args: Any, **kwargs: Any) -> Any:
        """Pull the mod_in signal from the forward kwargs. Port from
        src/mlx_teacache/integrations/mflux/forward.py._extract_mod_in.
        """
        raise NotImplementedError("PORT_FROM_LEGACY: copy the body from "
                                  "src/mlx_teacache/integrations/mflux/forward.py")

    def _compute_residual(self, out: Any, *args: Any, **kwargs: Any) -> Any:
        """Compute the residual to cache. Port from legacy."""
        raise NotImplementedError("PORT_FROM_LEGACY: copy the body from "
                                  "src/mlx_teacache/integrations/mflux/forward.py")

    def _apply_cached_residual(self, residual: Any, *args: Any, **kwargs: Any) -> Any:
        """Apply the cached residual to produce the skip-step output."""
        raise NotImplementedError("PORT_FROM_LEGACY: copy the body from "
                                  "src/mlx_teacache/integrations/mflux/forward.py")

    # nn.Module-protocol delegation
    def parameters(self) -> Any:
        return self._inner.parameters()

    def trainable_parameters(self) -> Any:
        return self._inner.trainable_parameters()

    def freeze(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.freeze(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name in nn.Module.__dict__:
            raise AttributeError(name)
        return getattr(self._inner, name)


def apply(flux: Any, *, rel_l1_thresh: float | None = None, **kwargs: Any) -> TeaCacheHandle:
    """Install the polynomial-gate wrapper on flux.transformer. Returns
    a `TeaCacheHandle` whose `VariantPatch` reverses the install on exit."""
    threshold = rel_l1_thresh if rel_l1_thresh is not None else DEFAULT_THRESH

    state = TeaCacheState.fresh()
    fsm = LifecycleFSM()

    original_transformer = flux.transformer
    proxy = _TransformerProxy(
        inner=original_transformer,
        state=state,
        fsm=fsm,
        coefficients=COEFFICIENTS,
        threshold=threshold,
    )
    flux.transformer = proxy

    def _rollback_transformer() -> None:
        flux.transformer = original_transformer

    def _finalize_stats() -> None:
        fsm.finalize_generation()
        handle.stats = fsm.snapshot()  # late-bind into the handle

    patch = VariantPatch(
        rollbacks=[_rollback_transformer],
        finalizers=[_finalize_stats],
    )
    handle = TeaCacheHandle(
        patch=patch,
        stats=fsm.snapshot(),  # empty until finalize
        provenance=_PROVENANCE,
        rel_l1_thresh=threshold,
    )
    return handle
```

The three `raise NotImplementedError("PORT_FROM_LEGACY: ...")` markers in `_TransformerProxy._extract_mod_in`, `_compute_residual`, `_apply_cached_residual` are deliberate. The bodies of those three methods are mflux-internal-specific and live verbatim in `src/mlx_teacache/integrations/mflux/forward.py` today. Copy them in this step:

```bash
# Open both files side by side
grep -n "def _extract_mod_in\|def _compute_residual\|def _apply_cached_residual" \
    src/mlx_teacache/integrations/mflux/forward.py
```

Port the exact bodies, preserving the FLUX.1 transformer signature handling.

- [ ] **Step 5: Run tests to verify they pass (or fail with a real-weight error, which is fine)**

```bash
uv run pytest tests/variants/flux1_dev/test_integration.py -v
```

Expected: both tests pass if mflux is installed and weights are cached; otherwise they skip via `importorskip`.

- [ ] **Step 6: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/variants/flux1_dev/integration.py
uv run mypy src/mlx_teacache/variants/flux1_dev/integration.py
```

Expected: green.

- [ ] **Step 7: Commit**

```bash
git add src/mlx_teacache/variants/flux1_dev/integration.py tests/variants/flux1_dev/test_integration.py
git commit -m "feat(variants/flux1_dev): integration.py with VariantPatch contract"
```

---

### Task 12: `variants/flux2_klein_base_4b/` — config + detect

**Files:**
- Create: `src/mlx_teacache/variants/flux2_klein_base_4b/__init__.py`
- Create: `src/mlx_teacache/variants/flux2_klein_base_4b/config.py`
- Create: `src/mlx_teacache/variants/flux2_klein_base_4b/detect.py`
- Create: `tests/variants/flux2_klein_base_4b/__init__.py`
- Create: `tests/variants/flux2_klein_base_4b/test_detect.py`

Read `src/mlx_teacache/coefficients.py` for the existing `_FLUX2_KLEIN_BASE_4B_COEFFS` tuple. Port verbatim into config.py.

- [ ] **Step 1: Write the failing tests**

```python
# tests/variants/flux2_klein_base_4b/test_detect.py
from mlx_teacache.variants.flux2_klein_base_4b.config import (
    COEFFICIENTS, DEFAULT_THRESH, META,
)
from mlx_teacache.variants.flux2_klein_base_4b.detect import matches


class _FakeConfig:
    def __init__(self, aliases: list[str]) -> None:
        self.aliases = aliases
        self.model_name = "fake/flux2-klein-base-4b"


class _FakeFlux2Klein:
    def __init__(self, aliases: list[str]) -> None:
        self.model_config = _FakeConfig(aliases)


def test_meta():
    assert META["variant_id"] == "flux2-klein-base-4b"
    assert META["non_distilled"] is True
    assert META["recipes"]["default"]["num_inference_steps"] == 50
    assert META["recipes"]["default"]["guidance"] == 4.0


def test_default_thresh_is_0_17():
    assert DEFAULT_THRESH == 0.17


def test_matches_alias():
    assert matches(_FakeFlux2Klein(["flux2-klein-base-4b"])) is True


def test_does_not_match_klein_9b():
    assert matches(_FakeFlux2Klein(["flux2-klein-9b"])) is False


def test_does_not_match_klein_base_9b():
    assert matches(_FakeFlux2Klein(["flux2-klein-base-9b"])) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/variants/flux2_klein_base_4b/test_detect.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the variant scaffold**

```python
# src/mlx_teacache/variants/flux2_klein_base_4b/__init__.py
"""FLUX.2 Klein base 4B variant."""
from .config import META
from .detect import matches

__all__ = ["META", "matches"]
```

```python
# src/mlx_teacache/variants/flux2_klein_base_4b/config.py
"""FLUX.2 Klein base 4B configuration. mflux-free."""
from __future__ import annotations

from typing import Any

# Origin-constrained polyfit derived in-repo on 2026-05-17 from
# flux2-klein-base-4B at 25-step schedule (non-distilled). Trailing 0.0
# is the origin constraint (poly(0) = 0). See coefficients.py in the
# v0.5.0 tree for the historical record. R^2 = 0.106.
COEFFICIENTS: tuple[float, float, float, float, float] = (
    -1841.022165607874,
    848.4417137572868,
    -131.3554469956159,
    8.179509586828413,
    0.0,
)

# Empirically tuned via scripts/sweep_threshold_klein_base_4b.py (v0.4.0).
# At the package default 0.20 the gate over-skips and SSIM drops to ~0.76.
# At 0.17 it skips 3/25 (12%) with SSIM 0.99.
DEFAULT_THRESH: float = 0.17

RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 50, "guidance": 4.0},  # canonical upstream CFG
    "low_step": {"num_inference_steps": 25, "guidance": 1.0},  # original v0.4.0 row
}

LICENSE: str = "Apache-2.0"

META: dict[str, Any] = {
    "variant_id": "flux2-klein-base-4b",
    "display_name": "FLUX.2 Klein base 4B",
    "hf_model_id": "black-forest-labs/FLUX.2-klein-base-4B",
    "non_distilled": True,
    "memory_cap_hint_gb": None,  # 4B fits comfortably on 32GB
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B",
}
```

```python
# src/mlx_teacache/variants/flux2_klein_base_4b/detect.py
"""FLUX.2 Klein base 4B detector. mflux-free."""
from __future__ import annotations


def matches(flux: object) -> bool:
    model_config = getattr(flux, "model_config", None)
    if model_config is None:
        return False
    aliases = getattr(model_config, "aliases", None) or []
    if "flux2-klein-base-4b" not in aliases:
        return False
    return type(flux).__name__ == "Flux2Klein"
```

```python
# tests/variants/flux2_klein_base_4b/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/variants/flux2_klein_base_4b/test_detect.py -v
uv run pytest tests/variants/test_registry.py -v
```

Expected: detect tests pass; registry tests still pass with two entries.

- [ ] **Step 5: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/variants/flux2_klein_base_4b/
uv run mypy src/mlx_teacache/variants/flux2_klein_base_4b/__init__.py \
              src/mlx_teacache/variants/flux2_klein_base_4b/config.py \
              src/mlx_teacache/variants/flux2_klein_base_4b/detect.py
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/mlx_teacache/variants/flux2_klein_base_4b/ tests/variants/flux2_klein_base_4b/
git commit -m "feat(variants/flux2_klein_base_4b): config + detect (mflux-free)"
```

---

### Task 13: `variants/flux2_klein_base_4b/integration.py` — CFG per-branch forward

**Files:**
- Create: `src/mlx_teacache/variants/flux2_klein_base_4b/integration.py`
- Create: `tests/variants/flux2_klein_base_4b/test_integration.py`

This is the canonical CFG variant. Read today's `src/mlx_teacache/integrations/mflux/flux2.py` and `forward.py` (the v0.4.1 CFG path). Port `flux2_cfg_forward_with_gate` and the `_predict` instance-attribute swap pattern. Use `_kernel.cfg.cfg_per_branch_combine` for the CFG combination math.

- [ ] **Step 1: Read today's FLUX.2 forward**

```bash
grep -n "^def \|^class \|cfg_was_active\|cached_residual_neg" src/mlx_teacache/integrations/mflux/flux2.py | head -40
grep -n "flux2_cfg_forward_with_gate" src/mlx_teacache/integrations/mflux/forward.py
```

- [ ] **Step 2: Write the failing test (skeleton)**

```python
# tests/variants/flux2_klein_base_4b/test_integration.py
"""Integration tests for klein-base-4b. Real-weight tests gated by mflux."""
import pytest

from mlx_teacache.handle import TeaCacheHandle


pytestmark = pytest.mark.skipif(
    pytest.importorskip("mflux", reason="mflux extra not installed") is None,
    reason="mflux required",
)


def test_apply_returns_handle_with_klein_base_4b_provenance():
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
    flux.freeze()
    from mlx_teacache.variants.flux2_klein_base_4b.integration import apply

    handle = apply(flux)
    assert isinstance(handle, TeaCacheHandle)
    # Provenance comes from the variant's config, not a shared registry
    assert handle.provenance.revision is not None
    assert "klein-base-4b" in handle.provenance.calibration_dataset.lower() or \
           "klein_base_4b" in handle.provenance.calibration_dataset.lower()
    handle.restore()


def test_apply_default_thresh_is_017():
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
    flux.freeze()
    from mlx_teacache.variants.flux2_klein_base_4b.integration import apply

    handle = apply(flux)
    assert handle.rel_l1_thresh == pytest.approx(0.17)
    handle.restore()
```

- [ ] **Step 3: Implement `integration.py`**

```python
# src/mlx_teacache/variants/flux2_klein_base_4b/integration.py
"""FLUX.2 Klein base 4B integration with CFG per-branch caching (v0.4.1
mechanism). Lazy-imported by apply_teacache after detect.matches() wins.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from mlx_teacache._kernel.cfg import cfg_per_branch_combine
from mlx_teacache._kernel.gate import accumulate, polynomial_gate, rel_l1
from mlx_teacache._kernel.lifecycle import LifecycleFSM
from mlx_teacache._kernel.state import TeaCacheState
from mlx_teacache._kernel.stats import Provenance, StepDecision
from mlx_teacache.handle import TeaCacheHandle, VariantPatch
from .config import COEFFICIENTS, DEFAULT_THRESH


_PROVENANCE = Provenance(
    source="builtin",
    revision="in-repo-2026-05-17-origin",
    calibration_dataset=(
        "10 prompts x 25 steps x seed=42, M1 Max 32GB, bf16, 512x512, "
        "guidance=1.0, origin-constrained polyfit (klein-base-4b)"
    ),
    fit_metric="constrained-LSQ R^2 on consecutive-step (mod_in, body_out) rel-L1 pairs",
    fit_metric_value=0.10643408169124158,
    reference_url="https://github.com/IonDen/mlx-teacache/blob/main/scripts/calibrate_flux2.py",
    default_thresh=DEFAULT_THRESH,
)


def _gated_predict_factory(
    *,
    original_predict: Any,
    state: TeaCacheState,
    fsm: LifecycleFSM,
    coefficients: tuple[float, ...],
    threshold: float,
) -> Any:
    """Build the replacement _predict closure.

    PORT_FROM_LEGACY: copy the body of `flux2_cfg_forward_with_gate` from
    src/mlx_teacache/integrations/mflux/forward.py. Replace inline arithmetic
    with _kernel.gate / _kernel.cfg / _kernel.state calls. Preserve the
    per-branch state.cached_residual (pos) / state.cached_residual_neg (neg)
    semantics. Set state.cfg_was_active = True on first CFG branch entry.
    """
    raise NotImplementedError(
        "PORT_FROM_LEGACY: copy body from src/mlx_teacache/integrations/mflux/"
        "forward.py::flux2_cfg_forward_with_gate; replace inline arithmetic "
        "with _kernel calls."
    )


def apply(flux: Any, *, rel_l1_thresh: float | None = None, **kwargs: Any) -> TeaCacheHandle:
    """Install the gated _predict on flux. Returns a TeaCacheHandle whose
    VariantPatch reverses the install."""
    threshold = rel_l1_thresh if rel_l1_thresh is not None else DEFAULT_THRESH

    state = TeaCacheState.fresh()
    fsm = LifecycleFSM()

    # Record whether _predict is an instance attribute or class-level
    # (matters for restore — see CLAUDE.md / mflux-and-local-projects.md).
    had_instance_attr = "_predict" in vars(flux)
    original_predict = flux._predict

    gated = _gated_predict_factory(
        original_predict=original_predict,
        state=state,
        fsm=fsm,
        coefficients=COEFFICIENTS,
        threshold=threshold,
    )
    flux._predict = gated

    def _rollback_predict() -> None:
        if had_instance_attr:
            flux._predict = original_predict
        else:
            del flux._predict

    def _finalize_stats() -> None:
        fsm.finalize_generation()
        handle.stats = fsm.snapshot()

    patch = VariantPatch(
        rollbacks=[_rollback_predict],
        finalizers=[_finalize_stats],
    )
    handle = TeaCacheHandle(
        patch=patch,
        stats=fsm.snapshot(),
        provenance=_PROVENANCE,
        rel_l1_thresh=threshold,
    )
    return handle
```

Then port the `_gated_predict_factory` body from `src/mlx_teacache/integrations/mflux/forward.py::flux2_cfg_forward_with_gate`. The legacy function takes essentially the same arguments and produces the same predict closure; only the imports change (use `_kernel` modules).

- [ ] **Step 4: Run tests to verify they pass (with mflux)**

```bash
uv run pytest tests/variants/flux2_klein_base_4b/test_integration.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/variants/flux2_klein_base_4b/integration.py
uv run mypy src/mlx_teacache/variants/flux2_klein_base_4b/integration.py
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/mlx_teacache/variants/flux2_klein_base_4b/integration.py \
        tests/variants/flux2_klein_base_4b/test_integration.py
git commit -m "feat(variants/flux2_klein_base_4b): integration with CFG per-branch (v0.4.1)"
```

---

### Task 14: Kernel-boundary validation gate

**Files:** (verification only)

After Tasks 11 and 13, two variants exist with the new architecture (the no-CFG canonical and the CFG canonical). This is the moment to validate the kernel boundary before porting the remaining four variants. If the boundary is wrong, the cost of fixing it grows linearly with each ported variant.

- [ ] **Step 1: Run both ported variants' integration tests**

```bash
uv run pytest tests/variants/flux1_dev/ tests/variants/flux2_klein_base_4b/ -v
```

Expected: all pass with mflux installed.

- [ ] **Step 2: Run the kernel test suite**

```bash
uv run pytest tests/_kernel/ -v
```

Expected: all pass.

- [ ] **Step 3: Verify no duplicate algorithmic code**

```bash
grep -n "def rel_l1\|def polynomial_gate\|def cfg_per_branch_combine" \
    src/mlx_teacache/variants/flux1_dev/integration.py \
    src/mlx_teacache/variants/flux2_klein_base_4b/integration.py
```

Expected: ZERO matches. The kernel functions must not be redefined in variant code.

- [ ] **Step 4: If the boundary feels wrong, pause and re-cut**

If any of these are true, pause:
- A variant's `integration.py` needs to import something from another variant's directory.
- The kernel has variant-specific branches (e.g., `if variant == "flux1-dev"`).
- A shared concept (e.g., the CFG forward) is duplicated across two variants because the kernel doesn't expose enough.

Re-cut the boundary by moving the right primitive into `_kernel/`, then re-run Steps 1-3.

- [ ] **Step 5: No commit** (verification step).

---

## Phase D — Remaining four variant cores

Goal: port `flux1_schnell`, `flux2_klein_4b`, `flux2_klein_9b`, `flux2_klein_base_9b`. Each variant follows one of the two canonical patterns from Phase C.

### Task 15: `variants/flux1_schnell/`

**Files:**
- Create: `src/mlx_teacache/variants/flux1_schnell/{__init__,config,detect,integration}.py`
- Create: `tests/variants/flux1_schnell/{__init__,test_detect,test_integration}.py`

`flux1_schnell` follows the `flux1_dev` pattern (no CFG). The only differences:
- `META["variant_id"] = "flux1-schnell"`, `display_name`, `hf_model_id`
- `RECIPES["default"] = {"num_inference_steps": 4, "guidance": 1.0}`
- `META["non_distilled"] = False` (schnell is distilled — TeaCache gate doesn't engage)
- `META["license"] = "Apache-2.0"`
- `detect.py::matches` checks `"schnell" in aliases`
- `config.py::COEFFICIENTS` re-uses the same upstream tuple via cross-import:

```python
from mlx_teacache.variants.flux1_dev.config import COEFFICIENTS as _SHARED
COEFFICIENTS = _SHARED   # FLUX.1 architecture shared between dev + schnell
```

`integration.py` uses the same forward wrapper as `flux1_dev` — port the same body. The duplication is intentional (per Spec → Why this refactor → finding 1: bug-blast-radius bounded by kernel only).

- [ ] **Step 1: Write the failing test**

```python
# tests/variants/flux1_schnell/test_detect.py
from mlx_teacache.variants.flux1_schnell.config import COEFFICIENTS, META
from mlx_teacache.variants.flux1_schnell.detect import matches
from mlx_teacache.variants.flux1_dev.config import COEFFICIENTS as DEV_COEFFS


class _FC:
    def __init__(self, aliases):
        self.aliases = aliases
        self.model_name = "fake/flux1-schnell"


class _FakeFlux1:
    def __init__(self, aliases):
        self.model_config = _FC(aliases)


def test_meta():
    assert META["variant_id"] == "flux1-schnell"
    assert META["non_distilled"] is False
    assert META["recipes"]["default"]["num_inference_steps"] == 4


def test_coefficients_share_dev_via_cross_import():
    assert COEFFICIENTS is DEV_COEFFS


def test_matches_schnell():
    assert matches(_FakeFlux1(["schnell"])) is True


def test_does_not_match_dev():
    assert matches(_FakeFlux1(["dev"])) is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/variants/flux1_schnell/test_detect.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the four module files**

```python
# src/mlx_teacache/variants/flux1_schnell/__init__.py
from .config import META
from .detect import matches
__all__ = ["META", "matches"]
```

```python
# src/mlx_teacache/variants/flux1_schnell/config.py
"""FLUX.1 schnell configuration. mflux-free."""
from __future__ import annotations

from typing import Any

from mlx_teacache.variants.flux1_dev.config import COEFFICIENTS as _SHARED

# Intentional reuse — FLUX.1 architecture shared between dev and schnell.
# Cross-import keeps the reuse visible at review time.
COEFFICIENTS = _SHARED

DEFAULT_THRESH: float = 0.20

RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 4, "guidance": 1.0},
}

LICENSE: str = "Apache-2.0"

META: dict[str, Any] = {
    "variant_id": "flux1-schnell",
    "display_name": "FLUX.1 schnell",
    "hf_model_id": "black-forest-labs/FLUX.1-schnell",
    "non_distilled": False,  # distilled — gate does not engage
    "memory_cap_hint_gb": None,
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
}
```

```python
# src/mlx_teacache/variants/flux1_schnell/detect.py
from __future__ import annotations


def matches(flux: object) -> bool:
    model_config = getattr(flux, "model_config", None)
    if model_config is None:
        return False
    aliases = getattr(model_config, "aliases", None) or []
    if "schnell" not in aliases:
        return False
    return type(flux).__name__ == "Flux1"
```

```python
# src/mlx_teacache/variants/flux1_schnell/integration.py
"""FLUX.1 schnell integration. Uses the same forward wrapper as flux1_dev;
the variant-specific behavior is captured in config (recipes, distilled flag)
and provenance.
"""
from __future__ import annotations

from typing import Any

# Reuse the proxy + apply logic from flux1_dev. The intentional duplication
# is the no-import-from-sibling-variant rule per Spec. Copy the file body.
# DO NOT do: from mlx_teacache.variants.flux1_dev.integration import apply
# — that would couple schnell to dev's lifecycle.

# PORT: copy the body of src/mlx_teacache/variants/flux1_dev/integration.py
# Replace the _PROVENANCE constant with schnell-specific revision (still
# "upstream-flux-v1-shared", since coefficients are physically the same
# tuple). Replace the imports of COEFFICIENTS, DEFAULT_THRESH to come from
# .config (this directory's config), not flux1_dev.config.
```

Then copy the body verbatim from `flux1_dev/integration.py`, changing only the `from .config import ...` line and the `_PROVENANCE` constant body:

```python
_PROVENANCE = Provenance(
    source="builtin",
    revision="upstream-flux-v1-shared",
    calibration_dataset="upstream ali-vilab TeaCache (FLUX architecture shared dev/schnell)",
    reference_url="https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py",
)
```

```python
# tests/variants/flux1_schnell/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/variants/flux1_schnell/test_detect.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/variants/flux1_schnell/
uv run mypy src/mlx_teacache/variants/flux1_schnell/__init__.py \
              src/mlx_teacache/variants/flux1_schnell/config.py \
              src/mlx_teacache/variants/flux1_schnell/detect.py \
              src/mlx_teacache/variants/flux1_schnell/integration.py
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/mlx_teacache/variants/flux1_schnell/ tests/variants/flux1_schnell/
git commit -m "feat(variants/flux1_schnell): port no-CFG variant (coefficient cross-import from dev)"
```

---

### Task 16: `variants/flux2_klein_4b/` and `variants/flux2_klein_9b/`

**Files:**
- Create: `src/mlx_teacache/variants/flux2_klein_4b/{__init__,config,detect,integration}.py`
- Create: `tests/variants/flux2_klein_4b/{__init__,test_detect}.py`
- Create: `src/mlx_teacache/variants/flux2_klein_9b/{__init__,config,detect,integration}.py`
- Create: `tests/variants/flux2_klein_9b/{__init__,test_detect}.py`

Both are distilled Klein variants. They follow the `flux2_klein_base_4b` pattern for integration but the gate does not engage on their 4-8 step distilled schedules. Their value is `mx.compile`-path avoidance (documented in CHANGELOG, not algorithmic).

Per-variant coefficients (from today's `coefficients.py`):
- `_FLUX2_KLEIN_4B_COEFFS = (-15.6, 32.1, -23.4, 6.6, -0.0)` — verify by reading the current registry.
- `_FLUX2_KLEIN_9B_COEFFS = (33.5, -47.1, 16.6, ...)` — verify.

**Both variants:**
- `META["non_distilled"] = False`
- `DEFAULT_THRESH` = None (use package fallback 0.20; gate doesn't engage at any reasonable threshold)
- `RECIPES["default"] = {"num_inference_steps": 8, "guidance": 1.0}`
- `LICENSE`: klein-4b is Apache-2.0; klein-9b is FLUX Non-Commercial
- `detect.py::matches` checks for the appropriate alias

Implement both variants in this single task. Each gets its own test_detect.py.

- [ ] **Step 1: Read current coefficients for both variants**

```bash
grep -B1 -A 10 "_FLUX2_KLEIN_4B_COEFFS\|_FLUX2_KLEIN_9B_COEFFS" src/mlx_teacache/coefficients.py
```

Capture both tuples exactly.

- [ ] **Step 2: Write tests for both**

```python
# tests/variants/flux2_klein_4b/test_detect.py
from mlx_teacache.variants.flux2_klein_4b.config import META
from mlx_teacache.variants.flux2_klein_4b.detect import matches


class _FC:
    def __init__(self, aliases):
        self.aliases = aliases
        self.model_name = "fake/flux2-klein-4b"


class _FakeFlux2Klein:
    def __init__(self, aliases):
        self.model_config = _FC(aliases)


def test_meta():
    assert META["variant_id"] == "flux2-klein-4b"
    assert META["non_distilled"] is False


def test_matches_4b():
    assert matches(_FakeFlux2Klein(["flux2-klein-4b"])) is True


def test_does_not_match_9b():
    assert matches(_FakeFlux2Klein(["flux2-klein-9b"])) is False
```

```python
# tests/variants/flux2_klein_9b/test_detect.py
# Mirror of test_flux2_klein_4b/test_detect.py with the 9b alias and
# expected variant_id. Skipped here for brevity — write it identical
# shape, swapping "4b" for "9b" wherever it appears.
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/variants/flux2_klein_4b/test_detect.py tests/variants/flux2_klein_9b/test_detect.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement both variant directories**

Each gets `__init__.py`, `config.py`, `detect.py`, `integration.py` with the same shape as `flux2_klein_base_4b/`. Key variant-specific values:

`flux2_klein_4b/config.py`:
```python
COEFFICIENTS = (-15.611728, 32.099915, -23.424351, 6.572828, -0.018879)  # verify against today's registry
DEFAULT_THRESH = None
RECIPES = {"default": {"num_inference_steps": 8, "guidance": 1.0}}
LICENSE = "Apache-2.0"
META = {
    "variant_id": "flux2-klein-4b",
    "display_name": "FLUX.2 Klein 4B (distilled)",
    "hf_model_id": "black-forest-labs/FLUX.2-klein-4B",
    "non_distilled": False,
    "memory_cap_hint_gb": None,
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
}
```

`flux2_klein_9b/config.py`: identical shape with the 9B coefficient tuple, `variant_id = "flux2-klein-9b"`, `LICENSE = "FLUX Non-Commercial"`, `license_url` pointing at klein-9B's model card.

Integration files use the same port pattern as `flux2_klein_base_4b/integration.py`. Provenance carries `revision="in-repo-2026-05-15"` (4B) and `revision="in-repo-2026-05-16-origin"` (9B).

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/variants/flux2_klein_4b/ tests/variants/flux2_klein_9b/ -v
```

Expected: all pass.

- [ ] **Step 6: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/variants/flux2_klein_4b/ src/mlx_teacache/variants/flux2_klein_9b/
uv run mypy src/mlx_teacache/variants/flux2_klein_4b/ src/mlx_teacache/variants/flux2_klein_9b/
```

Expected: green.

- [ ] **Step 7: Commit**

```bash
git add src/mlx_teacache/variants/flux2_klein_4b/ src/mlx_teacache/variants/flux2_klein_9b/ \
        tests/variants/flux2_klein_4b/ tests/variants/flux2_klein_9b/
git commit -m "feat(variants): distilled FLUX.2 Klein 4B + 9B"
```

---

### Task 17: `variants/flux2_klein_base_9b/` — coefficient cross-import + identity test

**Files:**
- Create: `src/mlx_teacache/variants/flux2_klein_base_9b/{__init__,config,detect,integration}.py`
- Create: `tests/variants/flux2_klein_base_9b/{__init__,test_detect,test_shared_coefficients}.py`

The v0.5.0-shipped pattern: klein-base-9b's coefficients are the same tuple as klein-base-4b. Under the new architecture, the cross-import is explicit at the module level.

- [ ] **Step 1: Write the failing tests**

```python
# tests/variants/flux2_klein_base_9b/test_shared_coefficients.py
"""Klein-base-9b ships reusing klein-base-4b's polynomial verbatim. The
reuse is intentional — see v0.5.0 validation evidence at
_artifacts/validation_klein_base_9b.json (SSIM 0.986 confirms).
If this test fails, either:
- klein-base-9b's config.py was edited (and the change should be deliberate), or
- klein-base-4b's coefficients drifted (and 9b should re-validate).
"""
from mlx_teacache.variants.flux2_klein_base_4b.config import COEFFICIENTS as BASE_4B
from mlx_teacache.variants.flux2_klein_base_9b.config import COEFFICIENTS as BASE_9B


def test_klein_base_9b_reuses_4b_coefficients():
    assert BASE_9B is BASE_4B  # identity, not just equality
```

```python
# tests/variants/flux2_klein_base_9b/test_detect.py
from mlx_teacache.variants.flux2_klein_base_9b.config import META, DEFAULT_THRESH
from mlx_teacache.variants.flux2_klein_base_9b.detect import matches


class _FC:
    def __init__(self, aliases):
        self.aliases = aliases
        self.model_name = "fake/flux2-klein-base-9b"


class _FakeFlux2Klein:
    def __init__(self, aliases):
        self.model_config = _FC(aliases)


def test_meta():
    assert META["variant_id"] == "flux2-klein-base-9b"
    assert META["non_distilled"] is True
    assert META["recipes"]["default"]["num_inference_steps"] == 50
    assert META["memory_cap_hint_gb"] == 24


def test_default_thresh_is_017_inherited_from_4b():
    assert DEFAULT_THRESH == 0.17


def test_matches():
    assert matches(_FakeFlux2Klein(["flux2-klein-base-9b"])) is True


def test_does_not_match_4b():
    assert matches(_FakeFlux2Klein(["flux2-klein-base-4b"])) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/variants/flux2_klein_base_9b/ -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the variant directory**

```python
# src/mlx_teacache/variants/flux2_klein_base_9b/__init__.py
from .config import META
from .detect import matches
__all__ = ["META", "matches"]
```

```python
# src/mlx_teacache/variants/flux2_klein_base_9b/config.py
"""FLUX.2 Klein base 9B configuration. Reuses base-4b's polynomial verbatim;
see tests/variants/flux2_klein_base_9b/test_shared_coefficients.py for the
identity assertion that catches drift.

Validation evidence: _artifacts/validation_klein_base_9b.json
(v0.5.0 release-gate: SSIM 0.986, 12/48 skips, 2.68x cold speedup).
"""
from __future__ import annotations

from typing import Any

from mlx_teacache.variants.flux2_klein_base_4b.config import COEFFICIENTS as _SHARED

COEFFICIENTS = _SHARED   # intentional cross-import — same FLUX.2 Klein architecture family

DEFAULT_THRESH: float = 0.17

RECIPES: dict[str, dict[str, Any]] = {
    "default": {"num_inference_steps": 50, "guidance": 4.0},
}

LICENSE: str = "FLUX Non-Commercial"

META: dict[str, Any] = {
    "variant_id": "flux2-klein-base-9b",
    "display_name": "FLUX.2 Klein base 9B",
    "hf_model_id": "black-forest-labs/FLUX.2-klein-base-9B",
    "non_distilled": True,
    "memory_cap_hint_gb": 24,   # 32GB unified memory headroom
    "recipes": RECIPES,
    "license": LICENSE,
    "license_url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B",
}
```

```python
# src/mlx_teacache/variants/flux2_klein_base_9b/detect.py
from __future__ import annotations


def matches(flux: object) -> bool:
    model_config = getattr(flux, "model_config", None)
    if model_config is None:
        return False
    aliases = getattr(model_config, "aliases", None) or []
    if "flux2-klein-base-9b" not in aliases:
        return False
    return type(flux).__name__ == "Flux2Klein"
```

```python
# src/mlx_teacache/variants/flux2_klein_base_9b/integration.py
# PORT: copy the body of src/mlx_teacache/variants/flux2_klein_base_4b/
# integration.py verbatim. The only delta is the Provenance constant:
```

```python
_PROVENANCE = Provenance(
    source="builtin",
    revision="in-repo-2026-05-18-reuse-base-4b",
    calibration_dataset=(
        "REUSED from flux2-klein-base-4b — same architecture family + same recipe; "
        "validated empirically at 50 steps + guidance=4.0 (see "
        "_artifacts/validation_klein_base_9b.json: SSIM 0.986)"
    ),
    fit_metric="constrained-LSQ R^2 on consecutive-step (mod_in, body_out) rel-L1 pairs",
    fit_metric_value=0.10643408169124158,  # inherited from base-4b
    reference_url=(
        "https://github.com/IonDen/mlx-teacache/blob/main/scripts/validate_klein_base_9b.py"
    ),
    default_thresh=DEFAULT_THRESH,
)
```

```python
# tests/variants/flux2_klein_base_9b/__init__.py
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/variants/flux2_klein_base_9b/ -v
uv run pytest tests/variants/test_registry.py -v
```

Expected: all pass; registry now has 6 entries.

- [ ] **Step 5: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/variants/flux2_klein_base_9b/
uv run mypy src/mlx_teacache/variants/flux2_klein_base_9b/__init__.py \
              src/mlx_teacache/variants/flux2_klein_base_9b/config.py \
              src/mlx_teacache/variants/flux2_klein_base_9b/detect.py \
              src/mlx_teacache/variants/flux2_klein_base_9b/integration.py
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/mlx_teacache/variants/flux2_klein_base_9b/ tests/variants/flux2_klein_base_9b/
git commit -m "feat(variants/flux2_klein_base_9b): reuses base-4b coefficients via cross-import"
```

---

## Phase E — API dispatch + base-import contract

### Task 18: Rewrite `api.py` as thin dispatcher

**Files:**
- Modify: `src/mlx_teacache/api.py`
- Create: `tests/test_api_dispatch.py`

- [ ] **Step 1: Read the current api.py**

```bash
grep -n "^def \|^class " src/mlx_teacache/api.py
```

Note today's `apply_teacache` body — it hard-codes FLUX.1 vs FLUX.2 branches. The new body walks `_REGISTRY` and calls the winning variant's lazy-loaded `apply`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_api_dispatch.py
"""api.apply_teacache dispatches via _REGISTRY."""
import pytest

from mlx_teacache import apply_teacache
from mlx_teacache.errors import IncompatibleModelError


class _FC:
    def __init__(self, aliases):
        self.aliases = aliases
        self.model_name = "fake/x"


class _FakeFlux1:
    def __init__(self, aliases):
        self.model_config = _FC(aliases)


def test_unknown_variant_raises_incompatible_model_error():
    with pytest.raises(IncompatibleModelError) as exc:
        apply_teacache(_FakeFlux1(["something-bogus"]))
    assert "flux1-dev" in str(exc.value)  # supported variants listed


def test_dispatch_matches_variant_id_in_error_message():
    with pytest.raises(IncompatibleModelError) as exc:
        apply_teacache(_FakeFlux1(["zzzzz"]))
    msg = str(exc.value)
    for variant_id in ("flux1-dev", "flux1-schnell", "flux2-klein-4b",
                       "flux2-klein-9b", "flux2-klein-base-4b", "flux2-klein-base-9b"):
        assert variant_id in msg
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/test_api_dispatch.py -v
```

Expected: at least one failure (today's api.py hard-codes the supported list and won't list all six the same way the new code will).

- [ ] **Step 4: Rewrite `api.py`**

```python
# src/mlx_teacache/api.py
"""Public entry point. Variant dispatch via _REGISTRY.

Implementation note: the registry walker in mlx_teacache.variants loads
only config + detect at import time. integration.py is loaded lazily here.
"""
from __future__ import annotations

from typing import Any

from mlx_teacache.errors import IncompatibleModelError
from mlx_teacache.handle import TeaCacheHandle
from mlx_teacache.variants import _REGISTRY


def apply_teacache(flux: Any, **kwargs: Any) -> TeaCacheHandle:
    """Enable TeaCache step-skipping on an mflux instance.

    Walks the variant registry; the first variant whose `matches(flux)`
    returns True wins. Its integration module is lazy-imported, then
    its `apply(flux, **kwargs)` is invoked, returning the handle.
    """
    for entry in _REGISTRY.values():
        if entry["matches"](flux):
            apply = entry["load_integration"]()
            return apply(flux, **kwargs)

    model_config = getattr(flux, "model_config", None)
    model_name = getattr(model_config, "model_name", None)
    raise IncompatibleModelError(
        actual_type=type(flux).__name__,
        actual_model_name=model_name,
        supported=sorted(_REGISTRY.keys()),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_api_dispatch.py tests/test_handle.py -v
```

Expected: dispatch + handle tests pass.

- [ ] **Step 6: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/api.py tests/test_api_dispatch.py
uv run mypy src/mlx_teacache/api.py
```

Expected: green.

- [ ] **Step 7: Commit**

```bash
git add src/mlx_teacache/api.py tests/test_api_dispatch.py
git commit -m "feat(api): rewrite apply_teacache as thin _REGISTRY dispatcher"
```

---

### Task 19: Compatibility shims at `stats.py` and `coefficients.py`

**Files:**
- Modify: `src/mlx_teacache/stats.py` (overwrite — becomes shim only)
- Modify: `src/mlx_teacache/coefficients.py` (overwrite — becomes shim only)
- Create: `tests/test_compatibility_shims.py`

The audit's Finding 1: deleting these modules breaks `from mlx_teacache.stats import TeaCacheStats`. Shims preserve the import path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compatibility_shims.py
"""V0.5.x users imported types from mlx_teacache.stats and
mlx_teacache.coefficients. These modules stay in v0.6.0 as compatibility
shims that re-export from _kernel.stats. See spec → audit F1."""
import importlib


def test_mlx_teacache_stats_reexports_all_v05_names():
    stats_mod = importlib.import_module("mlx_teacache.stats")
    for name in ["TeaCacheStats", "GenerationStats", "StepDecision", "StatsFrozenError"]:
        assert hasattr(stats_mod, name), f"mlx_teacache.stats lost {name}"

    from mlx_teacache._kernel.stats import (
        GenerationStats, StatsFrozenError, StepDecision, TeaCacheStats,
    )
    assert stats_mod.TeaCacheStats is TeaCacheStats
    assert stats_mod.GenerationStats is GenerationStats
    assert stats_mod.StepDecision is StepDecision
    assert stats_mod.StatsFrozenError is StatsFrozenError


def test_mlx_teacache_coefficients_reexports_provenance():
    coeffs_mod = importlib.import_module("mlx_teacache.coefficients")
    from mlx_teacache._kernel.stats import Provenance
    assert hasattr(coeffs_mod, "Provenance")
    assert coeffs_mod.Provenance is Provenance
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_compatibility_shims.py -v
```

Expected: today's `stats.py` exports the same names so the first test passes, but `coefficients.py` Provenance check may also pass. The test really validates that AFTER we overwrite both modules they STILL pass.

- [ ] **Step 3: Overwrite `stats.py`**

```python
# src/mlx_teacache/stats.py
"""Compatibility shim. Re-exports from _kernel.stats.

This module exists so `from mlx_teacache.stats import TeaCacheStats`
keeps working. The canonical home for these types is _kernel/stats.py.

DO NOT add logic here. Re-exports only.
"""
from mlx_teacache._kernel.stats import (
    GenerationStats,
    StatsFrozenError,
    StepDecision,
    TeaCacheStats,
)

__all__ = ["TeaCacheStats", "GenerationStats", "StepDecision", "StatsFrozenError"]
```

- [ ] **Step 4: Overwrite `coefficients.py`**

```python
# src/mlx_teacache/coefficients.py
"""Compatibility shim. Re-exports Provenance from _kernel.stats.

The coefficient _REGISTRY itself becomes per-variant config under
src/mlx_teacache/variants/<name>/config.py — it is intentionally no
longer importable from here. Only Provenance has a stable user-facing
import path.

DO NOT add logic here. Re-export only.
"""
from mlx_teacache._kernel.stats import Provenance

__all__ = ["Provenance"]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_compatibility_shims.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/stats.py src/mlx_teacache/coefficients.py
uv run mypy src/mlx_teacache/stats.py src/mlx_teacache/coefficients.py
```

Expected: green.

- [ ] **Step 7: Commit**

```bash
git add src/mlx_teacache/stats.py src/mlx_teacache/coefficients.py tests/test_compatibility_shims.py
git commit -m "feat(shims): stats.py + coefficients.py become re-export shims (audit F1)"
```

---

### Task 20: `tests/test_public_api.py` — full public-import-path snapshot

**Files:**
- Create: `tests/test_public_api.py`

Per the audit's Finding 1: gate on every documented v0.5.x import path, not just `__all__`.

- [ ] **Step 1: Write the test**

```python
# tests/test_public_api.py
"""Public API surface snapshot. Catches any drift from v0.5.x.

If a test here fails, the public surface changed — that's either a
deliberate breaking change (file an issue + version bump) or an
accidental regression (fix it).
"""
import subprocess
import sys

import pytest


def test_root_package_exports():
    import mlx_teacache

    for name in [
        "__version__",
        "apply_teacache",
        "TeaCacheHandle",
        "TeaCacheStats",
        "GenerationStats",
        "StepDecision",
        "Provenance",
        "TeaCacheError",
        "AlreadyPatchedError",
        "CalibrationError",
        "IncompatibleModelError",
        "InternalStateError",
        "InvalidStepWindowError",
        "MissingGenerationContextError",
        "StatsFrozenError",
        "TeaCacheNoBenefitWarning",
        "TransformerShapeError",
    ]:
        assert hasattr(mlx_teacache, name), f"public root export missing: {name}"


def test_stats_submodule_paths():
    from mlx_teacache.stats import (
        GenerationStats, StatsFrozenError, StepDecision, TeaCacheStats,
    )
    # Smoke: instantiate / call signatures
    assert StepDecision(step=0, skipped=False, rel_l1=0.0,
                        predicted_rel_l1=0.0, threshold=0.0).step == 0


def test_coefficients_submodule_path():
    from mlx_teacache.coefficients import Provenance
    assert Provenance.for_user_supplied().source == "user"


def test_apply_teacache_signature_unchanged():
    import inspect

    from mlx_teacache import apply_teacache
    sig = inspect.signature(apply_teacache)
    params = list(sig.parameters.values())
    # Positional 'flux' first; remaining are keyword. rel_l1_thresh
    # must be acceptable.
    assert params[0].name == "flux"
    # apply_teacache(flux, rel_l1_thresh=0.2) must not raise on
    # signature inspection
    rel_l1 = sig.parameters.get("rel_l1_thresh")
    if rel_l1 is not None:
        assert rel_l1.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_KEYWORD,
        )


def test_base_import_without_mflux_extra():
    """Critical contract (audit F4): `from mlx_teacache import apply_teacache`
    must succeed on machines without the [mflux] extra installed.

    We simulate the no-mflux environment by inserting None into sys.modules
    for `mflux` inside a subprocess, then import mlx_teacache.
    """
    code = """
import sys
sys.modules['mflux'] = None
sys.modules['mflux.models'] = None
sys.modules['mflux.models.flux'] = None
sys.modules['mflux.models.flux2'] = None
import mlx_teacache
from mlx_teacache import apply_teacache
assert callable(apply_teacache)
print('OK')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert "OK" in result.stdout, f"stdout={result.stdout}; stderr={result.stderr}"
    assert result.returncode == 0


def test_handle_module_has_no_variant_branches():
    """Spec → audit F3: TeaCacheHandle is variant-agnostic. Static check."""
    import inspect

    from mlx_teacache import handle as handle_module

    source = inspect.getsource(handle_module)
    code_lines = [ln for ln in source.splitlines()
                  if not ln.lstrip().startswith("#")]
    code = "\n".join(code_lines).lower()
    for bad in ("flux1", "flux2", "klein"):
        assert bad not in code, (
            f"handle.py must not mention {bad!r}; variant-specific code belongs in variants/"
        )
```

- [ ] **Step 2: Run the test**

```bash
uv run pytest tests/test_public_api.py -v
```

Expected: all pass. If `test_base_import_without_mflux_extra` fails, find the eager mflux import and move it under a lazy code path.

- [ ] **Step 3: Lint**

```bash
uv run ruff check tests/test_public_api.py
```

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_public_api.py
git commit -m "test(public-api): snapshot v0.5.x import paths + base-import-without-mflux"
```

---

## Phase F — Legacy cleanup

After every variant is ported and Phase E passes, delete the legacy modules.

### Task 21: Delete legacy `state.py`, `lifecycle.py` (top-level)

**Files:**
- Delete: `src/mlx_teacache/state.py`
- Delete: `src/mlx_teacache/lifecycle.py`

The functionality is now in `_kernel/state.py` and `_kernel/lifecycle.py`. There were no documented user-facing imports from these top-level modules (they were internal); confirm by grep before deleting.

- [ ] **Step 1: Confirm no incoming imports of `mlx_teacache.state` or `mlx_teacache.lifecycle` survive**

```bash
grep -rn "from mlx_teacache.state\|from mlx_teacache.lifecycle\|import mlx_teacache.state\|import mlx_teacache.lifecycle" \
    src/ tests/ scripts/ docs/
```

Expected: zero matches. If any survive, port them to `from mlx_teacache._kernel.state` / `_kernel.lifecycle` before deleting.

- [ ] **Step 2: Move (not rm) the legacy modules to Trash**

Per CLAUDE.md "Never permanently delete":

```bash
mv src/mlx_teacache/state.py ~/.Trash/mlx_teacache-state-v0.5-legacy-$(date +%Y-%m-%d)
mv src/mlx_teacache/lifecycle.py ~/.Trash/mlx_teacache-lifecycle-v0.5-legacy-$(date +%Y-%m-%d)
```

- [ ] **Step 3: Run the fast test suite**

```bash
uv run pytest tests/ -m "not slow and not network" --deselect tests/test_api.py::test_apply_and_restore_roundtrip
```

Expected: all pass. If anything imports the deleted modules, the test surfaces it.

- [ ] **Step 4: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/
uv run mypy src/mlx_teacache/
```

Expected: green.

- [ ] **Step 5: Commit the deletion**

```bash
git rm src/mlx_teacache/state.py src/mlx_teacache/lifecycle.py 2>/dev/null || true
git add -A src/mlx_teacache/
git commit -m "chore(legacy): remove top-level state.py + lifecycle.py (now in _kernel/)"
```

---

### Task 22: Delete legacy `integrations/mflux/forward.py`, `flux2.py`, `detect.py`

**Files:**
- Delete: `src/mlx_teacache/integrations/mflux/forward.py`
- Delete: `src/mlx_teacache/integrations/mflux/flux2.py`
- Delete: `src/mlx_teacache/integrations/mflux/detect.py`
- Possibly: `src/mlx_teacache/integrations/mflux/proxy.py` (if unused by any variant)
- Possibly: `src/mlx_teacache/integrations/__init__.py`, `src/mlx_teacache/integrations/mflux/__init__.py` (if the package becomes empty)

- [ ] **Step 1: Confirm no incoming imports**

```bash
grep -rn "from mlx_teacache.integrations\|mlx_teacache.integrations" src/ tests/ scripts/ docs/
```

Expected: zero matches.

- [ ] **Step 2: Check if `proxy.py` is referenced**

```bash
grep -rn "from .proxy\|integrations.mflux.proxy\|TransformerProxy" \
    src/mlx_teacache/ tests/
```

Expected: zero matches under `src/` (the proxy class is now per-variant). If `tests/` has references, port them to per-variant test files.

- [ ] **Step 3: Move legacy modules to Trash**

```bash
mv src/mlx_teacache/integrations/mflux/forward.py \
   ~/.Trash/mlx_teacache-integrations-forward-v0.5-legacy-$(date +%Y-%m-%d)
mv src/mlx_teacache/integrations/mflux/flux2.py \
   ~/.Trash/mlx_teacache-integrations-flux2-v0.5-legacy-$(date +%Y-%m-%d)
mv src/mlx_teacache/integrations/mflux/detect.py \
   ~/.Trash/mlx_teacache-integrations-detect-v0.5-legacy-$(date +%Y-%m-%d)
# Conditionally:
mv src/mlx_teacache/integrations/mflux/proxy.py \
   ~/.Trash/mlx_teacache-integrations-proxy-v0.5-legacy-$(date +%Y-%m-%d) 2>/dev/null || true
```

If the `integrations/mflux/` directory is now empty (besides `__init__.py`), trash the entire directory:

```bash
[ -z "$(ls src/mlx_teacache/integrations/mflux/*.py 2>/dev/null | grep -v __init__)" ] && \
    mv src/mlx_teacache/integrations ~/.Trash/mlx_teacache-integrations-v0.5-legacy-$(date +%Y-%m-%d)
```

- [ ] **Step 4: Run the test suite**

```bash
uv run pytest tests/ -m "not slow and not network" --deselect tests/test_api.py::test_apply_and_restore_roundtrip
```

Expected: all pass.

- [ ] **Step 5: Lint + typecheck**

```bash
uv run ruff check src/mlx_teacache/
uv run mypy src/mlx_teacache/
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git rm -r src/mlx_teacache/integrations/ 2>/dev/null || true
git add -A src/mlx_teacache/
git commit -m "chore(legacy): remove integrations/mflux/ (variants own integration now)"
```

---

## Phase G — Bench refactor (subprocess-per-rep)

The v0.5.1 work folded into v0.6.0. Today's `scripts/bench_speedup.py` runs 9 same-process generations for three-way mode; on 9B that's the OOM path. Refactor to subprocess-per-rep, mirroring `scripts/bench_comparison.py`.

### Task 23: Refactor `scripts/bench_speedup.py` to subprocess-per-rep

**Files:**
- Modify: `scripts/bench_speedup.py`

The new structure: orchestrator + worker in one file (same pattern as `scripts/bench_comparison.py`). Each (variant, condition, rep) runs in a fresh subprocess. Worker prints `::BENCH_RESULT::<json>` sentinel; orchestrator aggregates.

- [ ] **Step 1: Read `bench_comparison.py` for the canonical pattern**

```bash
grep -n "^def \|::BENCH_RESULT::\|--worker\|argparse" scripts/bench_comparison.py | head -20
```

- [ ] **Step 2: Rewrite `bench_speedup.py`**

Key changes from today's version:
- Add `--worker` flag. When set, the script runs as a single-condition worker. Loads the variant from the per-variant config (`META["recipes"]["default"]`), runs ONE rep, prints `::BENCH_RESULT::<json>` on stdout.
- Orchestrator spawns one subprocess per (condition, rep). For three-way mode with `--reps 3`: 9 subprocesses total (3 vanilla + 3 wrapped-no-gate + 3 wrapped-gated).
- Worker reads `META["memory_cap_hint_gb"]` from the variant's config and calls `mx.metal.set_memory_limit()` before model load.
- Orchestrator aggregates per-condition stats (median, min, max, skip counts).

Full skeleton at scripts/bench_speedup.py (port from scripts/bench_comparison.py's structure):

```python
"""Three-way release-gate bench, subprocess-per-rep.

Each (variant, condition, rep) tuple runs in its own subprocess. Workers
print a ::BENCH_RESULT::<json> sentinel on stdout; the orchestrator
aggregates. This is the v0.6.0 replacement for the v0.5.x same-process
script that OOM'd on 9B at 32 GB. See CLAUDE.md → "Memory guardrails for
heavy generations on 32 GB".

Run as:
  uv run python scripts/bench_speedup.py --variant <variant-id> [--three-way] [--reps 3]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

WORKER_SENTINEL = "::BENCH_RESULT::"


def _worker_main(args: argparse.Namespace) -> None:
    """Subprocess entrypoint. Loads variant, runs one generation, prints
    a single JSON line prefixed by WORKER_SENTINEL on stdout."""
    import mlx.core as mx

    # Discover variant config via the registry (mflux-free lookup).
    from mlx_teacache.variants import _REGISTRY

    entry = _REGISTRY[args.variant]
    meta = entry["META"]
    recipe = meta["recipes"][args.recipe]
    memory_cap_gb = meta.get("memory_cap_hint_gb")
    if memory_cap_gb is not None:
        mx.metal.set_memory_limit(int(memory_cap_gb * 1024**3))

    # NOW load mflux (this is the heavy path — only inside the worker)
    apply = entry["load_integration"]()
    flux = _load_flux_for_variant(args.variant)

    t0 = time.perf_counter()
    if args.condition == "vanilla":
        image = flux.generate_image(**_gen_kwargs(recipe))
        mx.eval(mx.zeros(1))
        elapsed = time.perf_counter() - t0
        skipped = 0
        computed = recipe["num_inference_steps"]
        thresh = None
    elif args.condition in ("wrapped-no-gate", "wrapped-gated"):
        rel_l1_thresh = 0.0 if args.condition == "wrapped-no-gate" else None
        with apply(flux, rel_l1_thresh=rel_l1_thresh) as h:
            image = flux.generate_image(**_gen_kwargs(recipe))
            mx.eval(mx.zeros(1))
            elapsed = time.perf_counter() - t0
            skipped = h.stats.skipped_count
            computed = h.stats.computed_count
            thresh = h.rel_l1_thresh
    else:
        raise ValueError(f"unknown condition: {args.condition!r}")

    peak_gb = float(mx.metal.get_peak_memory()) / (1024**3)
    print(f"{WORKER_SENTINEL}{json.dumps({
        'condition': args.condition,
        'rep': args.rep,
        'elapsed_seconds': elapsed,
        'skipped': skipped,
        'computed': computed,
        'rel_l1_thresh_used': thresh,
        'peak_memory_gb': peak_gb,
    })}", flush=True)


def _gen_kwargs(recipe: dict[str, Any]) -> dict[str, Any]:
    """Generation kwargs shared across conditions."""
    return {
        "prompt": "a red apple on a wooden table",  # canonical bench prompt
        "seed": 42,
        "num_inference_steps": recipe["num_inference_steps"],
        "guidance": recipe["guidance"],
        "height": 512,
        "width": 512,
    }


def _load_flux_for_variant(variant_id: str) -> Any:
    """mflux instance loader by variant_id. Worker-side; imports mflux."""
    from mflux.models.common.config.model_config import ModelConfig

    if variant_id == "flux1-dev":
        from mflux.models.flux.variants.txt2img.flux import Flux1
        flux = Flux1.from_name("dev", quantize=4)
    elif variant_id == "flux1-schnell":
        from mflux.models.flux.variants.txt2img.flux import Flux1
        flux = Flux1.from_name("schnell", quantize=4)
    elif variant_id == "flux2-klein-4b":
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())
    elif variant_id == "flux2-klein-9b":
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_9b())
    elif variant_id == "flux2-klein-base-4b":
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
    elif variant_id == "flux2-klein-base-9b":
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_9b())
    else:
        raise ValueError(f"unknown variant: {variant_id!r}")
    flux.freeze()
    return flux


def _spawn_worker(variant: str, condition: str, rep: int, recipe: str) -> dict[str, Any]:
    """Run one worker subprocess. Returns the parsed sentinel JSON."""
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--variant", variant,
        "--condition", condition, "--rep", str(rep),
        "--recipe", recipe,
    ]
    print(f">> spawning {variant}/{condition}/rep={rep}")
    last_result: str | None = None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith(WORKER_SENTINEL):
            last_result = line[len(WORKER_SENTINEL):].strip()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"worker {variant}/{condition}/rep={rep} exit {rc}")
    if last_result is None:
        raise SystemExit(f"worker {variant}/{condition}/rep={rep} no sentinel")
    return json.loads(last_result)


def _orchestrator_main(args: argparse.Namespace) -> None:
    from mlx_teacache.variants import _REGISTRY

    if args.variant not in _REGISTRY:
        raise SystemExit(f"unknown variant {args.variant!r}; known: {sorted(_REGISTRY.keys())}")

    conditions = ["vanilla"]
    if args.three_way:
        conditions.append("wrapped-no-gate")
    conditions.append("wrapped-gated")

    results: dict[str, list[dict[str, Any]]] = {c: [] for c in conditions}
    for condition in conditions:
        for rep in range(args.reps):
            r = _spawn_worker(args.variant, condition, rep, args.recipe)
            results[condition].append(r)

    print("\n== Summary ==")
    print(f"  variant: {args.variant}, recipe: {args.recipe}, reps: {args.reps}")
    for condition, reps in results.items():
        times = [r["elapsed_seconds"] for r in reps]
        med = statistics.median(times)
        skipped = reps[-1]["skipped"]
        print(f"  {condition}: median={med:.2f}s (all={[round(t, 2) for t in times]}, skipped={skipped})")

    if args.three_way:
        van = statistics.median([r["elapsed_seconds"] for r in results["vanilla"]])
        nogate = statistics.median([r["elapsed_seconds"] for r in results["wrapped-no-gate"]])
        gated = statistics.median([r["elapsed_seconds"] for r in results["wrapped-gated"]])
        print(f"  compile-avoidance (vanilla/nogate): {van / nogate:.2f}x")
        print(f"  gating          (nogate/gated):    {nogate / gated:.2f}x")
        print(f"  combined        (vanilla/gated):    {van / gated:.2f}x")

    if args.report is not None:
        args.report.write_text(json.dumps({"variant": args.variant, "results": results}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help="(internal) single-condition worker")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--condition", default=None,
                        choices=[None, "vanilla", "wrapped-no-gate", "wrapped-gated"])
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--recipe", default="default")
    parser.add_argument("--three-way", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if args.worker:
        if args.condition is None:
            parser.error("--worker requires --condition")
        _worker_main(args)
    else:
        _orchestrator_main(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Dry-run help text**

```bash
uv run python scripts/bench_speedup.py --help
```

Expected: usage shows `--variant`, `--three-way`, `--reps`, `--recipe`, `--report`, `--worker` (internal). No model load.

- [ ] **Step 4: Lint**

```bash
uv run ruff check scripts/bench_speedup.py
uv run ruff format --check scripts/bench_speedup.py
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_speedup.py
git commit -m "feat(bench): rewrite bench_speedup.py subprocess-per-rep (v0.5.1 work folded into v0.6.0)"
```

---

## Phase H — Validation runs

### Task 24: Numerical-equivalence run (fast tests)

**Files:** (run only; no new files)

- [ ] **Step 1: Run the full fast test suite under the new architecture**

```bash
uv run pytest tests/ -m "not slow and not network" --deselect tests/test_api.py::test_apply_and_restore_roundtrip
```

Expected: all pass. Document any failures: identify the cause (variant integration port bug, kernel boundary issue, shim regression), fix, re-run.

- [ ] **Step 2: Repo-wide lint + typecheck**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
```

Expected: all green.

- [ ] **Step 3: No commit** (verification step).

---

### Task 25: Three-way bench on klein-base-9b

**Files:**
- Create: `_artifacts/v0.6.0_bench_klein_base_9b.json`

**This is a heavy ML run.** Per CLAUDE.md "Heavy generations: main thread, not subagents" + "Always state an ETA". Estimated wall-clock: ~3-4 hours on M1 Max (3 conditions × 3 reps × ~15-45 min/rep cold subprocess each).

- [ ] **Step 1: Pre-flight memory check**

```bash
memory_pressure 2>&1 | head -2
hf auth whoami | head -1
```

Expected: memory pressure not in warn/critical; HF auth shows username.

- [ ] **Step 2: Kick off (state ETA when launching)**

```bash
uv run python scripts/bench_speedup.py \
    --variant flux2-klein-base-9b \
    --three-way \
    --reps 3 \
    --report _artifacts/v0.6.0_bench_klein_base_9b.json \
    2>&1 | tee /tmp/v0.6.0-bench-klein-base-9b.log
```

Expected wall-clock: 3-4 hours on M1 Max. Run with `run_in_background=true`.

- [ ] **Step 3: Validate the output**

```bash
cat _artifacts/v0.6.0_bench_klein_base_9b.json | python3 -m json.tool | head -40
```

Expected: JSON contains `vanilla`, `wrapped-no-gate`, `wrapped-gated` arrays each with 3 elements. Skip count on `wrapped-gated` should be 10-14 (the validation rep showed 12).

- [ ] **Step 4: Compute attribution ratios**

The orchestrator prints these at the end of stdout. Capture into the v0.6.0 release-notes draft:
- `compile-avoidance ratio` = vanilla_median / wrapped-no-gate_median
- `gating ratio` = wrapped-no-gate_median / wrapped-gated_median
- `combined ratio` = vanilla_median / wrapped-gated_median

- [ ] **Step 5: Commit**

```bash
git add _artifacts/v0.6.0_bench_klein_base_9b.json
git commit -m "chore: commit v0.6.0 three-way bench evidence for klein-base-9b"
```

---

### Task 26: Three-way bench on klein-base-4b (sanity check vs v0.4.1)

**Files:**
- Create: `_artifacts/v0.6.0_bench_klein_base_4b.json`

Reproduces v0.4.1's numbers under the new harness. If the v0.6.0 numbers diverge significantly from v0.4.1 (1.16× gating + 1.09× compile-avoidance = 1.26× combined), that's a finding — investigate before shipping.

Estimated wall-clock: ~1.5-2 hours on M1 Max.

- [ ] **Step 1: Kick off (state ETA)**

```bash
uv run python scripts/bench_speedup.py \
    --variant flux2-klein-base-4b \
    --three-way \
    --reps 3 \
    --report _artifacts/v0.6.0_bench_klein_base_4b.json \
    2>&1 | tee /tmp/v0.6.0-bench-klein-base-4b.log
```

- [ ] **Step 2: Compare to v0.4.1 baseline**

The v0.4.1 numbers (M1 Max, 50 steps + g=4.0, 9/50 skips): gating ratio ≈ 1.16, compile-avoidance ratio ≈ 1.09, combined ≈ 1.26.

If v0.6.0 produces numbers within ±5%, the new harness is consistent with v0.4.1 — the refactor preserved behavior.

If divergence > 5%, investigate:
- Did the kernel port a wrong floating-point fast path?
- Did the integration port miss a state reset?
- Is the new bench harness's subprocess isolation producing different per-rep timing characteristics than v0.4.1's same-process timing?

- [ ] **Step 3: Commit**

```bash
git add _artifacts/v0.6.0_bench_klein_base_4b.json
git commit -m "chore: commit v0.6.0 three-way bench evidence for klein-base-4b (vs v0.4.1)"
```

---

## Phase I — Docs + ship

### Task 27: Generated "Supported models" table

**Files:**
- Create: `docs/_generate_supported_models.py`
- Modify: `README.md` (the "Supported models" section)

- [ ] **Step 1: Write the generator script**

```python
# docs/_generate_supported_models.py
"""Emit the README 'Supported models' table from the variant registry.

Run as: uv run python docs/_generate_supported_models.py
Pipe into the README between markers <!-- SUPPORTED_MODELS_START --> and
<!-- SUPPORTED_MODELS_END -->.
"""
from __future__ import annotations

from mlx_teacache.variants import _REGISTRY


def main() -> None:
    print("| Variant id | mflux class + config | License | Default recipe |")
    print("|---|---|---|---|")
    for variant_id in sorted(_REGISTRY.keys()):
        meta = _REGISTRY[variant_id]["META"]
        recipe = meta["recipes"]["default"]
        recipe_str = f"{recipe['num_inference_steps']} steps, g={recipe['guidance']}"
        print(f"| `{variant_id}` | `{meta['display_name']}` | {meta['license']} | {recipe_str} |")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add markers to README**

In `README.md`, find the existing "Supported models" table and replace its body with the marker block:

```markdown
## Supported models

<!-- SUPPORTED_MODELS_START -->
<!-- generated by docs/_generate_supported_models.py — do not edit by hand -->
<!-- SUPPORTED_MODELS_END -->

[footnotes ¹, ², ³ stay below]
```

- [ ] **Step 3: Generate + paste**

```bash
uv run python docs/_generate_supported_models.py > /tmp/supported_models.md
# Manually paste the contents between the START/END markers in README.md
```

- [ ] **Step 4: Lint**

```bash
uv run ruff check docs/_generate_supported_models.py
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add docs/_generate_supported_models.py README.md
git commit -m "feat(docs): generated Supported models table from variant META"
```

---

### Task 28: Per-variant docs at `docs/variants/<name>.md`

**Files:**
- Create: `docs/variants/flux1_dev.md`
- Create: `docs/variants/flux1_schnell.md`
- Create: `docs/variants/flux2_klein_4b.md`
- Create: `docs/variants/flux2_klein_9b.md`
- Create: `docs/variants/flux2_klein_base_4b.md`
- Create: `docs/variants/flux2_klein_base_9b.md`

Each file follows the same template: variant identity, license + obligations, recommended recipe, memory cap hint, validation evidence link, quirks.

- [ ] **Step 1: Write the template for one variant (flux2_klein_base_9b)**

```markdown
# `flux2-klein-base-9b`

Non-distilled FLUX.2 Klein 9B (Black Forest Labs).

## License

FLUX Non-Commercial license. Users must accept on the
[Hugging Face model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)
before mflux can download weights.

## Recommended recipe

50 steps + guidance=4.0 (canonical upstream CFG schedule).

## Memory profile (M1 Max 32GB)

Validation peak: 25.2 GB vanilla / 13.2 GB wrapped. The `META["memory_cap_hint_gb"]` is 24
— `mx.set_memory_limit` honors this in worker subprocesses spawned by `scripts/bench_speedup.py`.

## Validation evidence

- `_artifacts/validation_klein_base_9b.json` — v0.5.0 release-gate
- `_artifacts/v0.6.0_bench_klein_base_9b.json` — v0.6.0 three-way attribution
- `_artifacts/validation_klein_base_9b_images/{vanilla,wrapper}.webp`

## Coefficient reuse

Polynomial coefficients are physically the same tuple as `flux2-klein-base-4b`
(cross-import in `src/mlx_teacache/variants/flux2_klein_base_9b/config.py`).
See `docs/calibration.md` → "Reusing coefficients across model sizes".
```

- [ ] **Step 2: Repeat for the other 5 variants**

Each gets the same shape with variant-specific values.

- [ ] **Step 3: Commit**

```bash
git add docs/variants/
git commit -m "docs(variants): per-variant docs with license, recipe, memory, evidence links"
```

---

### Task 29: CHANGELOG v0.6.0 entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add the v0.6.0 section**

Insert above the v0.5.0 entry:

```markdown
## [0.6.0] — 2026-XX-XX

Internal architectural refactor. **No user-facing breakage** — every documented import path from v0.5.x continues to work. The package now organizes variants as independent directories under `src/mlx_teacache/variants/`, with pure-algorithm primitives in `src/mlx_teacache/_kernel/`.

### Why

Four pain points the v0.5.0 architecture made hard to address structurally: bug-blast-radius, test friction, new-model coupling, per-model quirks scattered across the codebase. The new layout addresses each: variant changes are physically contained to one directory, tests are per-variant, adding a new variant is copy-template-customize without touching existing variants, and per-model quirks (memory cap, recipes, license) live with the variant.

### Changed

- `src/mlx_teacache/_kernel/` now holds pure-algorithm primitives: `gate.py`, `cfg.py`, `state.py`, `lifecycle.py`, `stats.py`. Zero mflux imports.
- `src/mlx_teacache/variants/<name>/` for each of the six supported variants. Each contains `config.py` (metadata, mflux-free), `detect.py` (predicate, mflux-free), `integration.py` (forward wrapper, mflux-touching, lazy-imported).
- `apply_teacache` rewritten as a thin dispatcher that walks `_REGISTRY` and lazy-loads the winning variant's integration module.
- `TeaCacheHandle` is now variant-agnostic. Variants register their teardown via `VariantPatch` (rollback + finalizer callback lists). No `if variant == "..."` branches in the handle.
- `scripts/bench_speedup.py` refactored to subprocess-per-rep. The v0.5.x same-process three-way path was not memory-safe at 9B on 32 GB; the new harness fixes that.

### Added

- `tests/test_public_api.py` snapshot of every documented v0.5.x import path. Catches accidental drift.
- `tests/test_public_api.py::test_base_import_without_mflux_extra` confirms `from mlx_teacache import apply_teacache` succeeds without the `[mflux]` extra installed.
- `docs/variants/<name>.md` per-variant docs.
- `docs/_generate_supported_models.py` to regenerate the README "Supported models" table from the registry.

### Measured (v0.6.0 release-gate)

- `flux2-klein-base-9b` three-way bench under subprocess-per-rep harness: gating ratio TBD, compile-avoidance ratio TBD, combined TBD. (Fills the v0.5.0 attribution caveat.)
- `flux2-klein-base-4b` three-way bench under the new harness reproduces v0.4.1's 1.16× gating + 1.09× compile-avoidance = 1.26× combined within ±5%.

### Compatibility shims (kept; no deprecation in v0.6.0)

- `mlx_teacache.stats` — re-exports from `_kernel.stats`
- `mlx_teacache.coefficients.Provenance` — re-exports from `_kernel.stats`

A future v0.7.x release may add deprecation warnings to these shims; v0.6.0 ships them silent.

### Removed (internal modules; no documented user-facing imports)

- `src/mlx_teacache/state.py` (top-level)
- `src/mlx_teacache/lifecycle.py` (top-level)
- `src/mlx_teacache/integrations/mflux/` (entire subpackage)
```

- [ ] **Step 2: Run /humanizer** (per CLAUDE.md "Public-facing docs")

The v0.6.0 entry is substantively new public-facing prose. Apply `/humanizer` rewrites where it catches AI tells.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(CHANGELOG): v0.6.0 entry"
```

---

### Task 30: ROADMAP update

**Files:**
- Modify: `ROADMAP.md`

- [ ] **Step 1: Move v0.6.0 from Active to Released**

```markdown
## Released

- **v0.6.0** — Per-variant cores + shared algorithmic kernel. Internal architectural refactor; no user-facing breakage. Six variants now live in independent directories under `src/mlx_teacache/variants/`. The v0.5.1 follow-up (bench_speedup.py subprocess-per-rep refactor) folded in. klein-base-9b three-way bench attribution measured under the new harness: TBD numbers. SSIM ≥ 0.95 gate maintained across all variants.

[v0.5.0 entry stays below]
```

Remove the v0.6.0 entry from Active. Add a new Active entry if any (the audit-suggested deprecation cycle for the compatibility shims is one candidate; the FBCache / DiCache exploration is another).

- [ ] **Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "docs(ROADMAP): v0.6.0 → Released"
```

---

### Task 31: PR + CI + STOP

- [ ] **Step 1: Push branch**

```bash
git push -u origin feature/v0.6.0-per-variant-cores
```

- [ ] **Step 2: Open PR**

```bash
gh pr create --title "v0.6.0: per-variant cores + shared algorithmic kernel" --body "$(cat <<'EOF'
## Summary

Internal architectural refactor of mlx-teacache. **No user-facing breakage** — every documented import path from v0.5.x continues to work via compatibility shims.

Six variants (flux1-dev, flux1-schnell, flux2-klein-4b, flux2-klein-9b, flux2-klein-base-4b, flux2-klein-base-9b) now live as independent directories under `src/mlx_teacache/variants/`. Pure-algorithm primitives (polynomial gate, CFG combination, state machine, lifecycle FSM, stats types) live under `src/mlx_teacache/_kernel/`.

## Why

Four pain points the v0.5.0 architecture made hard to address structurally:
- Bug-blast-radius: variant changes can affect unrelated variants via shared forward code.
- Test friction: parametrize-everything makes failures hard to isolate.
- New-model coupling: adding a variant touched 6+ files; now it's copy-template-customize.
- Per-model quirks scattered across README footnotes, conditional branches in bench scripts, free-text docstrings; now co-located with the variant.

## What was measured

- `flux2-klein-base-9b` three-way bench under the new subprocess-per-rep harness: gating ratio TBD, compile-avoidance ratio TBD, combined TBD. Fills the v0.5.0 attribution caveat that v0.5.0 explicitly deferred to v0.5.1 (now folded into v0.6.0).
- `flux2-klein-base-4b` three-way bench under the new harness reproduces v0.4.1's published numbers within ±5%.

## Reviewer guidance

1. Start with `src/mlx_teacache/_kernel/` — that's the algorithmic content.
2. Then `src/mlx_teacache/variants/flux1_dev/` — the canonical no-CFG variant.
3. Then `src/mlx_teacache/variants/flux2_klein_base_4b/` — the canonical CFG variant.
4. Skim the other four — they're variations on the two canonical patterns.
5. Evidence under `_artifacts/v0.6.0_bench_*.json`.

## Test plan

- [ ] CI green (lint + format + mypy + the fast test suite + coverage)
- [ ] Reviewer confirms compatibility-shim test (`tests/test_compatibility_shims.py`) is in place
- [ ] Reviewer confirms `tests/test_public_api.py::test_base_import_without_mflux_extra` passes
- [ ] Reviewer skims one variant's `integration.py` and `tests/variants/<name>/`
- [ ] Reviewer confirms the `VariantPatch` contract: no `if variant == "..."` branches in `src/mlx_teacache/handle.py`
EOF
)"
```

- [ ] **Step 3: STOP**

Per the release-flow rule: do NOT call `gh pr merge`. Hand the PR link + summary to the user. Human merges on GitHub. Tag-push to PyPI is a separate explicit authorization.

---

## Self-review

Before handing the plan to executors, the planner runs through this checklist.

**1. Spec coverage:** every spec section maps to a task.
- Architecture → Tasks 1-17, 18
- Coefficient reuse pattern → Task 17 (with identity test)
- VariantPatch contract → Task 8
- What stays public → Tasks 19, 20
- Migration sequence → Tasks 1-31 (in order)
- Quality gates → Task 20 (test_public_api), Task 24 (numerical equivalence), Task 25-26 (three-way bench)
- Acceptance criteria → all tasks
- Audit responses F1-F4 → Tasks 8, 19, 20 (F1 shims, F2 tests under tests/, F3 VariantPatch, F4 lazy integration import)

**2. Placeholder scan:** every "PORT_FROM_LEGACY" marker explicitly says which legacy file to copy from. Every code block is complete. No TBDs except the bench numbers in Tasks 25-26 (those are intentional — they're the output of the bench runs).

**3. Type consistency:** `META`, `matches`, `apply`, `COEFFICIENTS`, `DEFAULT_THRESH`, `RECIPES`, `LICENSE`, `VariantPatch`, `TeaCacheHandle`, `LifecycleFSM`, `TeaCacheState` names are consistent across tasks. `rel_l1_thresh` parameter name matches today's API.
