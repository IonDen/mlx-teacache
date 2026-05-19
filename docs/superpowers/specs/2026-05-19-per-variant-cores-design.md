# v0.6.0 — per-variant cores + shared algorithmic kernel (design)

**Date:** 2026-05-19
**Status:** approved by user 2026-05-19, ready for implementation plan.
**Target release:** v0.6.0
**Supersedes:** v0.5.1 (clean-attribution bench refactor). That work is folded into v0.6.0 step 5 (`bench_speedup.py` → subprocess-per-rep) because the bench naturally lives near the variant boundaries the refactor establishes.

## Goal

Restructure `src/mlx_teacache/` from "shared everything, per-variant config" to **per-variant assemblies + shared algorithmic kernel**. Six supported variants (`flux1-dev`, `flux1-schnell`, `flux2-klein-4b`, `flux2-klein-9b`, `flux2-klein-base-4b`, `flux2-klein-base-9b`) each get their own directory under `src/mlx_teacache/variants/`. Pure-algorithm primitives (polynomial gate, CFG combination math, state machine, lifecycle FSM, stats types) move to `src/mlx_teacache/_kernel/`. Variant directories own integration, config, detection, and tests.

Big-bang in v0.6.0. Public API (`apply_teacache`, `TeaCacheHandle`, `Provenance`, exception types) stays unchanged — no user-facing breakage.

## Why this refactor, and why now

Four pain points the user surfaced explicitly during brainstorming. Each is addressed structurally by per-variant cores; none was addressed structurally by per-variant config in the current architecture.

1. **Bug-blast-radius.** A change to the shared FLUX.2 forward (`integrations/mflux/flux2.py`) risks every Klein variant. Under per-variant cores, editing `variants/flux2_klein_base_9b/integration.py` physically can't affect `variants/flux2_klein_base_4b/`. Cross-variant blast is bounded to changes inside the algorithmic kernel; kernel changes are rare.
2. **Test friction.** Real-weight tests parametrize over 6 variants today, so each test runs N times. Slow, expensive (HF_TOKEN + GPU), and a failure in one variant hides the pattern in others. Per-variant cores split tests by variant; failures are focused.
3. **New-model coupling.** Adding `flux2-klein-base-9b` in v0.5.0 required edits to `detect.py`, `coefficients.py`, `api.py`, `calibrate_flux2.py`, `bench_speedup.py`, `test_parity_flux2.py`, `test_image_quality_flux2.py`, `test_detect.py`, `test_coefficients.py`. Under per-variant cores, adding a new variant is: copy a template directory, fill in config + integration, validate. No edits to existing variants.
4. **Per-model quirks belong in per-model code.** Memory profile (klein-base-9b: 24 GB cap), compile behavior (FLUX.2 eager wrapper vs compiled `_predict`), recommended recipes (base-4b: 50 steps + g=4.0), BFL safety-filter obligations, license. These are scattered across README footnotes, branch conditionals in `bench_speedup.py`, and free-text in docstrings today. Per-variant cores co-locate them with the variant's code.

## Out of scope for v0.6.0

- New variants. v0.6.0 ships the same six variants v0.5.0 supports.
- New gate mechanisms (FBCache, DiCache, TaylorSeer). ROADMAP items.
- Threshold-sweep tooling. ROADMAP item.
- Per-variant bench scripts living in the package surface. v0.6.0 keeps shared `scripts/bench_*.py` that dispatch by variant.
- Public API changes. Internal-only refactor.
- Deprecation cycle for internal modules. No user imports them; redirect shims only if a downstream report shows otherwise.

## Architecture

### Top-level layout

```
src/mlx_teacache/
├── __init__.py                     # public exports (unchanged surface)
├── api.py                          # apply_teacache — variant dispatch
├── handle.py                       # TeaCacheHandle (context manager, variant-agnostic)
├── errors.py                       # exception hierarchy (unchanged)
├── _kernel/                        # pure-algorithm primitives
│   ├── __init__.py
│   ├── gate.py
│   ├── cfg.py
│   ├── state.py
│   ├── lifecycle.py
│   └── stats.py
└── variants/
    ├── __init__.py                 # walks subdirs at import time, populates _REGISTRY
    ├── flux1_dev/
    │   ├── __init__.py             # exports META, matches, apply
    │   ├── config.py               # COEFFICIENTS, DEFAULT_THRESH, RECIPES, LICENSE
    │   ├── integration.py          # forward wrapper, mflux instance-attr swap, restore
    │   ├── detect.py               # matches(flux) -> bool
    │   └── tests/
    │       ├── test_integration.py
    │       └── test_parity.py
    ├── flux1_schnell/...
    ├── flux2_klein_4b/...          # distilled, no CFG path
    ├── flux2_klein_9b/...          # distilled, no CFG path
    ├── flux2_klein_base_4b/...     # non-distilled, CFG per-branch
    └── flux2_klein_base_9b/...     # non-distilled, CFG per-branch (reuses base-4b polynomial)
```

### Kernel surface

Pure functions and dataclasses. Zero mflux imports. Each variant's `integration.py` composes these into its forward wrapper.

```python
# src/mlx_teacache/_kernel/gate.py
def rel_l1(curr: mx.array, prev: mx.array) -> float:
    """Relative-L1 distance — the gate signal."""

def accumulate(prev_accumulated: float, delta: float) -> float:
    """Running-sum accumulator for the cumulative mod_in rel-L1."""

def polynomial_gate(
    coefficients: tuple[float, float, float, float, float],
    accumulated_rel_l1: float,
    threshold: float,
) -> bool:
    """Evaluate the calibration polynomial; return True if a skip is allowed."""
```

```python
# src/mlx_teacache/_kernel/cfg.py
def cfg_per_branch_combine(
    pos: mx.array, neg: mx.array, guidance_scale: float
) -> mx.array:
    """Standard CFG combination: neg + guidance * (pos - neg)."""
```

```python
# src/mlx_teacache/_kernel/state.py
@dataclass
class TeaCacheState:
    cached_residual: mx.array | None
    cached_residual_neg: mx.array | None  # for CFG per-branch
    accumulated_rel_l1: float
    cfg_was_active: bool

    def reset_for_new_generation(self) -> None: ...
    def record_decision(self, *, skip: bool) -> None: ...
```

```python
# src/mlx_teacache/_kernel/lifecycle.py
class LifecycleFSM:
    """Stats lifecycle: staging → finalize (natural completion) | abort (exception).
    Pure state transitions; emits TeaCacheStats when finalized."""
```

```python
# src/mlx_teacache/_kernel/stats.py
@dataclass(frozen=True)
class StepDecision: ...
@dataclass(frozen=True)
class GenerationStats: ...
@dataclass(frozen=True)
class TeaCacheStats: ...
```

The kernel is small on purpose: ~600-1000 lines total, all pure / testable without mflux. It captures only what's genuinely algorithmic. Anything that touches mflux internals (instance-attribute swaps, callback registration, model-config aliases) lives in the per-variant `integration.py`.

### Variant interface

Every `variants/<name>/__init__.py` re-exports three names. The package registry walks `variants/` at import time and registers each module by `META["variant_id"]`.

```python
# variants/flux1_dev/__init__.py
from .config import META
from .detect import matches
from .integration import apply

__all__ = ["META", "matches", "apply"]
```

```python
# variants/flux1_dev/config.py
COEFFICIENTS = (-94.9, 145.4, -77.0, 21.1, -0.0)   # upstream ali-vilab/TeaCache
DEFAULT_THRESH = 0.20
RECIPES = {
    "default": {"num_inference_steps": 25, "guidance": 3.5},
}
LICENSE = "FLUX.1-dev Non-Commercial License"
META = {
    "variant_id": "flux1-dev",
    "display_name": "FLUX.1 dev",
    "hf_model_id": "black-forest-labs/FLUX.1-dev",
    "non_distilled": True,
    "memory_cap_hint_gb": None,
    "recipes": RECIPES,
    "license": LICENSE,
}
```

```python
# variants/flux1_dev/detect.py
def matches(flux: object) -> bool:
    """Return True if this variant module owns this flux instance.
    Inspects flux's model_config.aliases. Strict — no overlap with sibling variants."""
```

```python
# variants/flux1_dev/integration.py
from mlx_teacache import _kernel
from .config import COEFFICIENTS, DEFAULT_THRESH

def apply(flux, *, rel_l1_thresh: float | None = None, ...) -> TeaCacheHandle:
    """Wire up the FLUX.1 forward wrapper. Returns a context-manager handle.
    On exit, restores flux to its pristine state."""
```

### Top-level dispatch

```python
# src/mlx_teacache/api.py
from mlx_teacache.variants import _REGISTRY
from mlx_teacache.errors import IncompatibleModelError

def apply_teacache(flux, **kwargs) -> TeaCacheHandle:
    for variant_id, module in _REGISTRY.items():
        if module.matches(flux):
            return module.apply(flux, **kwargs)
    raise IncompatibleModelError(
        actual_type=type(flux).__name__,
        actual_model_name=getattr(getattr(flux, "model_config", None), "model_name", None),
        supported=sorted(_REGISTRY.keys()),
    )
```

### Coefficient reuse (the klein-base-9b case)

The intentional reuse pattern (`flux2-klein-base-9b` shares `flux2-klein-base-4b`'s polynomial) becomes an explicit cross-import:

```python
# variants/flux2_klein_base_9b/config.py
from mlx_teacache.variants.flux2_klein_base_4b.config import COEFFICIENTS as _SHARED

COEFFICIENTS = _SHARED   # intentional reuse — see tests/test_shared_coefficients.py
```

The import statement makes the reuse visible at review time. A test in `flux2_klein_base_9b/tests/` asserts identity to catch accidental drift. This replaces today's shared-module-reference pattern in `coefficients.py`.

### What stays public, what becomes internal

**Stays public** (no user-visible change):
- `apply_teacache(flux, ...)` — same signature, same return type, same side effects.
- `TeaCacheHandle` — same context-manager protocol, same `.stats`, `.provenance`, `.restore()`.
- `Provenance` — same dataclass (moves to `_kernel/stats.py` but re-exported from package root).
- All exception types in `errors.py`.
- `__version__`.

**Becomes internal** (moves under `_kernel/` or `variants/`):
- Forward-wrapper code.
- State machine.
- Variant detection mechanism.
- Coefficient registry.

If a downstream report surfaces an internal import (e.g., `from mlx_teacache.coefficients import _REGISTRY`), we add a deprecation-shim module that re-exports under the old path with a `DeprecationWarning`. Not part of v0.6.0 unless someone needs it.

## Data flow

For inference (unchanged at the user-facing layer):

1. User: `from mlx_teacache import apply_teacache; with apply_teacache(flux) as handle: flux.generate_image(...)`.
2. `api.apply_teacache` walks `_REGISTRY` of variant modules.
3. First variant whose `matches(flux)` returns True wins.
4. That variant's `apply()` constructs the forward wrapper (importing primitives from `_kernel`), attaches it via instance-attribute swap, returns a `TeaCacheHandle`.
5. User runs the generation. The variant's wrapper handles per-step gating, residual caching, CFG combination math (if applicable). All algorithmic work goes through `_kernel` functions.
6. On context exit, the handle restores the flux instance, finalizes stats via `_kernel.lifecycle`, exposes the immutable `TeaCacheStats` on the handle.

## Migration sequence

1. **Kernel extraction.** Move `state.py`, `lifecycle.py`, `stats.py`, the gate algorithm out of `coefficients.py`, and the algorithmic core of `integrations/mflux/forward.py` + `flux2.py` into `_kernel/`. The kernel modules end up smaller than the originals because per-variant config and integration code leaves them.
2. **First two variant ports — boundary validation.** Create `variants/flux1_dev/` and `variants/flux2_klein_base_4b/`. These cover the no-CFG and CFG-per-branch paths respectively, so they validate the kernel surface. Their integration tests must pass before continuing.
3. **Remaining four variants.** Port `flux1_schnell` (~copy of flux1_dev with different coefficients), `flux2_klein_4b` and `flux2_klein_9b` (distilled, no CFG), `flux2_klein_base_9b` (CFG with coefficient cross-import from base_4b).
4. **API dispatch.** Rewrite `api.py` as the thin dispatcher described above. Delete old module-level public functions in favor of the variant-module ones.
5. **Bench refactor** (the original v0.5.1 work). Refactor `scripts/bench_speedup.py` to subprocess-per-rep. Worker prints `::BENCH_RESULT::` JSON sentinel; orchestrator aggregates. The variant's `META["memory_cap_hint_gb"]` drives the worker's MLX cap.
6. **Test migration.** Move parametrized tests into per-variant `tests/` directories. Shared test infrastructure (fixtures, helpers) lives in `tests/conftest.py` at the package root. Each variant's `tests/` is self-contained — no cross-variant imports.
7. **Numerical-equivalence validation.** For each variant, run the v0.5.0 baseline workloads through the new code. Compare:
   - Pure-core tests: bit-exact equivalence required.
   - Real-weight parity tests: SSIM ≥ 0.85 PR-gate (existing threshold).
   - Bench numbers: within 5% of the v0.5.0 measurements, accounting for run-to-run variance.
   The bench numbers for klein-base-9b also produce the clean three-way attribution that was the original v0.5.1 goal — that lands as part of v0.6.0.
8. **Docs.** Per-variant docs under `docs/variants/<name>.md` (license, recipes, memory cap, validation evidence, link to the COMPARISON.md row). README "Supported models" table built from `META` dicts (a small `docs/_generate_supported_models.py` script reads `_REGISTRY` and emits the table snippet). CHANGELOG v0.6.0 with the architectural shift framed honestly.
9. **PR + CI + ship.**

## Quality gates

- **Numerical equivalence with v0.5.0.** Per-variant baseline workloads must produce equivalent output (bit-exact for pure-core, SSIM ≥ 0.85 for real-weight, bench within 5%). Any divergence is a hard merge block.
- **Public API surface unchanged.** A diff of public symbols (`mlx_teacache.__all__` + the docstrings of exported types) shows no removed names and no signature changes.
- **No legacy code paths.** The old `integrations/mflux/forward.py`, `flux2.py`, `coefficients.py`, `state.py`, `lifecycle.py`, `stats.py` are deleted as part of the PR. v0.6.0 is the clean cut. (If a downstream import surfaces, we add a deprecation shim in v0.6.1.)
- **Three-way bench attribution lands for klein-base-9b.** The bench refactor's headline output is the clean split between gating contribution and `mx.compile`-path avoidance contribution. README footnote ³ + CHANGELOG v0.6.0 carry the measured numbers.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| 2-3 week scope creep / lost momentum | High | Per-variant numerical-equivalence gate is concrete and enforceable. If one variant won't reproduce v0.5.0 baselines after a day, pause and re-cut. |
| Kernel boundary turns out wrong (too much in kernel = no isolation; too little = duplication) | Medium | First two ported variants (flux1-dev + flux2-klein-base-4b) cover the no-CFG and CFG paths and act as kernel-boundary validation. Pause and re-cut the boundary if needed before doing the other four. |
| Test suite regressions | Medium | Existing parametrized tests run unchanged against the new code (parametrization driven by walking `_REGISTRY`). Only after every variant passes do we delete the legacy parametrize call sites. |
| OOM during validation runs (the v0.5.0 lesson) | Low after v0.5.0 fixes | Per-variant `memory_cap_hint_gb` in `META`, picked up by bench harness. Subprocess-per-rep bench refactor lands here. |
| Hidden cross-variant coupling discovered mid-refactor | Medium | First-two-variants validation catches most. Test for it explicitly: stand up flux1-dev's variant module, run its tests in isolation with `pytest variants/flux1_dev/tests/`. If kernel needs imports from other variant modules, the boundary is wrong. |
| Public API shape leaks through accidentally | Low | A test in `tests/test_public_api.py` snapshots `mlx_teacache.__all__` and the signatures of exported callables. Diff vs v0.5.0; any change is intentional and called out. |

## Effort estimate

| Phase | Calendar |
|---|---|
| Kernel extraction + boundary design | 3-5 days |
| First two variants (flux1-dev, flux2-klein-base-4b) — boundary validation | 2-3 days |
| Remaining four variants (flux1-schnell, flux2-klein-4b, flux2-klein-9b, flux2-klein-base-9b) | 2-3 days |
| Test migration + numerical-equivalence validation | 3-4 days (includes unattended bench runs) |
| Bench_speedup subprocess-per-rep refactor + clean three-way attribution run on klein-base-9b | 1-2 days |
| Docs + PR + CI + ship | 1-2 days |
| **Total** | **12-19 days focused / 2.5-4 weeks calendar** |

## Release packaging

Single feature branch (`feature/v0.6.0-per-variant-cores`). Single PR. Single tag (`v0.6.0`).

The PR is intentionally large. Reviewer guidance in the PR body:

1. Start with `src/mlx_teacache/_kernel/` — that's the algorithmic content.
2. Then `src/mlx_teacache/variants/flux1_dev/` — the canonical no-CFG variant.
3. Then `src/mlx_teacache/variants/flux2_klein_base_4b/` — the canonical CFG variant.
4. Skim the other four — they're variations on the two canonical patterns.
5. Numerical-equivalence evidence under `_artifacts/v0.6.0-baseline/`.

Per the release-flow rule: PR opens, human merges on GitHub, tag-push to PyPI is a separate explicit authorization.

## Acceptance criteria

- [ ] `src/mlx_teacache/_kernel/` exists with pure-algorithm primitives. No mflux imports.
- [ ] `src/mlx_teacache/variants/<name>/` exists for all six variants. Each exports `META`, `matches`, `apply`.
- [ ] `apply_teacache(flux)` dispatches via `_REGISTRY`. Unsupported variants raise `IncompatibleModelError` with the same message shape as v0.5.0.
- [ ] Numerical-equivalence run committed at `_artifacts/v0.6.0-baseline/` showing per-variant equivalence vs v0.5.0.
- [ ] Three-way bench on klein-base-9b under the new subprocess-per-rep harness produces clean attribution. Numbers land in README footnote ³ and CHANGELOG v0.6.0.
- [ ] `tests/test_public_api.py` confirms the public surface didn't drift.
- [ ] Legacy modules (`integrations/mflux/forward.py`, `flux2.py`, top-level `coefficients.py`, `state.py`, `lifecycle.py`, `stats.py`) are deleted.
- [ ] README "Supported models" table generated from variant `META` dicts.
- [ ] CHANGELOG v0.6.0 frames the refactor honestly (architectural shift, no user-facing changes, future-variant velocity is the payoff).
- [ ] ROADMAP: v0.6.0 moves to Released; new Active item if any.
- [ ] PR opens with three-way bench numbers and per-variant validation evidence in the body.

## Open questions

- **Bench script per-variant module or shared dispatcher?** v0.6.0 keeps the shared dispatcher (`scripts/bench_speedup.py` reads `_REGISTRY`); per-variant bench modules are deferred. The variant's `META["recipes"]["default"]` drives the bench config.
- **`docs/variants/<name>.md` content scope.** Each file has: license + obligations, recommended recipes, memory cap hint, validation evidence path, the COMPARISON.md row link, any model-specific quirks. The README footnote-³-style prose becomes the per-variant doc.
- **Generated docs.** A small `docs/_generate_supported_models.py` reads `_REGISTRY` and emits the "Supported models" table. Run pre-commit. Decided yes — table drift is a real failure mode.
