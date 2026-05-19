# Audit: per-variant cores design

Source: `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md`
Date: 2026-05-19
Scope: Material design and release risks in the v0.6.0 per-variant-core refactor spec.

## Findings

### 1. Deleting `stats.py` conflicts with the "no user-facing breakage" contract

Severity: High
Refs: `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md:12`, `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md:211`, `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md:252`, `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md:254`, `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md:302`, `src/mlx_teacache/__init__.py:14`, `src/mlx_teacache/__init__.py:40`

Evidence: The spec promises no user-facing breakage and lists `TeaCacheStats`, `GenerationStats`, and `StepDecision` as public exported types, but the acceptance criteria require deleting the top-level `stats.py`. Today those public types are imported from `mlx_teacache.stats` into the package root. The proposed public-surface gate only checks `mlx_teacache.__all__` plus exported signatures, so it would miss import-path breakage such as `from mlx_teacache.stats import TeaCacheStats`. The same concern applies to `Provenance` if `coefficients.py` is deleted without a shim.

Impact: v0.6.0 can keep root exports intact while still breaking existing users who import the public stats/provenance types from their current modules. That contradicts the spec's "no user-facing breakage" claim and would be hard to catch with the proposed API gate.

Fix: Treat `mlx_teacache.stats` as public compatibility surface for v0.6.0. Keep a thin shim that re-exports the moved stats classes, or expand the compatibility gate to include documented/current import paths and explicitly decide which ones get a deprecation cycle. Do the same for `Provenance` if `mlx_teacache.coefficients` is removed.

### 2. Putting variant tests under `src/mlx_teacache/variants/.../tests` conflicts with current typecheck and wheel layout

Severity: Medium-High
Refs: `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md:56`, `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md:241`, `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md:265`, `pyproject.toml:66`, `pyproject.toml:77`, `.github/workflows/ci.yml:33`

Evidence: The spec places `tests/` directories inside each variant package. The current wheel config packages `src/mlx_teacache`, and the current mypy target is `src/mlx_teacache`; CI installs only the `typecheck` group before running `uv run mypy src/mlx_teacache`. Moving test modules under `src/mlx_teacache` means those tests become part of the package tree and part of the typecheck target, unlike today's top-level `tests/` directory.

Impact: The refactor can accidentally ship test modules in the wheel and can make the typecheck job inspect pytest/mflux test code under a dependency set that does not install the test groups. That turns a structural refactor into CI/package churn and blurs the intended production-vs-test boundary.

Fix: Keep tests under a top-level layout such as `tests/variants/<variant>/...`, while variant runtime code stays under `src/mlx_teacache/variants/<variant>/`. If tests must live under `src/`, update the spec to require explicit wheel/mypy/coverage exclusions and CI dependency changes before implementation starts.

### 3. Variant-owned restore needs an explicit handle contract

Severity: Medium-High
Refs: `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md:40`, `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md:54`, `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md:170`, `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md:230`, `src/mlx_teacache/api.py:60`, `src/mlx_teacache/api.py:64`, `src/mlx_teacache/api.py:285`

Evidence: The spec says `TeaCacheHandle` is variant-agnostic and variant `integration.py` owns restore, but it does not define how variant-specific teardown is attached to the handle. Today `TeaCacheHandle.restore()` hard-codes FLUX.1 vs FLUX.2 mutation reversal, and `apply_teacache()` hard-codes the patch choice. Without a new finalizer/rollback interface, the refactor either leaves variant-specific branches inside the shared handle or forces every new patch shape to edit shared API/handle code.

Impact: This undermines the main architecture goal that new variants are added by filling a variant directory without touching existing shared code. It also risks restore bugs if a variant adds a different mutation shape and the central handle does not know how to reverse it.

Fix: Add an explicit contract before planning implementation: for example, variant `apply()` returns or installs a `VariantPatch` object with ordered rollback/finalize callbacks, and `TeaCacheHandle.restore()` only runs those callbacks plus shared stats/sentinel cleanup. Make this an acceptance criterion for the refactor.

### 4. Import-time registry loading can break the optional-`mflux` import guarantee

Severity: Medium
Refs: `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md:50`, `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md:128`, `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md:132`, `docs/superpowers/specs/2026-05-19-per-variant-cores-design.md:134`, `src/mlx_teacache/api.py:4`, `pyproject.toml:28`

Evidence: `mflux` is an optional extra, and the current public facade explicitly defers `mflux` imports so `from mlx_teacache import apply_teacache` works without mflux installed. The spec changes dispatch to an import-time registry that walks variant modules, and each variant package re-exports `apply` from `integration.py`. Because variant integration is the layer that touches mflux internals, this design needs a lazy-import rule, but the spec does not state one.

Impact: A straightforward implementation can make `import mlx_teacache` import all variant integration modules and fail on machines that installed only the base package. The generated README-table script would inherit the same problem if it reads `_REGISTRY` by importing runtime modules.

Fix: Split registry metadata from integration imports. `variants/__init__.py` should load only metadata and lightweight `matches` functions that preserve the no-mflux import contract, or register lazy module paths and import `integration.py` only after a variant match inside `apply_teacache()`. Add an acceptance check that base-package import succeeds without the `[mflux]` extra installed.
