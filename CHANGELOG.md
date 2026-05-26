# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] — 2026-05-21

Per-variant cores + shared algorithmic kernel. The single 315-line `api.py` and the per-family integration modules (`flux1.py`, `flux2.py`, `forward.py`, `detect.py` under `integrations/mflux/`) are replaced by:

- `src/mlx_teacache/_kernel/` — pure-algorithm primitives (`gate`, `cache`, `stats`, `coefficients`). No mflux imports anywhere in this subtree; `tests/_kernel/test_kernel_no_mflux_import.py` walks the modules in a simulated no-mflux env and asserts every import succeeds.
- `src/mlx_teacache/variants/<id>/` — one subpackage per FLUX variant with `config.py` (META + COEFFICIENTS + RECIPES, mflux-free), `detect.py` (alias-based matcher, mflux-free), `integration.py` (the per-variant `apply()` plus its share of the verbatim forward port, mflux-imported lazily).
- `src/mlx_teacache/handle.py` — variant-agnostic `TeaCacheHandle` + `VariantPatch` rollback/finalizer contract. Variants build a `VariantPatch` describing their teardown; the handle runs rollbacks in reverse order on `restore()`. Stats commit/discard stays in the mflux lifecycle wrapper — `restore()` does not finalize stats (audit finding F2).

Behavior is byte-equivalent to v0.5.0 for end users. The 4-kwarg `apply_teacache(flux, *, rel_l1_thresh, coefficients, skip_first_n_steps, skip_last_n_steps)` signature is preserved (audit F3); `tests/test_api_dispatch.py` + `tests/test_public_api.py` enforce this via `inspect.signature` and a subprocess that forces `mflux` unimportable and confirms `import mlx_teacache` still works (audit F4).

### Added
- `src/mlx_teacache/_kernel/{gate,cache,stats,coefficients}.py` — verbatim extraction from the legacy module locations. Module docstring is the only diff; field sets, function bodies, and import-time behavior unchanged.
- `src/mlx_teacache/variants/__init__.py` — `_REGISTRY` walker that eagerly imports each variant's `config.py` + `detect.py` on first `mlx_teacache.variants` import; `integration.py` is loaded lazily via `entry["load_integration"]()` only after `apply_teacache` picks the winning variant.
- `src/mlx_teacache/handle.py::TeaCacheHandle` with `VariantPatch`. `tests/test_handle.py` includes a static-grep audit that `handle.py` mentions no variant names.
- Six variant subpackages under `src/mlx_teacache/variants/` covering all v0.5.x FLUX models: `flux1_dev`, `flux1_schnell`, `flux2_klein_4b`, `flux2_klein_9b`, `flux2_klein_base_4b`, `flux2_klein_base_9b`.
- `tests/conftest.py` — session-level `mx.set_wired_limit(20 GB)` + `mx.set_memory_limit(22 GB)` cap. Prevents kernel watchdog panics from misrouted parity tests that load real FLUX models (root-caused after the v0.5.x test suite triggered a kernel panic on 2026-05-20 with the conftest unguarded).
- `tests/test_api_dispatch.py` — gates the 4-kwarg `apply_teacache` signature via `inspect.signature` (audit F3 regression guard).
- `tests/test_public_api.py` — snapshots every documented v0.5.x import path + the subprocess "import without mflux" check (audit F4).
- `docs/_generate_supported_models.py` — reads `_REGISTRY` and emits the README's `## Supported models` table between `<!-- SUPPORTED_MODELS_START -->` markers. Future variants land in README automatically when registered.
- `docs/variants/<id>.md` — one page per FLUX variant covering mflux constructor, recipe + defaults, coefficient provenance, license obligations, quirks.

### Changed
- `src/mlx_teacache/api.py` is now an 86-line dispatcher that walks `_REGISTRY`. The FLUX.1 + FLUX.2 branches that used to live inline moved into each variant's `integration.py::apply()`.
- `src/mlx_teacache/integrations/mflux/lifecycle.py` is kept as the shared lifecycle module (`GenerationContextCallback`, `wrap_generate_image`, `_remove_callback_by_identity`). `_remove_callback_by_identity` moved here from `api.py`; all six variant integrations now import it from `integrations/mflux/lifecycle`.
- `src/mlx_teacache/coefficients.py` shrinks from ~235 lines to an 8-line compatibility shim that re-exports `Provenance` and `validate_custom` from `_kernel.coefficients`. `_REGISTRY`, `load_builtin`, and the per-variant coefficient tuples are gone — each variant's `config.py` owns its own COEFFICIENTS literal now (with the vendoring/calibration comment block carried over).
- `scripts/bench_speedup.py` refactored to subprocess-per-rep. Each (variant, condition, rep) runs in a fresh Python subprocess; workers print `::BENCH_RESULT::<json>` sentinels and the orchestrator aggregates median/min/max + peak memory + skip counts. Memory caps applied in each worker before model load. Folds the v0.5.1 "always intended" fix into this release.
- `scripts/calibrate_flux2.py` and several tests retargeted to import from the variant subpackages instead of the deleted `integrations/mflux/forward.py`.

### Removed
- `src/mlx_teacache/integrations/mflux/{forward,flux1,flux2,detect}.py` — 1000+ lines of legacy code. The verbatim ports live in `variants/<id>/integration.py`. Files moved to `~/.Trash` per `CLAUDE.md`'s "never `rm`" rule rather than deleted, in case any reviewer wants to inspect the verbatim-port mapping.
- `mlx_teacache.coefficients._REGISTRY` + the four per-variant coefficient tuples + `load_builtin`. The legacy registry served its purpose during the migration (transcription-error catcher); each variant owns its tuple now.

### Measured

Three-way bench on `flux2-klein-base-9b` at the canonical 50-step + g=4.0 recipe (3 reps, subprocess-per-rep, M1 Max 32 GB, bf16, q4):

- **Combined: 1.36×** (vanilla 517.6 s median, wrapper 380.6 s median)
- **Gating contribution: 1.34×** (no-gate 509.3 s → gated 380.6 s) — the v0.4.1 effect
- **`mx.compile`-path avoidance: 1.02×** (vanilla 517.6 s → no-gate 509.3 s) — the v0.4 effect, much smaller than measured on klein-base-4b
- Skip count: 13/48 active steps at `rel_l1_thresh=0.17` (stable across 3 reps)
- Wrapper peak memory: ~10 GB vs vanilla's ~22 GB

**Correction to v0.5.0's headline.** v0.5.0 reported a 2.68× combined speedup on klein-base-9b (vanilla 2744 s, wrapper 1025 s). That measurement was inflated by same-process MLX state leakage: the v0.5.0 bench harness ran vanilla and wrapper sequentially in one Python interpreter, so the vanilla rep paid full-cold MLX compilation cost while the wrapper rep inherited warm allocator state. Wall-clock difference under that setup conflates the variant difference with the cold-vs-warm gap. v0.6.0's subprocess-per-rep harness gives every (variant, condition, rep) its own fresh interpreter — both vanilla and wrapper are now genuinely cold. The honest number is 1.36×, in line with the v0.4.1 klein-base-4b result (1.26×). v0.5.0's `README.md` and `docs/variants/flux2-klein-base-9b.md` are updated in this release.

Three-way bench on `flux2-klein-base-4b` at the same recipe (3 reps, subprocess-per-rep, M1 Max 32 GB, bf16, q4):

- **Combined: 1.23×** (vanilla 236.2 s median, wrapper 191.8 s median)
- **Gating contribution: 1.22×** (no-gate 233.4 s → gated 191.8 s)
- **`mx.compile`-path avoidance: 1.01×** (vanilla 236.2 s → no-gate 233.4 s) — effectively noise on M1 Max for this recipe
- Skip count: 9/48 active steps at `rel_l1_thresh=0.17` (stable across 3 reps, byte-identical to v0.4.1's algorithmic skip count)
- Wrapper peak memory: ~5.9 GB vs vanilla's ~10.7 GB

The 1.23× combined lands inside the day-to-day noise band on the v0.4.1 claim of 1.26× — no refactor regression. The decomposition shifts honestly: v0.4.1 attributed 1.16× to gating and 1.09× to compile-avoidance, but subprocess isolation reveals that gating is doing essentially all the work (1.22×) and compile-avoidance is at noise level (1.01×). The 4B decomposition tracks the 9B finding (gating 1.34× / compile-avoidance 1.02×) — same mechanism dominance across both base variants.

Full evidence: `_artifacts/v0.6.0_bench_klein_base_9b.json`, `_artifacts/v0.6.0_bench_klein_base_4b.json`, and `tests/_artifacts/bench_images/{klein-base-9b,klein-base-4b}/`.

### Why this refactor

v0.5.0 made it clear that adding a new FLUX variant required edits to four cross-cutting files (`detect.py`, `coefficients.py::_REGISTRY`, `forward.py`, `api.py::apply_teacache`) plus matching test plumbing. The per-variant layout reduces this to a directory copy: a new FLUX variant lands as `variants/<new-id>/{__init__,config,detect,integration}.py` plus `tests/variants/<new-id>/`. The plan amendment after the kernel-boundary audit (T14) confirmed no `gate_step` / `poly_eval` / `mean_abs_rel_l1` redefinitions in variants — the kernel functions live in `_kernel/` and stay there.

The architectural pieces flush a v0.5.1 backlog item (subprocess-per-rep bench) into this release because the memory-safety story for 9B + CFG on 32 GB depends on it.

## [0.5.0] — 2026-05-19

Adds `flux2-klein-base-9b` (non-distilled FLUX.2 Klein 9B, FLUX Non-Commercial license) as a supported variant. Ships by reusing `flux2-klein-base-4b`'s polynomial coefficients verbatim and the same `rel_l1_thresh=0.17` default — justified by the shared architecture family and identical 25-step / g=1.0 calibration recipe — then validating empirically at the canonical 50-step CFG recipe before tagging.

### Added
- `flux2-klein-base-9b` in `src/mlx_teacache/integrations/mflux/detect.py` (`VariantId` Literal + `_SUPPORTED` tuple + alias branch).
- `flux2-klein-base-9b` registry entry in `src/mlx_teacache/coefficients.py` pointing at the shared `_FLUX2_KLEIN_BASE_4B_COEFFS` tuple. Provenance comment cites the reuse rationale and links the validation evidence path.
- `scripts/validate_klein_base_9b.py` — one-shot release-gate harness. Generates one fixed prompt at 50 steps + g=4.0 vanilla + wrapped, decodes through the VAE, computes SSIM, writes `_artifacts/validation_klein_base_9b.json`. Exits non-zero if SSIM < 0.95.
- `klein-base-9b` choice in `scripts/bench_speedup.py` (50 steps + g=4.0; three-way mode default-on).
- `klein-base-9b` parametrization in `tests/test_parity_flux2.py` and `tests/test_image_quality_flux2.py` (real-weight test suites, behind `HF_TOKEN`).

### Changed
- `tests/test_detect.py`: v0.4 rejection test replaced with acceptance test.
- `tests/test_coefficients.py`: new unit tests asserting `flux2-klein-base-9b` coefficients are identity-equal to `flux2-klein-base-4b` and `default_thresh == 0.17` (catches accidental drift from the intentional reuse).
- `scripts/calibrate_flux2.py`: `klein-base-9b` is no longer stubbed with `NotImplementedError`. The script is runnable for users who want to override the reused coefficients with a fresh fit; v0.5.0 itself does not run it.
- README "Supported models" gains a `flux2-klein-base-9b` row with a footnote covering the license and the validation evidence. README "When to use" updated to mention both non-distilled Klein variants.

### Measured

- **Validation SSIM**: 0.986 (50 steps, guidance=4.0, seed=42, 1024×768, M1 Max 32GB, bf16, q4, subprocess-isolated cold rep). Clears the 0.95 release gate.
- **Wall-clock**: vanilla 2744s, wrapper 1025s → **2.68×** combined speedup. 12 of 48 active steps skipped at `rel_l1_thresh=0.17`. Wrapper peak memory 13.2 GB vs vanilla 25.2 GB (the wrapper bypasses mflux's compiled `_predict` and avoids its activation overhead).
- **Caveat on the attribution.** The 2.68× combines step-skipping (the v0.4.1 gating effect) with `mx.compile`-path avoidance (the v0.4 effect that drops vanilla peak memory and runs the eager wrapper kernel-path). A clean three-way bench (vanilla / wrapped-no-gate / wrapped-gated) would attribute the two mechanisms separately. Deferred to v0.5.1 because the existing `bench_speedup.py` runs 9 same-process generations and is not memory-safe at 9B on 32 GB; refactoring it to subprocess-per-rep is the v0.5.1 follow-up.
- **Evidence**: `_artifacts/validation_klein_base_9b.json` (full JSON with hardware + memory peaks + skip counts) and `_artifacts/validation_klein_base_9b_images/{vanilla,wrapper}.webp` (side-by-side images, perceptually equivalent).

### Why the coefficient reuse is honest

`flux2-klein-base-4b` and `flux2-klein-base-9b` share the same FLUX.2 Klein transformer architecture (different depth / hidden size, same block layout) and the same non-distilled 25-step / guidance=1.0 calibration recipe. The polynomial maps cumulative input-modulation rel-L1 onto output rel-L1, a per-step property of the architecture rather than the parameter count. Validating empirically at the shipping recipe before merge converts this "should transfer" assumption into a measured fact. If the validation fails, v0.5.0 holds and a fresh calibration runs in a follow-up branch.

## [0.4.1] — 2026-05-17

CFG-engaged TeaCache for FLUX.2. The canonical upstream recipe (`guidance_scale=4.0, num_inference_steps=50`) on `flux2-klein-base-4b` now runs through a gated forward with one shared decision per step and a cached residual per branch. Measured 1.26× wall-clock vs vanilla mflux on M1 Max (1.16× from step-skipping, 1.09× from `mx.compile`-path avoidance; 9/50 skips stable across 3 reps). SSIM ≥ 0.85 PR-gate passed; cosine ≥ 0.97 parity vs real mflux at threshold=0.

### Added
- `flux2_cfg_forward_with_gate` in `src/mlx_teacache/integrations/mflux/forward.py`. One shared polynomial-gate decision per step (the `mod_in` signal is encoder-independent; see `forward.py:258-304`), two cached residuals (positive + negative branch). CFG combination math runs after the per-branch tail.
- `TeaCacheState.cached_residual_neg` for the negative branch under CFG. Cleared alongside `cached_residual` in `reset_for_new_generation`.
- `GenerationStats.cfg_was_active` now derives from a new `_Staging.cfg_was_active` flag set by the predict closure on first CFG branch entry. Replaces the obsolete `cfg_fallback > 0` derivation.
- Three-way bench protocol on `scripts/bench_speedup.py --variant klein-base-4b`: vanilla mflux / wrapped-no-gate (`rel_l1_thresh=0`, compile-avoidance only) / wrapped-gated (full v0.4.1). Separates the v0.4 compile-avoidance effect from the v0.4.1 gating effect, so future regressions land on the right mechanism.
- `scripts/calibrate_flux2.py` gains `--guidance`, `--num-inference-steps`, `--fit-branch-policy` CLI flags plus a CFG-aware capturing closure (computes both branches, returns CFG-combined noise to the scheduler, captures per-branch `body_out` plus the shared `mod_in`). Useful if a future variant's g=1.0 polynomial does not engage under CFG; v0.4.1 itself ships the existing polynomial unchanged.

### Changed
- `_vanilla_flux2_cfg_predict()` no longer runs in any production path. It stays in `src/mlx_teacache/integrations/mflux/flux2.py` with a docstring labeling it test-only and is used only by the diagnostic parity test in `tests/test_parity_flux2.py`.
- Lifecycle's distilled-step "no benefit" warning no longer suppresses on `guidance > 1.0` for FLUX.2. The regular `possible_skips == 0` check is the single source of truth.
- **Behavior change: skip-window validation under CFG.** An all-CFG generation with `skip_first_n_steps + skip_last_n_steps >= num_inference_steps` previously ran vanilla math silently. v0.4.1 raises `InvalidStepWindowError` via the lazy-validation path, which is now lifted up to fire on the first gated step regardless of CFG. Same validation as non-CFG v0.4.0.

### Deprecated
- `TeaCacheStats.cfg_fallback_steps`. Always `0` from v0.4.1+ because CFG no longer falls back. Use `GenerationStats.cfg_was_active` instead. Slated for removal in v1.0.

## [0.4.0] — 2026-05-17

This release ships `flux2-klein-base-4b` (Apache-2.0, non-distilled FLUX.2 Klein 4B). It is the first FLUX.2 variant where TeaCache step-skipping engages at the package default. The polynomial gate fires 3/25 skips and the wrapper measures 1.41× wall-clock vs vanilla on M1 Max at 25 steps — both FLUX.2 speedup mechanisms contribute (step-skipping plus `mx.compile`-path avoidance). SSIM > 0.99 vs vanilla. CFG-engaged caching for FLUX.2 is deferred to v0.4.1.

### Added
- **`flux2-klein-base-4b` support.** Apply TeaCache to mflux's `Flux2Klein` with `ModelConfig.flux2_klein_base_4b()`. Coefficients calibrated in-repo on M1 Max (10 prompts × 25 steps, origin-constrained polyfit, R² = 0.106). At the per-variant default `rel_l1_thresh=0.17` the gate skips 3/25 steps. Measured 1.41× wall-clock vs vanilla on M1 Max — ~12% from step-skipping alone (`25/(25-3) - 1`), the rest from `mx.compile`-path avoidance, same mechanism that drives the wall-clock benefit on distilled Klein 4B / 9B at 8 steps. SSIM ≥ 0.99 vs vanilla on the red-apple bench prompt (≥ 0.85 PR-gate). Cosine parity ≥ 0.97 at threshold 0.
- **Per-variant default `rel_l1_thresh` mechanism.** `Provenance` gains a `default_thresh: float | None = None` field. When a caller of `apply_teacache(flux)` does not pass `rel_l1_thresh` explicitly, the per-variant default is consulted; if `None` (the default for FLUX.1 and distilled Klein), the package-wide 0.20 fallback applies — preserving existing behavior for those variants. `flux2-klein-base-4b` ships with `default_thresh=0.17`. Resolution priority: explicit kwarg > per-variant default > 0.20 fallback.
- New `--variant klein-base-4b` argument on `scripts/bench_speedup.py` (25 steps, g=1.0).
- New `--variant klein-base-4b` argument on `scripts/calibrate_flux2.py` (replaces the v0.3 `_not_wired("v0.4.0")` placeholder).
- New `scripts/_calibration_flux2_klein_base_4b.json` calibration report.

### Changed
- `apply_teacache()`'s `rel_l1_thresh` parameter accepts a sentinel default (`_UNSET`) so the function can distinguish "user did not pass anything" from "user explicitly passed 0.20" and route through the per-variant default lookup correctly. The public-facing signature documentation is updated. `TeaCacheHandle.rel_l1_thresh` still exposes the resolved `float` value.
- Test parametrization for FLUX.2 image-quality + parity tests extended with `klein-base-4b`; `_gen_kwargs_klein()` is now variant-aware (distilled Klein at 8 steps, base-4b at 25 steps). Resolves the audit finding that base-4b would otherwise have been validated at the distilled 8-step schedule, evaluating the calibrated polynomial outside its fit range.
- `detect.identify_variant()` returns `"flux2-klein-base-4b"` for the new variant id; `VariantId` Literal + `_SUPPORTED` tuple extended accordingly.

### Scope notes
- TeaCache on base-4b is engaged at `guidance=1.0` only. At `guidance > 1.0`, the wrapper records a `cfg-fallback` decision and runs vanilla mflux per the v0.1 design. The upstream BFL model card recommends `guidance_scale=4.0`; that recipe runs at vanilla mflux speed in v0.4.0 and gets caching in v0.4.1 (per-branch caching for FLUX.2).
- Distilled FLUX.2 Klein 4B + 9B remain out of scope for algorithmic step-skipping by design (already documented in v0.3.0). Their `Provenance.default_thresh` stays `None`, so callers see no behavior change.
- The polynomial fit on base-4b has R² = 0.106 — much lower than FLUX.1-family (~0.8+) or Klein 9B (0.47). At the FLUX.1-tuned package default 0.20 the gate over-fires (19/25 skips, SSIM 0.76). The 0.17 per-variant default was tuned empirically against a threshold sweep — reproducer at `scripts/sweep_threshold_klein_base_4b.py`; the cliff above 0.17 is sharp (at 0.175, 14 skips fire and SSIM collapses to 0.78). Better fit methods are open research for a future release.

## [0.3.0] — 2026-05-16

This release ships `flux2-klein-9b` support and corrects misleading performance framing from v0.2.0. The polynomial gate works as advertised on FLUX.1-dev at long schedules; on FLUX.2 Klein at distilled 4-8 step defaults it does not engage, and the README + benchmarks now say so explicitly.

### Added
- **`flux2-klein-9b` support.** Apply TeaCache to mflux's `Flux2Klein` with
  `ModelConfig.flux2_klein_9b()`. Coefficients calibrated in-repo on M1 Max
  (10 prompts × 8 steps, origin-constrained polyfit). Output quality
  preserved (SSIM ≥ 0.85 PR-gate). The polynomial gate produces 0 skips at
  the package default threshold on Klein 9B's 8-step schedule; wall-clock
  improvement (~1.5-2.0× measured) comes from `mx.compile`-path avoidance
  rather than caching. See README "Benchmarks → How the speedup happens"
  and the postmortem at
  `docs/superpowers/notes/2026-05-16-flux2-teacache-non-engagement-postmortem.md`.
- **`scripts/bench_speedup.py`** — committed reproducer for all benchmark
  numbers in the README. Pins seed, prompt, image dimensions, step count;
  warmup + 3 timed reps; reports median wall-clock and per-rep skip
  telemetry. README's Benchmarks rows now all carry a one-line `uv run`
  command users can rerun on their hardware.
- **`--fit-mode {free, origin}`** flag on `scripts/calibrate_flux2.py`.
  `origin` constrains the polynomial through (0, 0) so the predicted
  output rel-L1 is 0 when the input rel-L1 is 0. Added during the v0.3.0
  diagnosis when the unconstrained Klein 9B fit produced `poly(0) ≈ 5.36`
  (physically nonsensical). The Klein 9B registry entry uses `origin`.
  Calibration JSON also now carries the raw `(x, y)` arrays so future
  refits can run offline.
- README `## License obligations` section flagging the FLUX.2 Klein
  non-commercial terms and BFL safety-filter requirements for 9B.

### Changed
- **README benchmarks rewritten** with measured numbers from
  `scripts/bench_speedup.py` on M1 Max 32GB. The table now reports skip
  counts alongside wall-clock so users can see which rows are TeaCache
  step-skipping (FLUX.1-dev: 6/25 skipped, 1.44×) and which are
  `mx.compile`-path avoidance only (Klein 4B + 9B at 8 steps: 0/8 skipped,
  1.26-1.93×).
- **README "How the speedup happens" subsection added** explaining the two
  mechanisms (step-skipping vs compile-path avoidance) and which fires on
  which variant + schedule.
- `scripts/calibrate_flux2_klein.py` renamed to `scripts/calibrate_flux2.py`
  with a `--variant` flag (klein-4b, klein-9b wired; klein-base-4b and
  klein-base-9b declared but raise `NotImplementedError`, wired in v0.4.0
  and v0.5.0).
- `scripts/_calibration_flux2_klein.json` renamed to
  `_calibration_flux2_klein_4b.json` (4B coefficients themselves unchanged).
- `TeaCacheHandle.variant_id` now reuses `detect.VariantId`; the
  `IncompatibleModelError` `supported` list in `api.py` has a single source
  of truth (`detect._SUPPORTED`).
- FLUX.2 `_predict` defensive guard broadened from
  `variant_id == "flux2-klein-4b"` to `variant_id.startswith("flux2-")`.

### Removed
- **`Img2ImgNotSupportedError`** (deprecated in v0.2.0 with explicit
  removal-in-v0.3.0 intent). Migration: catch the underlying
  `IncompatibleModelError` or `InvalidStepWindowError` instead.

### Correction of v0.2.0 framing
- The v0.2.0 README presented the "~1.2× Klein 4B speedup" as a TeaCache
  step-skipping outcome. Today's bench (`bench_speedup.py`) shows that
  Klein 4B at the 8-step default produces 0 cache-skipped steps across 3
  reps; the 1.26× wall-clock improvement comes entirely from
  `mx.compile`-path avoidance. v0.2.0 was correct on quality (output is
  preserved) and on the wall-clock measurement, but wrong on the
  mechanism. v0.3.0 corrects the README + CHANGELOG framing. Distilled
  schedules are now declared out of scope for algorithmic step-skipping
  (see `Limitations`). v0.4.0 targets `flux2-klein-base-4b` (non-distilled,
  Apache-2.0) — the first FLUX.2 variant where the polynomial gate is
  expected to engage on its own.

## [0.2.0] — 2026-05-16

### Added
- **img2img support** for FLUX.1 dev/schnell and FLUX.2 Klein 4B. Pass
  `image_path` + `image_strength > 0` to `flux.generate_image()` with TeaCache
  active. Caching engages on the effective denoising window
  (`num_inference_steps - init_time_step` predict calls per mflux's `Config`
  semantics). Same polynomial coefficients as txt2img — verified by SSIM gates
  on a fixed init-image suite.
- **`TeaCacheNoBenefitWarning`** category emitted once per handle when the
  current configuration cannot produce any cache-skipped steps — i.e., when
  `active_num_steps - skip_first_n_steps - skip_last_n_steps <= 1`. Examples:
  very short schedules (`num_inference_steps=3` with default skip windows),
  or aggressive `skip_first` / `skip_last` on any schedule. Suppress via the
  standard `warnings` module.
- `docs/calibration.md` documents the coefficient calibration procedure
  (cross-reference for `scripts/calibrate_flux2_klein.py`).

### Changed
- `InvalidStepWindowError` message now references "active denoising steps"
  and reports both nominal and active step counts under img2img.
- FLUX.1 forward path: cache reset is now lifecycle-owned (was: triggered by
  `t == 0`); step indexing inside the gate uses a 0-based counter (was:
  absolute scheduler timestep). No behavior change for txt2img; fixes
  silent state leakage under img2img.
- FLUX.2 predict closure: removed the redundant cache reset (lifecycle owns
  it now); the context-consumption guard and `MissingGenerationContextError`
  paths are unchanged.
- README per-chip Performance section corrected: M1 Pro / M2 Pro are eager
  in mflux 0.17.5 (the `is_m1_or_m2()` predicate only excludes Max + Ultra
  variants). M5 TensorOps wording softened to "may lose some or all" pending
  hardware-side benchmark confirmation.
- `docs/manual-verification.md` rewritten to use a working recipe
  (`after_loop` callback for latent capture; FLUX.2 cosine oracle).

### Deprecated
- `Img2ImgNotSupportedError` is deprecated; constructing it issues
  `DeprecationWarning`. It is no longer raised internally. Removal planned
  for v0.3.0.

### Fixed
- Stats finalization under img2img now passes the active denoising step count
  (not nominal) to `finalize_last_generation`, preserving the
  `len(decisions) == num_inference_steps` invariant.

## [0.1.1] — 2026-05-15

### Changed
- Repo-wide `ruff format` pass (no behavior change).
- Replaced a `# type: ignore[arg-type]` in `integrations/mflux/flux2.py` with explicit `assert ... is not None` for the CFG-fallback `negative_prompt_embeds` / `negative_text_ids` narrowing. Same runtime behavior; cleaner type story.

### Fixed
- CI: `test-pure-core` no longer runs on Linux (MLX has no Linux wheel — `libmlx.so` is macOS-only). Moved to `macos-14`.
- CI: `test-parity` is now gated behind `workflow_dispatch` because it needs gated HuggingFace weights (FLUX.1-dev terms-acceptance) that a fresh GitHub-hosted runner can't provide.
- CI: image-quality test modules use `pytest.importorskip("skimage.metrics")` so the pure-core job can collect them cleanly when scikit-image isn't in the active dependency group.
- CI: coverage floor adjusted from an aspirational 100% to a structural 70%. The mflux integration paths are only reachable with real model weights and live behind `pytest.mark.parity`. Bumping later is fine once we wire HF auth.

## [0.1.0] — 2026-05-15

### Added
- Initial public release.
- `apply_teacache(flux, *, rel_l1_thresh=0.20, ...)` for FLUX.1 dev/schnell and FLUX.2 Klein 4b.
- Context-manager-compatible `TeaCacheHandle` with live `.stats`, `.provenance`, `.restore()`.
- Built-in polynomial coefficients vendored from ali-vilab/TeaCache (FLUX.1) and derived in-repo (FLUX.2 Klein 4b).
- Custom-coefficient override path.
- Auto-disable on FLUX.2 CFG (`guidance > 1.0`); bit-exact vanilla fallback.
- img2img rejection with `Img2ImgNotSupportedError`.
- Threshold-zero fast path: at `rel_l1_thresh <= 0` the wrapper skips building cache tensors entirely (cheap no-op).
- Five-tier test pyramid: shape/dtype unit tests, paired same-process latent parity (bit-exact for FLUX.1, cosine ≥ 0.97 for FLUX.2), image-level SSIM gates on VAE-decoded outputs.
- Trusted-Publishing release pipeline.

### Calibration notes (2026-05-15)
- Default `rel_l1_thresh` chosen by visual comparison + SSIM measurement: at 0.25 some text/synthetic prompts changed rendering style; at 0.20 outputs are indistinguishable from vanilla while still skipping ~25% of steps.
- FLUX.1-dev / 25 steps / M1 Max: measured 1.48× speedup at default threshold, SSIM ≥ 0.80 on a 5-prompt suite.
- FLUX.1 polynomial coefficients corrected from a transcription error in an earlier revision (`c0..c3` were ~10× too large); now match ali-vilab upstream exactly.

### Known limitations
- v0.1 supports txt2img only.
- FLUX.2 Klein variants other than `flux2-klein-4b` are not in v0.1.
- Distilled-step schedules (FLUX.1 schnell 4-step, Klein 4-step) see no measurable speedup.
- M3+ users lose mflux's `mx.compile` of `_predict`; net behavior unmeasured in v0.1 (M1 Max benchmarks only).
- FLUX.2 parity is numerical, not bit-exact: vanilla-compiled vs wrapper-eager Metal kernel dispatch differs by ~1 ULP per element. Cosine similarity ≥ 0.99 measured; image-level SSIM is the user-facing guarantee. CFG-fallback path remains bit-exact.
