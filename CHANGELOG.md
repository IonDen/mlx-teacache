# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.10.0] — 2026-09-01

Gate-correctness release. On FLUX.1, the FLUX.2 Klein family and Z-Image, generated images at the shipped defaults are unchanged, and not merely to within a tolerance: re-running the comparison showcase produced files byte-identical to the ones 0.9.x committed, with every skip count and per-step pattern re-measured identical. Qwen-Image is the exception. It skips more of its schedule than it did in 0.9.x, which is the corrected gate working as intended on the variant most sensitive to it, and its output moves accordingly — details in the first entry below. The real-weights parity lane (FLUX.1, FLUX.2 Klein, Z-Image, Qwen-Image) passed with no quality failures.

### Changed
- **The gate anchors on the immediately previous step.** The relative-L1 signal now compares each step's modulated block-0 input (or Z-Image's first-main-layer residual) against the previous gated step, computed or skipped, instead of against the last *computed* step. That is how the calibration scripts measure their training pairs and how upstream TeaCache anchors, so the polynomial is now fed the quantity it was fit on. After a skip this can let a step qualify that 0.9.x would have recomputed. FLUX.1-dev, klein-base-4b, klein-base-9b and Z-Image did not move at their shipped defaults, skipping 6/25, 9/50, 13/50 and 15/50 with the same per-step patterns as 0.9.x. Qwen-Image did move: at its shipped 0.30 it now skips 33 of 48 active steps where 0.9.x skipped 24 on the same 768×768 recipe. That is worth 3.02× against 0.9.x's 2.32× and costs some fidelity, with SSIM against vanilla going from 0.978 to 0.967 — still clear of the 0.95 floor its parity gate enforces, but a visible change from the images 0.9.x produced. Qwen's threshold was picked at the SSIM knee under the old anchoring, so the same number now sits further along the speed/quality curve; lower it if you want 0.9.x's operating point back. Re-check image quality if you run a threshold you tuned yourself.
- **A runaway guard bounds cached-residual reuse.** The wrapper forces a recompute after 8 consecutive skips (`MAX_CONSECUTIVE_SKIPS` in `mlx_teacache/_kernel/gate.py`) regardless of the accumulator. The calibrated polynomials clamp to zero for small deltas, which could otherwise stall the accumulator under the threshold and let one cached residual be reused indefinitely. At the shipped defaults the longest run of consecutive skips is 1 on FLUX.1-dev, klein-base-9b and Z-Image, 2 on klein-base-4b, and 4 on Qwen-Image, which skips much the largest share of its schedule. The cap therefore sits above every measured operating point and mainly bounds degenerate settings such as an all-zero polynomial. Upstream TeaCache has no such cap; this is a deliberate departure, like the existing clamp on the polynomial output.
- **Distilled variants warn at apply time.** `apply_teacache()` on `flux2-klein-4b` or `flux2-klein-9b` now emits `TeaCacheNoBenefitWarning` immediately, since their few-step distilled schedules never let the gate engage; previously the warning came only after a generation. Under `filterwarnings = error` this raises at apply. Every other variant is unaffected.
- The gate kernel owns the anchor update; the per-variant forwards no longer write it. Same behavior, one place to reason about.
- A non-finite gate signal (the `numerical-miss` path) now also drops the residual cached before it and zeroes the accumulator, so the next finite step re-seeds instead of being allowed to skip on a residual that is two or more steps stale. Unreachable in a healthy generation; closes a gap the review found.
- `scripts/calibrate_flux2.py` and `scripts/sweep_threshold_klein_base_4b.py` cap wired memory before loading a model and run one subprocess per prompt / per threshold, writing each result to disk as it finishes and resuming past completed chunks. `scripts/bench_speedup.py` and `scripts/bench_comparison.py` gained the same per-chunk persistence and resume (`--max-chunks` / `--max-workers` bound one invocation), and their reports now record the per-step skip pattern and the longest skip streak.

### Added
- A default-threshold SSIM gate for Z-Image in the parity lane (512×512, q8, 50 steps, guidance 4.0): the wrapper must skip 10–20 of the 48 active steps and hold SSIM ≥ 0.97 against vanilla (measured 15 skips, SSIM 0.991).
- `docs/calibration.md` documents the anchoring convention, the runaway guard, and a per-variant table of the longest observed skip streaks; `docs/m3-plus-tradeoff.md` gains a benchmark protocol for community-submitted numbers (process isolation, warm-up discard, three repetitions, mechanism attribution, one shared recipe, mains power).
- A doc-pin guard test keeps `docs/manual-verification.md`'s install pin on the latest released version.

### Performance
- All five engaged variants re-benched three-way (subprocess-per-rep, three cold reps per condition, M1 Max 32 GB, mflux 0.18.0), each run started on mains power. FLUX.1-dev at 25 steps: **1.57×** median (vanilla 113.1 s → 71.9 s), 1.50× fastest-to-fastest; the wrapper time matches 0.6.3's 71.0 s and the ratio moved because vanilla measured slower and wider this session, so read 1.5× as the conservative headline. klein-base-4b CFG: **1.22×** (233.9 s → 192.4 s; 0.6.0 measured 1.23×). klein-base-9b CFG: **1.37×** (520.6 s → 379.1 s; 0.6.0 measured 1.36×). Z-Image: **1.31×** (227.4 s → 174.2 s); 0.7.0's 1.17× at the same recipe had a host-constrained wrapper time, since the skip count and pattern are identical. Qwen-Image at its pinned 768×768 recipe: **3.02×** against vanilla (850.3 s → 282.0 s) and **2.73×** against the wrapper's own no-gate run (770.7 s → 282.0 s), from 33 of 48 active steps skipped — the largest gain of any variant either way. The 0.10× between those two baselines is measurement drift, not a wrapper effect: the no-gate output is pixel-identical to vanilla's, and the conditions were timed in separate blocks hours apart, so the true figure sits between them. This recipe also peaks at 26.2 GB, above the 25.0 GB recommended working set on a 32 GB M1 Max, and completes only under low host memory pressure. Reports under `_artifacts/v0.10.0_bench_*.json`; the README benchmark table and per-variant pages cite them.

## [0.9.3] — 2026-07-25

Maintenance release. Gate math, coefficients, and generated output are unchanged.

### Changed
- **Python 3.10 is now supported** (`requires-python >= 3.10`, previously 3.11). No library code needed changes: `mlx` itself allows 3.10, and the only 3.11-only constructs in the repo were `tomllib` imports in two test files (guarded with `tomli` on 3.10) and `datetime.UTC` in two bench scripts (replaced with `timezone.utc`). Lint and type checking now target the 3.10 floor.
- CI runs the mflux test lane on Python 3.10 through 3.13, and a configuration-guard test suite pins the declared floor, the PyPI classifiers, the CI matrix, the mypy/ruff targets, and the README badge to a single supported-versions list — a version cannot be advertised without a CI job running it.
- The README Python badge states the floor (`3.10+`) instead of enumerating individual versions.

## [0.9.2] — 2026-07-10

Maintenance release. Gate math, coefficients, and generated output are unchanged.

### Fixed
- A callback-registration failure during `apply_teacache()` now restores the original FLUX.1 or Qwen transformer instead of leaving a proxy installed without a handle. `restore()` also attempts every teardown action before reporting a failure. The double-apply sentinel remains set until teardown succeeds, so a second patch cannot be nested on a partially restored model.
- The test-suite memory guard now derives a positive wired-memory cap from each machine's recommended working-set limit. Wired and advisory memory caps are installed independently, so an unavailable wired-limit API no longer prevents the soft cap from being applied.
- The pinned README install command now uses valid PEP 508 ordering: `mlx-teacache[mflux]==0.9.2`.

### Changed
- Source distributions no longer include the test tree without its excluded reference fixtures. A package smoke job verifies that the sdist stays lean and does not cite ignored test artifacts.
- CI now runs the fast suite against both mflux 0.17.5 and the newest release allowed by `mflux>=0.17.5,<0.19`.
- Public docs no longer point at ignored benchmark and threshold-sweep outputs. The API docs include Qwen-Image's `0.30` default, the first README example uses the recommended FLUX.1-dev recipe, and stale test routing and integration commentary were removed.

## [0.9.1] — 2026-06-19

Maintenance release: supports the current mflux, with no change to any generated image.

### Changed
- **Allow mflux 0.18.x** (pin `>=0.17.5,<0.19`). The lower bound rises to 0.17.5, the version v0.9.0 shipped and was validated on (0.17.0–0.17.4 were never validated). No library code changed: the integration ports the same private mflux internals, which 0.18.0 leaves intact, so output matches v0.9.0. Compatibility was checked per variant against the installed 0.18.0 source and confirmed by generating with FLUX.1, FLUX.2 Klein, Z-Image, and Qwen-Image.
- PyPI keywords and the package description now name Qwen-Image and Z-Image, not only FLUX.

### Added
- A weight-free contract test pinning the real-mflux structures the integration relies on (the callback-registry layout, each variant's model-config aliases, and the precision default), so a future mflux change to any of them surfaces in CI instead of at generation time.
- A CI job that runs the fast lane against the lowest supported mflux (0.17.5), guarding the lower bound.

## [0.9.0] — 2026-06-18

### Added
- **Qwen-Image support** (`qwen-image`) — Alibaba's ~20B dual-stream MMDiT (Apache-2.0). At the package default it skips 25 of 48 active steps (~52%) for a 1.74× warm-median speedup, with output visually equivalent to vanilla (SSIM 0.987 at the quality-first threshold). Qwen-Image is FLUX-shaped, so the gate taps the FLUX-canonical modulated block-0 input (Signal A), calibrated in-repo at R² 0.849 — well above Z-Image's 0.40 and the FLUX.2 family's 0.11–0.47. Per-variant default `rel_l1_thresh=0.30`, set from `scripts/sweep_threshold_qwen.py` (SSIM degrades gracefully with no cliff).
- This is the first variant that proxies `flux.transformer` (the FLUX.1 pattern) **and** runs true two-pass CFG. Qwen has no `_predict` factory and no `mx.compile`, so the integration re-walks `QwenTransformer.__call__` through a proxy instead of replacing `_predict`. `generate_image` calls the transformer twice per step — positive then negative caption, combined outside it — so a branch-pairing state machine threads one shared gate decision and two cached residuals across the pair. Sharing one decision is exact, not an approximation: the gate signal depends on the latents and timestep, never the caption. The whole speedup is step-skipping; there is no compiled `_predict` to bypass the way the FLUX.2 variants do.
- `docs/variants/qwen-image.md`; a Qwen-Image row in `COMPARISON.md`; committed artifacts `scripts/calibrate_qwen.py`, `scripts/sweep_threshold_qwen.py`, and `scripts/_calibration_qwen.json`. The calibration and sweep are chunked (one worker subprocess per prompt / per threshold, resumable) so a long run survives an interruption.

### Notes
- The Qwen-Image recipe is 768×768 / 50 steps (the official Qwen recipe). On a 32 GB Mac, stock 4-bit quantization over-quantizes Qwen's sensitive layers and yields a grainy texture — a Qwen + q4 limitation, independent of TeaCache (the wrapper faithfully reproduces whatever the base model generates). The showcase portraits use a mixed-precision build (8-bit edge transformer blocks + bf16 embeddings; ~30.4 GB peak); mlx-teacache itself stays quantization-agnostic, and the shipped coefficients (calibrated on stock q4) transfer cleanly to it. The variant page has the construction snippet.

## [0.8.1] — 2026-06-13

Maintenance release — no new models and no change to generated images or benchmark numbers (gate math and coefficients untouched). Defensive cleanups and internal hygiene; a normal txt2img or img2img run behaves exactly as it did on 0.8.0.

### Fixed
- **Teardown restores a pre-existing instance `_predict` instead of deleting it.** The FLUX.2 and Z-Image variants patch `flux._predict` at the instance level, and `handle.restore()` used to delete that attribute unconditionally — which would discard a caller's own instance-level `_predict` if one had been set before `apply_teacache()`. Restore now records whether `_predict` was an instance attribute and puts the original back, the same way `generate_image` was already handled. On a stock model `_predict` is a class method, so the usual case is unchanged.
- **A malformed variant now fails with a named error.** Building the variant registry wraps each variant's import and metadata, so a broken or incomplete variant raises a `CalibrationError` naming the offending subpackage instead of an opaque `ImportError` or `KeyError` at `import mlx_teacache`.
- **FLUX.1 img2img window check uses the active step count.** A defensive fallback in the FLUX.1 forward — reached only if the per-generation step count was never set up — validated the skip window against the nominal schedule length instead of the active denoising count (`num_inference_steps - init_time_step`). It now uses the active count, matching the lifecycle. Normal generations never reach this path.

### Internal
- The package root imports its public types (`Provenance` and the stats types) from the canonical kernel modules rather than the deprecated top-level compatibility shims; the shims stay in place, so importing from them still works.
- Removed `from __future__ import annotations` across the package, with a guard test to keep it from returning. Added comments on the gate's `max(0.0, ...)` clamp (an intentional divergence from upstream) and the FLUX.2 active-step-count invariant.

## [0.8.0] — 2026-06-11

Correctness release — no new models and no change to generated images or benchmark numbers (gate math and coefficients untouched). The work is error handling, input validation, and tests that turn red when the cache goes dormant.

### Changed (behavior)
- **Invalid custom coefficients now fail at the call site.** `apply_teacache(coefficients=...)` accepts any 5-element sequence of finite floats and normalizes it to a tuple, so `handle.coefficients` is always a `tuple`. Entries that are nan, inf, or non-numeric raise `TeaCacheValueError` immediately instead of producing a gate that never skips. The rejected inputs were never valid per the documented "all finite" contract; what changes is when you find out. All seven variants now report `handle.provenance.source == "user"` after an override — five of them previously kept the builtin attribution.
- **`rel_l1_thresh=0.0` now warns.** Zero sits inside the valid range but hard-disables caching (every step computes, no speedup), which read as "safest setting" to anyone assuming lower = more conservative. A one-time `TeaCacheDisabledWarning` fires at apply time, and the `apply_teacache` docstring now states the direction: higher threshold = more skips. Zero stays accepted — it is the deliberate vanilla-equivalent reference mode the parity suite depends on. Suppress with `warnings.filterwarnings("ignore", category=TeaCacheDisabledWarning)`.

### Fixed
- **`apply()` is transactional on every variant.** Only flux1-dev guarded its install sequence; on the other six, a failure mid-apply (for example `wrap_generate_image` raising after the callback registered) left a registered callback, a swapped transformer or instance `_predict`, and no handle to restore with. All seven now reverse completed mutations in reverse install order before re-raising, and rollback-on-failure tests cover each one.
- **`except TeaCacheError` now catches everything the package raises.** `StatsFrozenError` was rooted on bare `Exception`; argument validation raised bare `ValueError` (now `TeaCacheValueError`, which is still a `ValueError`, so existing handlers keep working); `CalibrationError` was exported but had no raise site — it now backs an import-time self-check that validates every builtin coefficient tuple, so a corrupt package fails loudly at `import mlx_teacache` instead of degrading silently.

### Tests / CI
- The fast fake-based integration tests (FLUX.1 proxy, FLUX.2 predict, the lifecycle/dispatch suite) sat behind the manual parity lane and never ran on a normal PR. They now run in the per-PR lanes, and a collection guard pins the routing so they cannot drift back.
- The FLUX.2 image-quality oracle was rebuilt per recipe. It previously asserted a single unmeasured 0.85 SSIM across four variants at different step counts, and the distilled rows had dropped their skip assertion — a fully disabled cache passed green. Base txt2img rows now assert cache engagement plus floors measured at their own recipe (base-4b: 3/25 skips, SSIM 0.9927; base-9b: 7/25 skips, SSIM 0.9920 — 25 steps, seed 42, guidance 1.0, q4, M1 Max 32 GB; floors committed at 0.95 with headroom). Distilled and img2img rows keep finiteness checks only until a real measurement exists. flux1-dev's default-threshold skip count is pinned to the 5–7 band from the committed bench, so a quiet 6-to-1 decay in the advertised speedup turns the suite red.
- Suite-honesty pass: the callback-registry fakes now mirror the real mflux contract (bare-name lists, conditional registration — the path production code actually takes); skip-step residual reconstruction and CFG per-branch cache independence are pinned against independently derived references; an AST gate catches function-local `mflux` imports in `_kernel/`; tautological and permanently-skipped tests were removed. Carried from earlier hardening on main: committed coefficient/calibration artifacts are value-pinned, the `TransformerShapeError` guard is covered, and the re-export shims are identity-asserted.

### Docs
- `apply_teacache` documents the per-variant default thresholds and `handle.rel_l1_thresh`. The calibration guide routes recalibration to `variants/<id>/config.py` and `integration.py` — the dead top-level `coefficients.py` shim is no longer presented as an edit target, with a regression guard to keep it that way. The stale 1.48× FLUX.1 figure is reconciled to the benched 1.46×, the README threshold table is labeled as single-run illustration, and the hero logo renders at full width.

## [0.7.0] — 2026-06-01

### Added
- **Z-Image base support** (`z-image-base`) — the first non-FLUX variant; unlike the FLUX variants, its gate is calibrated on a latent-dependent internal signal rather than a modulation input. Z-Image (Tongyi-MAI, Apache-2.0) is a single-stream DiT whose adaLN modulation is timestep-only, so there is no cheap caption-independent signal to gate on the way the FLUX variants do. The gate taps the first-main-layer residual rel-L1, calibrated in-repo with `scripts/calibrate_z_image.py` (Signal B, origin-constrained R² = 0.400 / held-out 0.179). A caption-independent noise-refiner tap (Signal A) was tried and rejected at R² = 0.069 — its rel-L1 range is too compressed to track the body. Per-variant default `rel_l1_thresh=0.12`, set at the SSIM knee from `scripts/sweep_threshold_z_image.py`.
- Self-contained mini-kernel at `src/mlx_teacache/variants/z_image_base/{config,detect,integration}.py`. It re-walks `ZImageTransformer.__call__` with the gate, caches `main_out - unified_in` per CFG branch, and reconstructs `unified_in + cached_residual` on a skipped step. No sibling-variant imports — the variant defines its own internal handle and depends only on `_kernel/`, the public handle, and the shared mflux lifecycle. CFG combine matches mflux's `noise + guidance * (noise - negative_noise)`.
- `docs/variants/z-image-base.md`; a Z-Image row in `COMPARISON.md` (640×896 q8 portrait); committed artifacts `scripts/_calibration_z_image.json` and `scripts/_bench_z_image_v0_7_0.json`.
- Scripts and tests: `scripts/calibrate_z_image.py`, `scripts/sweep_threshold_z_image.py`, `tests/test_forward_z_image.py` (pure detect/config/registry), `tests/test_calibrate_z_image.py` (fit helper), `tests/test_parity_z_image.py` (threshold-0 cosine parity + skip-path engagement, mflux-marked).

### Changed
- `scripts/bench_speedup.py` and `scripts/bench_comparison.py` now carry a per-variant quantize (q8 for Z-Image) and resolution. `bench_comparison.py` adds a per-variant wired-memory cap and optional buffer-cache clearing between reps, needed for q8 at 640×896 where a single generation peaks ~18.7 GB but the cache otherwise accumulates across reps and OOMs the Metal command buffer.

### Performance
- `z-image-base` at the 512×512 red-apple bench recipe (subprocess-per-rep, q8, 50 steps, g=4.0): **1.17× combined**, 15 of 48 active steps skipped, SSIM 0.991, peak memory 17.2 GB → 11.9 GB. The wall-clock win is entirely gating; `mx.compile`-path avoidance is not a tailwind on Z-Image (the no-gate wrapper measured no faster than vanilla). The memory drop comes from the eager wrapper bypassing mflux's compiled `_predict`, not from gating. At the 640×896 COMPARISON portrait recipe: 1.33× warm, SSIM 0.957.

## [0.6.3] — 2026-05-31

### Added
- `_artifacts/v0.6.3_bench_flux1_dev.json` — committed three-way `bench_speedup.py` report (3 reps, subprocess-per-rep) for the FLUX.1-dev headline, so the README's headline row ships with the artifact that produced it, the way the klein rows already do.
- `tests/test_bench_artifacts.py` — pins the README FLUX.1-dev row (speedup, skip count, per-condition seconds) to that committed JSON so the headline can't drift from its artifact again.

### Changed
- FLUX.1-dev headline corrected **1.44× → 1.46×** across the README, `docs/m3-plus-tradeoff.md`, and the `mflux_teacache_flux1.py` example. The figure now comes from `_artifacts/v0.6.3_bench_flux1_dev.json` (vanilla 103.8s → wrapper 71.0s median, 6/25 skips). v0.6.1's changelog called 1.44× "the reproducible `bench_speedup.py` number" before any JSON was committed. The three-way run measures 1.46× and puts the entire win on gating (1.47× gating × 1.00× compile-avoidance), confirming step-skipping as the cause on this recipe.

### Fixed
- Removed dangling links to local-only working-docs paths from four user-facing docs the README links to (`docs/m3-plus-tradeoff.md`, `docs/calibration.md`, `docs/variants/flux2-klein-9b.md`, `docs/variants/flux2-klein-base-9b.md`), inlining the relevant reasoning where the link carried it. Corrected the `tests/conftest.py` docstring to describe the actual `_MFLUX_FILES` exact-match auto-marker logic instead of the glob it implied.

## [0.6.2] — 2026-05-27

Discoverability sweep. No runtime behavior changed; this release is a docs + metadata patch.

### Added
- `examples/` directory with three runnable scripts:
  - `mflux_teacache_flux1.py` — Flux1.from_name("dev") + apply_teacache, 25 steps. Headline 1.44× recipe.
  - `mflux_teacache_flux2_base.py` — Flux2Klein with base-4b model_config, 50 steps + g=4.0. Where step-skipping actually engages on FLUX.2.
  - `mflux_combined_with_taef.py` — symmetric counterpart to mlx-taef's combined-use example; same generation, just discoverable from this repo.
- README "Which library do I need?" section under the intro. Three-paragraph decision tree cross-linking `mlx-taef`.

### Changed
- PyPI `keywords` expanded 5 → 12 (added `apple-silicon`, `flux1`, `flux2`, `mflux`, `inference-acceleration`, `step-skipping`, `image-generation`, `cfg`) so need-based PyPI search returns this package.
- PyPI `project.urls` adds `Source`, `Changelog`, `Comparison`, and `Roadmap` entries pointing at the committed docs on `main`.
- Trove classifier `Development Status :: 3 - Alpha` → `4 - Beta`. The v0.6.x line ships with measured benchmarks and the subprocess-per-rep harness; Alpha was no longer an honest signal.

## [0.6.1] — 2026-05-26

Docs-only fast-follow. No code or behavior changes. Corrects stale and self-contradictory claims across README, COMPARISON, and the distilled-klein per-variant docs after the v0.6.0 release shipped.

### Changed
- README "Quick start" install pin: bumped the `==0.4.1` example to `==0.6.1`.
- README headline + threshold guide + chip table + Limitations + bench-table: aligned FLUX.1-dev at 25 steps to **1.44×** (the reproducible `bench_speedup.py` number) and removed the conflicting "1.48×" appearances from the earlier sweep era. Both numbers describe the same recipe; keeping one is the honest fix.
- README "Per-variant notes" + bench-table row + footnote ⁴ + Limitations: replaced `flux2-klein-base-4b` CFG headline of 1.26× combined / 1.16× gating / 1.09× compile-avoidance (v0.4.1 same-process measurement) with the v0.6.0 subprocess-per-rep numbers: **1.23× combined / 1.22× gating / 1.01× compile-avoidance**. Combined number is within day-to-day noise of v0.4.1's claim; decomposition shifted honestly under cold isolation. Same correction pattern previously applied to the 9B 2.68× → 1.36×.
- README "How the speedup happens" → mechanism (2): replaced the blanket "~1.2-1.9× faster than the compiled path even when zero steps get skipped" claim. v0.6.0's three-way bench measured compile-avoidance at 1.01-1.02× on klein-base 50-step CFG, so the wide range applies specifically to short distilled schedules (8 steps), not to long non-distilled ones. Both regimes are now called out by their measured numbers.
- README bench table: added a `flux2-klein-base-9b` row mirroring the 4B CFG row (1.36× combined, 13/50 skips, footnote ⁵) so the 9B story is in the main table rather than only in the per-variant notes.
- README bench-table footer: added v0.6.0 measurement dates (2026-05-26) and the harness distinction (subprocess-per-rep vs same-process). Distilled klein rows footnoted † as pending re-bench.
- README footnote ³ (klein-base-4b @ 25-step): flagged the 1.41× as v0.4.0 same-process measurement, not yet re-bench'd; v0.6.0's 50-step finding suggests the decomposition may shift on re-measurement.
- README "How it works" calibration sentence: now mentions all four FLUX.2 calibrated tuples (distilled klein-4b/9b, base-4b each have their own; klein-base-9b cross-imports base-4b) instead of only the distilled pair.
- COMPARISON.md header rewritten to acknowledge two harnesses (`bench_comparison.py` for flux1-dev + klein-base-4b, `bench_speedup.py --three-way` for klein-base-9b). The previous "Every number comes from bench_comparison.py" claim was false after the 9B row was added.
- COMPARISON.md "Test machine" version bumped 0.4.1 → 0.6.1.
- COMPARISON.md "Reproducing these numbers" split into two reproducer commands (bench_comparison.py for flux1-dev + 4B, bench_speedup.py for 9B) with their respective wall-times.
- docs/variants/flux2-klein-4b.md + flux2-klein-9b.md: tightened the "~1.5-2.0× wall-clock improvement" claim. v0.4-era same-process figure flagged as pending re-bench; v0.6.0's re-measurement of the same mechanism on klein-base 50-step CFG came out at 1.01-1.02× so the larger distilled figure is specific to short schedules where per-step dispatch overhead dominates wall-clock.

### Not changed
- All code paths. The 6 variant integrations, the dispatcher, the kernel, and the lifecycle wrapper are untouched.
- CHANGELOG entries for v0.6.0 and earlier. Historical entries record what was true at release time and stay as-is; v0.6.1 corrects what current docs say about those releases, not the releases themselves.

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

**Correction to v0.5.0's headline.** v0.5.0 reported a 2.68× combined speedup on klein-base-9b (vanilla 2744 s, wrapper 1025 s). That measurement was inflated by same-process MLX state leakage: the v0.5.0 bench harness ran vanilla and wrapper sequentially in one Python interpreter, so the vanilla rep paid the full-cold MLX compilation cost while the wrapper rep inherited warm allocator state from it. Wall-clock difference under that setup conflates the variant difference with the cold-vs-warm gap. v0.6.0's subprocess-per-rep harness gives every (variant, condition, rep) its own fresh interpreter, so vanilla and wrapper are both genuinely cold. The honest number is 1.36×, in line with v0.4.1's klein-base-4b result of 1.26×. v0.5.0's `README.md` and `docs/variants/flux2-klein-base-9b.md` are updated in this release.

Three-way bench on `flux2-klein-base-4b` at the same recipe (3 reps, subprocess-per-rep, M1 Max 32 GB, bf16, q4):

- **Combined: 1.23×** (vanilla 236.2 s median, wrapper 191.8 s median)
- **Gating contribution: 1.22×** (no-gate 233.4 s → gated 191.8 s)
- **`mx.compile`-path avoidance: 1.01×** (vanilla 236.2 s → no-gate 233.4 s) — effectively noise on M1 Max for this recipe
- Skip count: 9/48 active steps at `rel_l1_thresh=0.17` (stable across 3 reps, byte-identical to v0.4.1's algorithmic skip count)
- Wrapper peak memory: ~5.9 GB vs vanilla's ~10.7 GB

The 1.23× combined lands inside the day-to-day noise band on the v0.4.1 claim of 1.26×, so the verbatim ports preserved v0.4.1 behavior. What did shift is the decomposition: v0.4.1 attributed 1.16× to gating and 1.09× to compile-avoidance, but subprocess isolation reveals that gating is doing essentially all the work (1.22×) and compile-avoidance is at noise level (1.01×). The 4B decomposition matches the 9B finding (gating 1.34×, compile-avoidance 1.02×): the same mechanism dominance across both base variants.

Full evidence: `_artifacts/v0.6.0_bench_klein_base_9b.json` and `_artifacts/v0.6.0_bench_klein_base_4b.json`. Regenerate side-by-side images with `scripts/bench_comparison.py`.

### Why this refactor

In v0.5.0, adding a new FLUX variant required edits to four cross-cutting files (`detect.py`, `coefficients.py::_REGISTRY`, `forward.py`, `api.py::apply_teacache`) plus the matching test plumbing. The per-variant layout reduces that to a directory copy: a new FLUX variant lands as `variants/<new-id>/{__init__,config,detect,integration}.py` plus `tests/variants/<new-id>/`. The kernel functions (`gate_step`, `poly_eval`, `mean_abs_rel_l1`) live in `_kernel/` and are not redefined in the per-variant integrations.

The architectural pieces also bring the subprocess-per-rep bench into this release, because the memory-safety story for 9B + CFG on 32 GB depends on it.

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
  for the full mechanism breakdown.
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
