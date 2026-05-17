# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
