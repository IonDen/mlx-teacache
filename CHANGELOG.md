# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
