# FLUX.2 Klein TeaCache non-engagement — postmortem

**Date:** 2026-05-16
**Surface:** v0.3.0 PR #2 (Klein 9B support)
**Severity:** Misleading user-facing performance claims in shipped v0.2.0 + about-to-ship v0.3.0; no correctness regression.
**Status:** Findings final; v0.3.0 honesty rewrite shipped.

> **Coda — 2026-05-17:** The "v0.4 research task to fix it properly" track described below (FirstBlockCache port, per-step-index lookup, TaylorSeer / DiCache adaptations) is **no longer planned**. After consideration, distilled-schedule algorithmic step-skipping has been declared an explicit non-goal — the ROADMAP "Out of scope" section documents the decision. v0.4 instead targets `flux2-klein-base-4b` (non-distilled, Apache-2.0, designed for 20-50 step generation), which is the first FLUX.2 variant where the existing polynomial gate is expected to engage on its own. The research references in this postmortem are retained for the historical record and for anyone who wants to pick up that thread; the library itself will not. The wall-clock benefit on distilled Klein from `mx.compile`-path avoidance remains documented and supported.

## What happened

While preparing v0.3.0 release of mlx-teacache (FLUX.2 Klein 9B support), the implementation passed every quality gate (SSIM ≥ 0.85, cosine ≥ 0.97, parity oracle green) and was tagged. Before merging, we wrote a committed `scripts/bench_speedup.py` and ran it against the three "supported with measured speedup" variants. The bench surfaced:

| Variant | num_steps | Vanilla median | Wrapper median | Speedup | Skipped | Computed |
|---|---|---|---|---|---|---|
| FLUX.1-dev | 25 | 103.67 s | 71.79 s | **1.44×** | **6 / 25** | 17 / 25 |
| FLUX.2 Klein 4B | 8 | 28.1 s | 22.3 s | 1.26× | **0 / 8** | 6 / 8 |
| FLUX.2 Klein 9B | 8 | 119.0 s | 61.8 s | 1.93×† | **0 / 8** | 8 / 8 |

(All measured on M1 Max 32GB, mflux 0.17.5, quantize=4, 512×512, seed=42, red-apple prompt; 3 timed reps + warmup; median reported. Numbers match the v0.3.0 README "Benchmarks" table — single source of truth, produced by `scripts/bench_speedup.py` with the v2 origin-constrained 9B coefficients.)

† Klein 9B wall-clock has high variance from thermal throttling on M1 Max at quantize=4 — the bench runs all vanilla reps before all wrapper reps (`scripts/bench_speedup.py:152-188`), so a thermally-throttled vanilla rep paired with a recovered wrapper rep inflates the median. The steady-state range across reps is roughly 1.5-2.0× depending on system load. The 0/8 skip count is stable across all reps and is the load-bearing finding of this postmortem; the exact speedup ratio is not. An earlier free-fit calibration run on the same hardware recorded 98.52 s → 58.36 s (1.69×); the conclusion (0 skips, compile-avoidance only) was identical.

**FLUX.1-dev is genuinely doing what TeaCache is designed to do.** 6 of 25 steps reused the cached residual; the 1.44× speedup matches the v0.2.0 README's 1.48× claim within Metal kernel-dispatch noise.

**FLUX.2 Klein 4B and 9B at their distilled 8-step default produce ZERO skips across all reps.** Every eligible step computed. The wrapper still ran 1.2× to 1.7× faster than vanilla mflux, but the win comes from a *different* mechanism: mlx-teacache's eager wrapper replaces mflux's `mx.compile`-wrapped `_predict` with an eager Python closure. On chips where mflux compiles `_predict` (M1 Max in our case), this trades the compiled-graph optimization for live per-step gating. With 0 skips the live-gating "benefit" is illusory — the wrapper is faster because it sidesteps `mx.compile`'s recompile/dispatch overhead at quantize=4, not because TeaCache decided to skip anything.

This contradicted the v0.2.0 README's framing of FLUX.2 Klein 4B's "~1.2× speedup" as a TeaCache caching outcome.

## Root cause

The polynomial gate decides per step:

- input `x = rel_l1(mod_in_t, mod_in_{t-1})` — how much the modulated block-0 input changed since the previous step
- predicted `y = poly_eval(coefficients, x)` — how much the transformer body output will change
- gate sums predicted `y` since the last *actual* compute step. If the running sum ≥ `rel_l1_thresh`, the gate says **compute** and resets the sum. Otherwise it says **skip** and reuses the cached residual.

The threshold default is `0.20`.

For the Klein 9B calibration data (10 prompts × 8 steps × seed=42), the empirical `y` values fall in the range `[0.25, 0.88]`. That is: between any two adjacent 8-step Klein trajectory points the body output ALREADY changes by ≥ 25% rel-L1. Even before the polynomial fit gets involved, the data says "no two adjacent steps are similar enough to skip at threshold 0.20."

We refit the polynomial twice during diagnosis:

1. **`fit-mode=free`** (the original v0.3.0 calibration): unconstrained `numpy.polyfit`. R² = 0.5421, `poly(0) = 5.36`. The polynomial passed through (0, 5.36), which is physically nonsensical (no input change should never predict a 5.36 rel-L1 output change). Its output was always above the threshold over the fit range.

2. **`fit-mode=origin`** (recalibration during this incident, shipped in v0.3.0): constrained least-squares with `poly(0) = 0`. R² = 0.4710, coefficients `(-523.84, 530.25, -177.64, 20.89, 0.0)`. Slightly worse R² as expected when adding a constraint, but physically sensible. Evaluated over the fit-range `x ∈ [0.1316, 0.4498]` the polynomial output spans `[0.269, 0.724]` — still always above the 0.20 default threshold. (Empirical `y` over the same data: `[0.250, 0.880]`.)

Either polynomial faithfully reflects the data: **the data itself says no skips are possible at the threshold.** The Klein 4B polynomial behaves similarly — evaluated over its fit-range `x ∈ [0.2001, 0.4253]` (empirical `y ∈ [0.261, 1.008]`), the polynomial output spans `[0.424, 0.882]`, all above the default threshold.

This is consistent with Klein being distilled to do significant work per step. The whole denoising trajectory collapses into 4 or 8 steps; each step is consequential. TeaCache's premise — adjacent steps are similar enough that we can skip some — does not hold for heavily-distilled short schedules.

## Why this wasn't caught earlier

Three contributing factors:

1. **No committed benchmark script before v0.3.0 PR #2.** v0.2.0 shipped the "1.2× Klein 4B" claim from a single ad-hoc timing run (no `scripts/bench_*.py`, no captured skip telemetry). Without a way to re-validate, the framing fossilized.

2. **The quality gates pass even when the cache is inactive.** SSIM ≥ 0.85 is satisfied because with 0 skips, the wrapper's output differs from vanilla only by Metal kernel-dispatch noise (cosine ≥ 0.99). The gates don't include a skip-count assertion — there's no `assert handle.stats.skipped_count > 0` anywhere.

3. **The `TeaCacheNoBenefitWarning` (v0.2.0) gates on schedule shape, not polynomial behavior.** It fires when `eligible - 1 <= 0` (active steps minus skip windows). For Klein 9B at 8 steps with default skip windows, `eligible = 6`, `possible_skips = 5` — so the warning correctly doesn't fire. But the polynomial gate never actually triggers any skips. The warning would have caught a schedule-shape problem but couldn't catch a polynomial-vs-threshold mismatch.

## What we're shipping (v0.3.0)

The v0.3.0 release goes ahead as an **honesty release**:

- All four variants (`flux1-dev`, `flux1-schnell`, `flux2-klein-4b`, `flux2-klein-9b`) stay in the supported list. Structurally they all work: `apply_teacache(flux)` returns a handle, output quality is preserved, restore() works, all the lifecycle/gate/forward/cache code paths are exercised.
- The README's "Benchmarks" table is rewritten with measured numbers from `scripts/bench_speedup.py`. Skip counts are reported alongside wall-clock so users can see what's caching vs. what's `mx.compile` avoidance.
- The README adds an explicit "Mechanism" column / paragraph distinguishing **step-skipping (FLUX.1)** from **compile-path avoidance (FLUX.2 Klein at default threshold + schedule)**.
- The CHANGELOG calls out the v0.2.0 Klein 4B framing as inaccurate and corrects it.
- `scripts/bench_speedup.py` is committed as the source-of-truth reproducer.
- `--fit-mode {free, origin}` is added to the calibration script; the 9B entry in `_REGISTRY` uses the origin-constrained fit.

The library is still genuinely useful on FLUX.2 Klein — users get the wall-clock improvement whether or not the cache engages — but the README will not let the user think it's coming from caching when it isn't.

## What v0.4 should investigate (research task)

This postmortem is the brief for a future v0.4 research effort. The questions:

- **Is there a cache-style technique that actually works on distilled short schedules?** Block-level / first-block caching (FBCache, FirstBlockCache) skips part of each step instead of skipping whole steps — different premise.
- **What thresholds and calibration approaches do other implementations use for FLUX.2 schnell / Klein?** Upstream `ali-vilab/TeaCache` and ComfyUI-TeaCache may have already hit this wall and have remediation. (See the references section below — populated by the 2026-05-16 research subagent.)
- **Is the `mx.compile`-avoidance speedup on FLUX.2 worth exposing as a separate library?** A 1.2-1.7× wall-clock improvement that doesn't depend on TeaCache's polynomial gate at all could be a much smaller, more reliable library.
- **For Klein 9B specifically, would a higher default threshold (e.g. 0.5) trigger skips without unacceptable quality loss?** Quick to test once the research suggests acceptable bounds.

## What other implementations do (May 2026 research)

A research subagent surveyed `ali-vilab/TeaCache`, `diffusers`, NVIDIA's FLUX.2 work, the academic literature (TeaCache CVPR 2025, SeaCache, DiCache), and community ComfyUI nodes. Headline findings:

- **No FLUX.2 path exists in `ali-vilab/TeaCache`.** The repo has only `TeaCache4FLUX/teacache_flux.py` (FLUX.1-dev, polynomial fit at `num_inference_steps=28`). Issue #83 shows a user tried TeaCache on FLUX.1-schnell at 4 steps and got a floating-point exception with no maintainer fix. Closest upstream acknowledgment that short/distilled schedules break the stock implementation.
- **The TeaCache paper (arXiv:2411.19108) calibrates only on T ≥ 30 schedules.** It does not discuss distilled models. The implicit assumption is trajectory smoothness across many steps — accumulated rel-L1 needs multiple steps to cross a threshold. That mechanism is exactly what breaks at 4-8 steps.
- **NVIDIA's FLUX.2-dev blog uses `teacache_thresh=0.05`** — four times lower than our 0.20 default — for ~32% skips at 50 steps. But this is FLUX.2-**dev** (non-distilled), not Klein. They publish no Klein numbers.
- **No source publishes a Klein skip-rate / quality calibration.** Runware's FLUX.2 Klein 4B and 9B docs expose `teaCache`, `fbCache`, and `dbCache` accelerator knobs with default values (`teaCacheDistance=0.5`, `dbCacheThreshold=0.25`) and document 4-step default inference for Klein 9B. They do not publish skip-rate, image-quality deltas, or the underlying calibration data, so the mechanism behind those defaults cannot be inferred from the docs — only that the knobs are exposed at values four times higher than our 0.20 default.
- **Community consensus:** "Distilled models are faster because they use fewer steps. TeaCache is for quality-preserving speedup on the same number of steps." (paraphrased from the TeaCache paper and multiple ComfyUI guides). Klein's 4-8 steps IS the speedup; layering TeaCache on top is uncharted territory.

### Alternatives worth chasing in v0.4

- **FirstBlockCache (FBCache)** — now in diffusers mainline as `apply_first_block_cache`. Architecture-agnostic, no polynomial calibration: runs only the first transformer block, gates on `absmean(block_1_residual - cached_residual)`, skips the rest if below threshold. Reported 1.3-3.0× on FLUX.1-dev at 28 steps. No published Klein measurement — but the lack of a polynomial step is what makes it a candidate for short schedules.
- **Per-step-index lookup table.** Offline profile of Klein at 8 steps: which step indices have small adjacent body-output change? If e.g. steps 4-5 are reliably more similar than steps 1-2, hardcode that into a per-step decision instead of running a continuous gate. Sidesteps the polynomial-vs-threshold mismatch entirely.
- **TaylorSeer (`TaylorSeerCacheConfig` in diffusers, arXiv:2503.06923)** — fixed-interval cache (`cache_interval=N`) instead of adaptive threshold. Conceivably tune to `cache_interval=2` for Klein 8-step and accept some quality loss. Untested in open sources at short schedules.
- **DiCache (arXiv:2508.17356)** — online probe profiling: a shallow-layer L1 acts as proxy for full output change. Spearman 0.8 correlation with TeaCache's signal but eliminates the offline-calibration step. Relevant for Klein where polynomial calibration is hard.

### Threshold-tuning is NOT the v0.4 answer for Klein

At Klein 9B's measured `y_min = 0.25`, every adjacent-step output change already exceeds 0.20. Pushing `rel_l1_thresh` above 0.25 would technically trigger some skips, but the quality degradation on a 4-8 step distilled trajectory is likely severe — each step is doing too much work for any one of them to safely skip. v0.4 should investigate block-level (FBCache) or step-index-table approaches, not just chase a different gate threshold.

### References

1. `ali-vilab/TeaCache` — FLUX.1-dev polynomial source, the upstream reference. No FLUX.2 path exists. `https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py`
2. NVIDIA Blackwell FLUX.2-dev blog — only published `teacache_thresh=0.05` on FLUX.2; 50-step non-distilled. `https://developer.nvidia.com/blog/scaling-nvfp4-inference-for-flux-2-on-nvidia-blackwell-data-center-gpus/`
3. diffusers CacheMixin / FirstBlockCacheConfig docs — FBCache, FasterCache (`is_guidance_distilled`), TaylorSeer, PAB all in one API. `https://huggingface.co/docs/diffusers/api/cache`
4. TeaCache paper, arXiv:2411.19108 (CVPR 2025) — establishes polynomial rescaling; calibration on T ≥ 30 only. `https://arxiv.org/abs/2411.19108`
5. SeaCache, arXiv:2602.18993 — spectral filter on distance metric; FLUX.1-dev 50-step, PSNR 26.3 vs TeaCache's 20.8. `https://arxiv.org/html/2602.18993v2`
6. DiCache, arXiv:2508.17356 — online probe profiling, no offline calibration; 3.22× on FLUX.1-dev. `https://arxiv.org/html/2508.17356v1`
7. Runware FLUX.2 Klein docs — expose `teaCache` / `fbCache` / `dbCache` knobs with default values (`teaCacheDistance=0.5`, `dbCacheThreshold=0.25`), no published skip-rate / quality calibration. `https://runware.ai/docs/models/bfl-flux-2-klein-4b` and `https://runware.ai/docs/models/bfl-flux-2-klein-9b`
8. HuggingFace FLUX.2-klein-9b-fp8 community discussion — reports anatomy artifacts at 4 steps, recommends 8 steps. `https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/discussions/2`
9. `capitan01R/ComfyUI-Flux2Klein-Enhancer` — community node for Klein post-processing; possible signal for community caching experiments. `https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer`
10. `ali-vilab/TeaCache` issue #83 — FLUX.1-schnell 4-step floating-point exception, no upstream fix. The closest direct confirmation that distilled 4-step is unsupported in the reference implementation. `https://github.com/ali-vilab/TeaCache/issues/83`

## Lessons

These show up as feedback memories so they apply across future projects:

- **Performance claims need a committed benchmark.** Added to global `~/.claude/CLAUDE.md` on 2026-05-16. Every model / variant we claim to accelerate needs a committed `scripts/bench_*.py` with warmup + ≥3 timed reps, captured skip telemetry, and a README reproducer command.
- **Quality gates must include a "did the feature engage" assertion.** SSIM passing while skipped_count == 0 means the feature is dormant; the test passed by accident. Future image-quality tests on FLUX.2 should assert `skipped_count > 0` or explicitly document why 0 skips is the expected outcome.
- **The v0.2.0 `TeaCacheNoBenefitWarning` was too narrow.** It catches schedule-shape problems but not polynomial-vs-threshold mismatches. A second variant of the warning ("polynomial doesn't dip below threshold over its calibration range") could fire at `apply_teacache` time and tell users they need to bump threshold before they spend wall-clock on a no-op gate.
