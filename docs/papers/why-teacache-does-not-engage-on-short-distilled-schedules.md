# Why the TeaCache Gate Did Not Engage on Short Distilled FLUX Schedules

*A negative result from `mlx-teacache`, and the measurement practice it forced*

> 📄 [Read on the website](https://ineshin.space/papers/why-teacache-does-not-engage-on-short-distilled-schedules/) — same paper, formatted for reading.

A wall-clock improvement can be real while its explanation is wrong. In May 2026 the
`mlx-teacache` project shipped, and then corrected, exactly that combination: its TeaCache wrapper
ran measurably faster than vanilla [mflux](https://github.com/filipstrand/mflux) on FLUX.2 Klein's
distilled 8-step schedule, and the caching mechanism the library exists to provide never fired
once. Every eligible denoising step computed in full. Zero steps were skipped. The speed difference
therefore could not be attributed to caching; source inspection pointed to an unrelated
execution-path difference that the benchmark of the day could not isolate.

This note documents the negative result: why the TeaCache polynomial gate, with its shipped
input signal, calibrated fits, and default threshold, did not engage on the tested 8-step
distilled Klein trajectories, and why zero skips was consistent with the gate's decision rule
rather than evidence of a software fault. It then lays out what can and cannot be concluded
about the wall-clock difference, and it closes with the methodological lesson, which
generalizes past diffusion caching: a feature's benefit must be attributed by measuring whether
the feature engaged, not by assuming the headline mechanism did the work.

The note is documentary. Numbers are source-reported from the project's public calibration files,
benchmark reports, and release documentation, cross-checked against the released
[README](https://github.com/IonDen/mlx-teacache/blob/main/README.md),
[changelog](https://github.com/IonDen/mlx-teacache/blob/main/CHANGELOG.md), and
[per-variant documentation](https://github.com/IonDen/mlx-teacache/tree/main/docs/variants);
nothing was rerun for this write-up. The v0.3-era distilled measurements used an M1 Max with 32 GB
unified memory, mflux 0.17.5, 4-bit quantization, 512×512 output, a fixed prompt and seed, a warmup,
and three timed repetitions per condition in one process. The later v0.6.x controls cited below
used the same chip, memory size, mflux version, quantization, resolution, prompt, and seed, but ran
each condition and repetition cold in a fresh subprocess and reported the median of three runs.

## 1. The gate and its premise

TeaCache ([arXiv:2411.19108](https://arxiv.org/abs/2411.19108)) starts from an empirical property
of diffusion sampling: across much of the denoising trajectory, consecutive transformer outputs
are nearly the same. The expensive transformer body produces a residual that is added to its
input, and that residual stays roughly stable for stretches of adjacent steps. TeaCache trains a
small polynomial that predicts how much the body output will change from how much a cheap input
signal changed. In the FLUX implementations that signal is the relative-L1 distance of the
modulated block-0 input between the current and previous step. The gate accumulates the predicted output change
since the last fully computed step. While the accumulated prediction stays under a threshold
(`rel_l1_thresh`, package default 0.20), the wrapper reuses the cached residual and skips the
transformer body; when it crosses the threshold, the step computes in full and the accumulator
resets.

Where the premise holds, the mechanism does what the paper says. On FLUX.1-dev at 25 steps the
gate skips 6 of 25 steps at the default threshold, for a measured 1.46× wall-clock speedup with
visually equivalent output (subprocess-isolated harness, three repetitions; a three-way
decomposition attributes 1.47× to step-skipping and 1.00× to everything else). Engagement is
threshold-banded even there: single-run sweeps at thresholds 0.10 and 0.15 recorded zero skips on
the same 25-step schedule. The gate only fires inside a band where the predicted per-step change
is small relative to the threshold.

![Flow diagram of the TeaCache gate decision: a cheap input signal feeds a calibrated polynomial, the prediction accumulates, and a threshold check either skips the transformer body or computes the step in full. On FLUX.1-dev's 25-step schedule the skip branch is reached on 6 of 25 steps; on the 8-step distilled Klein schedules the compute branch is the only branch ever reached, with 0 of 8 steps skipped](https://raw.githubusercontent.com/IonDen/mlx-teacache/main/docs/papers/diagrams/gate-engagement-two-schedules.svg)

*Figure 1. The gate's per-step decision, and where each tested schedule ends up. The decision
flow is conceptual structure; the two skip counts are measured, repetition-stable results. The
Klein branch summarizes the section 3 calibration finding (the shipped origin-constrained fit
predicts above the 0.20 default threshold over the observed input-signal range), not a separate
measurement. The editable
[PlantUML source](https://github.com/IonDen/mlx-teacache/blob/main/docs/papers/diagrams/gate-engagement-two-schedules.puml)
is published with the paper.*

## 2. The observation: faster wall clock, zero skips

The project's v0.2.0 release described a "~1.2× speedup" on FLUX.2 Klein 4B as a TeaCache caching
outcome. That figure came from a single ad-hoc timing run. No benchmark script was committed, and
no skip telemetry was captured alongside the wall clock.

While preparing v0.3.0 (FLUX.2 Klein 9B support), the project committed a reproducible benchmark:
[`scripts/bench_speedup.py` at v0.3.0](https://github.com/IonDen/mlx-teacache/blob/v0.3.0/scripts/bench_speedup.py) pins seed, prompt,
dimensions, and step count, and records per-repetition skip counts next to the timings. Its first
run across the supported variants produced this:

| Variant | Steps | Vanilla median | Wrapper median | Skipped steps |
|---|---|---|---|---|
| FLUX.1-dev | 25 | 103.67 s | 71.79 s | **6 / 25** |
| FLUX.2 Klein 4B (distilled) | 8 | 28.1 s | 22.3 s | **0 / 8** |
| FLUX.2 Klein 9B (distilled) | 8 | 119.0 s | 61.8 s | **0 / 8** |

The load-bearing column is the last one. The skip counts were stable across every repetition:
FLUX.1-dev genuinely cached, and both distilled Klein variants computed every eligible step while
still finishing ahead of vanilla mflux.

The wall-clock ratios in that table deserve less trust than the skip counts, for reasons the
project only fully understood later. These were same-process measurements (all vanilla
repetitions, then all wrapper repetitions, in one Python interpreter), and the Klein 9B median in
particular combined a thermally throttled vanilla repetition (227 s) with a recovered wrapper
repetition (46 s), inflating the apparent ratio to 1.93×. The original raw distilled benchmark
report is not public, and those rows have not been repeated under the later cold subprocess
harness. This note therefore treats the distilled-row ratios only as evidence that the wrapper
was faster in those runs, not as precise or reusable magnitudes. The zero-skip result, by
contrast, is recorded in the release documentation as stable across repetitions.

## 3. Zero skips is the gate working correctly

The v0.3.0 calibration recipe captured, for each variant, 10 prompts × 8 steps at a fixed seed:
70 consecutive-step pairs of (input-signal relative-L1, body-output relative-L1).

For Klein 9B, the empirical body-output change between adjacent steps spans **0.250 to 0.880**
(the raw arrays are committed in
[`scripts/_calibration_flux2_klein_9b.json`](https://github.com/IonDen/mlx-teacache/blob/main/scripts/_calibration_flux2_klein_9b.json)).
Read that against the 0.20 default threshold: in the recorded sample, every adjacent pair in the
8-step trajectories changed by at least 25% relative-L1. That empirical floor does not by itself
determine what a fitted polynomial predicts between observations. The gate acts on the fitted
prediction, not directly on the empirical body-output distance.

The project fit the polynomial twice during the diagnosis:

- The unconstrained fit produced `poly(0) ≈ 5.36`, an implausibly large predicted output change at
  zero input change. It was rejected.
- The origin-constrained fit (R² = 0.4710, forced through `poly(0) = 0`, with coefficients shipped
  in v0.3.0) stayed above 0.20 over the observed input range of 0.1316–0.4498.

The shipped fit's above-threshold predictions, together with the recorded zero-skip telemetry,
explain the 9B gate outcome at the default threshold. Klein 4B has the same documented outcome:
its public summary records an empirical output-change range of 0.261–1.008, and the release
documentation records zero skips at the default threshold. Its committed calibration report is
summary-only, so this note does not claim a point-by-point reconstruction of the 4B fit.

FLUX.1-schnell is a related but weaker observation. Its public variant page records zero skips at
the 4-step default, where the skip-first and skip-last windows leave only two gate-eligible steps.
It has no calibration artifact comparable to the Klein 9B raw sample, so it is not part of the
quantitative Klein result.

Distillation is a plausible explanation for these observations. A distilled model compresses the
denoising trajectory into a handful of consequential strides, which is consistent with larger
changes between adjacent steps. This study does not isolate distillation from model family,
scheduler, gate signal, polynomial form, or step count, so it does not establish that distillation
itself removes all cacheable similarity.

Raising the threshold is not a characterized remedy. Engagement begins only when the threshold
exceeds some accumulated fitted predictions; the empirical 0.25 floor is not that decision
boundary. No threshold sweep established where skips begin or whether their image quality is
acceptable on these schedules. The project therefore kept distilled schedules out of scope for
step-skipping.

## 4. What the evidence says about the speed

[mflux 0.17.5 wraps the FLUX.2 predict function in `mx.compile`](https://github.com/filipstrand/mflux/blob/v.0.17.5/src/mflux/models/flux2/variants/txt2img/flux2_klein.py)
on every Apple Silicon chip except base and Pro M1/M2, using its
[chip-family predicate](https://github.com/filipstrand/mflux/blob/v.0.17.5/src/mflux/utils/apple_silicon.py).
To keep the gate's per-step decision live, [`mlx-teacache` replaces that compiled function with an
eager Python closure](https://github.com/IonDen/mlx-teacache/blob/main/src/mlx_teacache/variants/flux2_klein_4b/integration.py):
a compiled function would bake the first step's gate decision into the trace. On chips where
mflux compiles, the side effect is that the wrapper also sidesteps the compiled execution path
entirely. On the test machine (M1 Max, 4-bit quantization,
8-step schedule) the wrapper's eager path ran faster than vanilla mflux's compiled path in the
same-process measurements. Zero skips rules out caching as the cause. Source inspection makes the
eager-versus-compiled execution-path difference the leading explanation, but the 8-step benchmark
did not isolate that effect from thermal state, condition order, allocator state, or other wrapper
differences.

Two later corrections sharpened the attribution.

First, the project rebuilt its benchmark as subprocess-per-repetition: every (variant, condition,
repetition) triple runs in a fresh Python interpreter, so each timing starts from a cold state.
Under that harness, the eager-versus-compiled difference on 50-step non-distilled Klein recipes
contributes only **1.01–1.02×**. This shows that the effect is small on those recipes. The
rebuilt harness did not remeasure the 8-step distilled rows. Fixed per-step overhead taking a larger share of a short run
is a plausible explanation for their larger apparent difference, but the original measurements
remain same-process and confounded, so that explanation is a hypothesis rather than a result.

Second, the harness change independently corrected a headline elsewhere in the project: a 2.68×
claim on a non-distilled 9B variant collapsed to **1.36×** once both conditions ran cold. Under
the same-process ordering, the compiled vanilla condition had paid MLX's full cold-start
compilation cost, while the wrapper condition, running second, inherited warm allocator state
from it. Two separate numbers, one measurement pathology: same-process A/B timing on MLX
misattributes.

The corrected pieces establish a narrower picture. In the clean three-way measurements,
step-skipping is the dominant mechanism where it engages (1.47× of FLUX.1-dev's 1.46×;
1.22–1.34× of the non-distilled Klein results), while execution-path avoidance contributes only
1.00–1.02× on those 25–50-step recipes. For the earlier 8-step rows, the firm conclusion is only
that caching contributed nothing because the skip count was zero; the magnitude and exclusivity
of the execution-path effect were not isolated.

## 5. Why every gate passed anyway

The v0.3.0 finding was not caught by the test suite, and could not have been, because every
existing check answered a different question.

The image-quality gates (SSIM against vanilla output) pass *trivially* when the cache never
engages: with zero skips, the wrapper's output differs from vanilla only by kernel-dispatch
noise. A quality assertion cannot distinguish "the feature preserved quality" from "the feature
did nothing." Nothing in the suite asserted `skipped_count > 0`.

The library's no-benefit warning checked schedule shape (whether the step count minus the
protected windows left any skippable steps), not gate behavior. Klein 9B at 8 steps has five
skippable steps in principle, so no warning fired; the polynomial simply never chose to use them.

And until v0.3.0 there was no committed benchmark at all, so the v0.2.0 claim was never
re-derivable. A single unreproducible timing fossilized into documentation.

The general form of the failure: a feature can be structurally exercised (the wrapper installs,
runs, restores, and produces correct images) while never functionally engaging. Tests that only
inspect outputs cannot tell those apart. The fix shipped in v0.3.0 as a reporting change and a
policy: skip counts published next to every wall-clock number, an explicit mechanism attribution
per benchmark row, and a benchmark script committed for subsequent measurements.

## 6. Scope, corroboration, and what the project did

The negative result is deliberately narrow. It covers the tested gate signal (modulated block-0
input relative-L1), the shipped degree-4 polynomial fits, the default 0.20 threshold, and the
distilled Klein 4B and 9B 8-step recipes on the stated hardware. The separately documented
FLUX.1-schnell result has less supporting evidence and is not included in that quantitative scope.
This note does **not** claim that all caching techniques fail on distilled diffusion models. Block-level
approaches such as FirstBlockCache
([diffusers cache documentation](https://huggingface.co/docs/diffusers/api/cache)) skip part of
every step rather than whole steps, a different premise this project has not tested.

External sources give context, not direct corroboration of the gate result. The upstream
[ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) repository has no FLUX.2 path, and its
[issue #83](https://github.com/ali-vilab/TeaCache/issues/83) records an unresolved floating-point
exception from a user running FLUX.1-schnell at four steps; it does not establish non-engagement or
the same failure mechanism. The [TeaCache paper](https://arxiv.org/abs/2411.19108) evaluates video
models rather than distilled FLUX schedules. [NVIDIA's published FLUX.2-dev configuration](https://developer.nvidia.com/blog/scaling-nvfp4-inference-for-flux-2-on-nvidia-blackwell-data-center-gpus/)
uses 50 steps and `teacache_thresh=0.05`, skipping an average of 16 of 50 steps across 20 prompts;
it is not a Klein result. Runware exposes TeaCache controls for
[Klein 4B](https://runware.ai/docs/models/bfl-flux-2-klein-4b) and
[Klein 9B](https://runware.ai/docs/models/bfl-flux-2-klein-9b), but those pages publish no skip-rate
or quality calibration for the controls. None of these linked sources demonstrates step-level
caching engaging on a 4–8-step distilled FLUX schedule.

The project's resolution shipped across two releases. v0.3.0 corrected the v0.2.0 mechanism
framing in the README and changelog, published skip telemetry alongside every timing, and
declared distilled schedules out of scope for step-skipping. Subsequent releases redirected the
library to schedules where the premise holds: the non-distilled Klein variants skip 9–13 of 48
active steps at 50-step recipes (1.23× and 1.36× under the cold-isolated harness), and later
non-FLUX variants engage similarly. For the tested Klein recipes, the implementation followed its
decision rule; the shipped gate configuration and trajectory did not produce a skip.

| Claim | Status | Source |
|---|---|---|
| 0 skips across all repetitions, Klein 4B and 9B at 8 steps | source-reported measurement preserved in release documentation; the original raw report is not public | [README](https://github.com/IonDen/mlx-teacache/blob/main/README.md); [changelog](https://github.com/IonDen/mlx-teacache/blob/main/CHANGELOG.md) |
| Klein 9B adjacent-step output change 0.250–0.880 | committed calibration data (raw arrays) | [`scripts/_calibration_flux2_klein_9b.json`](https://github.com/IonDen/mlx-teacache/blob/main/scripts/_calibration_flux2_klein_9b.json) |
| Klein 4B empirical output-change range 0.261–1.008 | committed summary; raw per-pair arrays are not public | [`scripts/_calibration_flux2_klein_4b.json`](https://github.com/IonDen/mlx-teacache/blob/main/scripts/_calibration_flux2_klein_4b.json) |
| distilled-row wall-clock ratios (1.26×, 1.93×) | thermally confounded, same-process era; superseded for attribution | [changelog](https://github.com/IonDen/mlx-teacache/blob/main/CHANGELOG.md); [README benchmark footnotes](https://github.com/IonDen/mlx-teacache/blob/main/README.md#benchmarks) |
| execution-path contribution 1.01–1.02× at 50 steps | subprocess-isolated measurement | [v0.6.0 benchmark artifacts](https://github.com/IonDen/mlx-teacache/tree/main/_artifacts) |
| Short distilled schedules intrinsically defeat step-level caching | hypothesis consistent with these observations; not established by this study | bounded evidence above |

## 7. Lessons

**Measure whether the feature engaged, not only the wall clock.** If a library claims a
mechanism, its benchmark must count mechanism events (here, skipped steps) next to every
timing. A speedup number without an engagement count cannot attribute anything; this project
shipped a wrong attribution for exactly one release cycle, the one where telemetry was missing.

**Quality gates pass when the feature is dormant.** An output-similarity assertion is satisfied
by a no-op. Feature test suites need either an explicit engagement assertion or a documented
statement of why zero engagement is the expected outcome for that configuration.

**Same-process A/B timing can misattribute on MLX.** Thermal drift across sequential conditions
inflated one ratio in this project's history; cold-start asymmetry, where one condition pays
compilation and allocator warmup that the next condition inherits, inflated another. Cold
subprocess isolation per condition and repetition is what made the honest numbers reproducible.

**Ship the correction in the product's own voice.** The corrected README states, per variant,
what mechanism produces the measured win and where the feature does not help. That includes
telling users to run distilled Klein through vanilla mflux. A negative result documented where users read
is worth more than a quiet fix.

## References and source notes

- Committed calibration files, later cold-isolated benchmark reports, and the current benchmark
  harness in the `mlx-teacache` repository:
  [`scripts/bench_speedup.py`](https://github.com/IonDen/mlx-teacache/blob/main/scripts/bench_speedup.py),
  [`scripts/_calibration_flux2_klein_9b.json`](https://github.com/IonDen/mlx-teacache/blob/main/scripts/_calibration_flux2_klein_9b.json),
  [`scripts/_calibration_flux2_klein_4b.json`](https://github.com/IonDen/mlx-teacache/blob/main/scripts/_calibration_flux2_klein_4b.json),
  and the versioned benchmark reports under [`_artifacts/`](https://github.com/IonDen/mlx-teacache/tree/main/_artifacts).
  The `_artifacts/` reports support the later controls, not the historical distilled timing rows.
- Released documentation used as documentary cross-checks: [README](https://github.com/IonDen/mlx-teacache/blob/main/README.md)
  ("Benchmarks" and "How the speedup happens"), [changelog](https://github.com/IonDen/mlx-teacache/blob/main/CHANGELOG.md) (v0.3.0
  correction of the v0.2.0 framing; v0.6.0 harness correction),
  [`docs/calibration.md`](https://github.com/IonDen/mlx-teacache/blob/main/docs/calibration.md), and the
  [per-variant pages](https://github.com/IonDen/mlx-teacache/tree/main/docs/variants).
- TeaCache: [arXiv:2411.19108](https://arxiv.org/abs/2411.19108) (CVPR 2025);
  [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) and
  [issue #83](https://github.com/ali-vilab/TeaCache/issues/83).
- [NVIDIA's FLUX.2 inference-scaling article](https://developer.nvidia.com/blog/scaling-nvfp4-inference-for-flux-2-on-nvidia-blackwell-data-center-gpus/)
  (TeaCache threshold on non-distilled FLUX.2-dev).
- [Hugging Face diffusers cache documentation](https://huggingface.co/docs/diffusers/api/cache)
  (FirstBlockCache and related block-level techniques).
- Runware's [Klein 4B](https://runware.ai/docs/models/bfl-flux-2-klein-4b) and
  [Klein 9B](https://runware.ai/docs/models/bfl-flux-2-klein-9b) API documentation (exposed cache
  controls without published skip-rate or quality calibration).

---

*Prepared 2026-07-19. Last updated 2026-07-24. Denis Ineshin.*
