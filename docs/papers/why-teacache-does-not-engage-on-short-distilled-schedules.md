# Why the TeaCache Gate Did Not Engage on Short Distilled FLUX Schedules

*A negative result from `mlx-teacache`, and the measurement practice it forced*

> 📄 [Read on the website](https://ineshin.space/papers/why-teacache-does-not-engage-on-short-distilled-schedules/) — same paper, formatted for reading.

## Update, 2026-11-13: what changed in this paper

This revision corrects two statements and removes figures that have no committed results file. The
[previous version](https://github.com/IonDen/mlx-teacache/blob/b8fc4ff/docs/papers/why-teacache-does-not-engage-on-short-distilled-schedules.md)
gave a cause for the withdrawn 2.68× figure on Klein base 9B. The tracked record of that run does
not support it: the run used subprocess-per-condition isolation at 768 × 1024 on another prompt,
once, while the later 1.36× measurement ran at 512 × 512. Section 4 now states what the record
shows and that the cause was not isolated. The previous version also said the project shipped a
wrong attribution only in the release that lacked skip telemetry; a later attribution carried
telemetry and was still wrong, and section 7 now says so. The benchmark seconds of the first
distilled runs, a throttling account, a threshold sweep and a fitted intercept are cut or
described without figures. Six statements with no public source are gone as well: that the
v0.2.0 figure came from a single ad-hoc timing run, that nothing in the test suite then asserted a
skip, that the upstream TeaCache repository has no FLUX.2 path, that issue #83 concerned
FLUX.1-schnell at four steps, a reference to a Runware Klein 9B page, and a claim of visually
equivalent output. Figure 1 no longer calls its skip counts stable across repetitions, and it
labels the Klein count as reported in the release notes. Figures that later releases remeasured
are dated to their run. Evidence links are pinned to release tags: v0.17.0 for most, v0.3.0 for
the first benchmark script, and v0.5.0 for the record of the 2.68× run. The gate result and the
argument are unchanged.

---

A wall-clock improvement can be real while its explanation is wrong. In May 2026 the
`mlx-teacache` project shipped, and then corrected, exactly that combination: its TeaCache wrapper
ran faster than vanilla [mflux](https://github.com/filipstrand/mflux) on FLUX.2 Klein's
distilled 8-step schedule, and the caching mechanism the library exists to provide never fired.
Every eligible denoising step computed in full. The speed difference therefore could not be
attributed to caching; source inspection pointed to an unrelated execution-path difference that
the benchmark of the day could not isolate.

This note documents the negative result: why the TeaCache polynomial gate, with its shipped
input signal, calibrated fits, and default threshold, did not engage on the tested 8-step
distilled Klein trajectories, and why zero skips was consistent with the gate's decision rule
rather than evidence of a software fault. It then lays out what can and cannot be concluded
about the wall-clock difference, and it closes with the methodological lesson, which
generalizes past diffusion caching: a feature's benefit must be attributed by measuring whether
the feature engaged, not by assuming the headline mechanism did the work.

The note is documentary. Its numbers come from the project's committed calibration files and
benchmark reports, linked at release tags; where a result has no committed results file, the
note describes what was observed and how it is known, and quotes no figure. Nothing was rerun
for this write-up. The first distilled runs came from the v0.3.0 benchmark script, which fixes
4-bit quantization, 512×512 output, one prompt and seed, a warmup, and three timed repetitions per
condition, all in one Python process; the v0.3.0 release notes name an M1 Max with 32 GB as the
machine, and the runs' mflux version was not saved with them. The later cold-harness reports cited
below (v0.6.0, v0.6.3, v0.10.0 and v0.12.0) ran on an Apple M1 Max with 32 GB of unified memory at
4-bit, 512×512, with one prompt and seed shared across them, each condition and repetition cold in
a fresh subprocess, and report the median of three repetitions. Each of those reports records its
mflux version and the Darwin kernel release; none records the MLX version. The record of the
withdrawn 2.68× run differs on several of these points and is described on its own terms in
section 4.

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

Where the premise holds, the mechanism does what the paper says. In the project's v0.6.3 run of
FLUX.1-dev at 25 steps (mflux 0.17.5, cold subprocess per repetition, three repetitions;
[`_artifacts/v0.6.3_bench_flux1_dev.json`](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/_artifacts/v0.6.3_bench_flux1_dev.json)),
the gate skipped 6 of 25 steps in each of the three repetitions at the default threshold, for a 1.46× wall-clock
speedup; a three-way decomposition in the same report attributes 1.47× to step-skipping and 1.00×
to everything else. A later v0.10.0 run on mflux 0.18.0 measured 1.57× with the same 6 of 25 skips
([`_artifacts/v0.10.0_bench_flux1_dev.json`](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/_artifacts/v0.10.0_bench_flux1_dev.json));
its vanilla repetitions spread from 104.7 s to 119.8 s, so its decomposition is less clean. The
gate only fires where the predicted per-step change is small relative to the threshold.

![Flow diagram of the TeaCache gate decision: a cheap input signal feeds a calibrated polynomial, the prediction accumulates, and a threshold check either skips the transformer body or computes the step in full. On FLUX.1-dev's 25-step schedule the skip branch is reached on some steps; on the 8-step distilled Klein schedules the compute branch is the only branch reached in the reported runs, and no step is skipped](https://raw.githubusercontent.com/IonDen/mlx-teacache/main/docs/papers/diagrams/gate-engagement-two-schedules.svg)

*Figure 1. The gate's per-step decision, and where each tested schedule ends up. The decision
flow is conceptual structure. The FLUX.1-dev count is the one the v0.6.3 run recorded in each of
its three repetitions; the Klein branch summarizes the section 3 calibration finding for 9B (the
shipped origin-constrained fit predicts above the 0.20 default threshold over the observed
input-signal range) together with the zero skips the v0.3.0 release notes report for 4B and 9B,
not a separate measurement.
The editable
[PlantUML source](https://github.com/IonDen/mlx-teacache/blob/main/docs/papers/diagrams/gate-engagement-two-schedules.puml)
is published with the paper.*

## 2. The observation: faster wall clock, zero skips

The project's v0.2.0 release described a "~1.2× speedup" on FLUX.2 Klein 4B as a TeaCache caching
outcome, as the v0.3.0 [changelog](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/CHANGELOG.md)
records in its correction. No benchmark script was committed with that figure, so it could not be
re-derived.

While preparing v0.3.0 (FLUX.2 Klein 9B support), the project committed a reproducible benchmark:
[`scripts/bench_speedup.py` at v0.3.0](https://github.com/IonDen/mlx-teacache/blob/v0.3.0/scripts/bench_speedup.py) pins seed, prompt,
dimensions, and step count, and reports per-repetition skip counts next to the timings. Its first
runs across the supported variants showed the split the rest of this note explains: FLUX.1-dev
skipped steps, while both distilled Klein variants skipped none and still finished ahead of
vanilla mflux. The v0.3.0 release notes report the result that way. The raw report of those runs
is not public, so this note quotes no timings or ratios from them.

The wall-clock side of that result deserves less trust than the skip side. All vanilla
repetitions ran first, then all wrapper repetitions, in one Python interpreter, so thermal state,
condition order, and allocator state were shared between the two arms, and nothing in the record
separates them from the wrapper's effect. Those rows have not been repeated under the later cold
subprocess harness, and the project's
[README benchmark notes](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/README.md#benchmarks)
quote no timing for them. This note therefore treats the distilled rows only as evidence that the
wrapper was faster in those runs, not as a precise or reusable magnitude.

## 3. Zero skips is the gate working correctly

The v0.3.0 calibration recipe captured, for each variant, 10 prompts × 8 steps at a fixed seed:
70 consecutive-step pairs of (input-signal relative-L1, body-output relative-L1).

For Klein 9B, the empirical body-output change between adjacent steps spans **0.250 to 0.880**
(the raw arrays are committed in
[`scripts/_calibration_flux2_klein_9b.json`](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/scripts/_calibration_flux2_klein_9b.json)).
Read that against the 0.20 default threshold: in the recorded sample, every adjacent pair in the
8-step trajectories changed by about 25% relative-L1 or more. That empirical floor does not by itself
determine what a fitted polynomial predicts between observations. The gate acts on the fitted
prediction, not directly on the empirical body-output distance.

The project fit the polynomial twice during the diagnosis:

- The unconstrained fit predicted an implausibly large output change at zero input change, as the
  project's [calibration notes](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/docs/calibration.md)
  record. It was rejected.
- The origin-constrained fit (R² = 0.4710, forced through `poly(0) = 0`, with coefficients shipped
  in v0.3.0) stayed above 0.20 over the observed input range of 0.1316–0.4498.

The second claim can be checked by hand. The package ships the Klein 9B coefficients highest
degree first as −523.84, 530.25, −177.64, 20.89 and 0
([`config.py` at v0.17.0](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/src/mlx_teacache/variants/flux2_klein_9b/config.py#L21-L27)),
so with x the input-signal relative-L1 the predicted output change is

```text
p(x) = −523.84·x⁴ + 530.25·x³ − 177.64·x² + 20.89·x

p(0.1316) ≈ 0.72   bottom of the observed range
p(0.28)   ≈ 0.34   near the interior minimum
p(0.39)   ≈ 0.46   near the interior maximum
p(0.4498) ≈ 0.27   top of the observed range
```

The curve falls to about 0.34, rises to about 0.46 and falls again before the range ends, so its
lowest value on the observed range is about 0.27 at the top of the range, still above 0.20.

The shipped fit's above-threshold predictions, together with the zero skips the release notes
report, explain the 9B gate outcome at the default threshold. Klein 4B has the same documented
outcome: its committed summary records an empirical output-change range of 0.261–1.008
([`scripts/_calibration_flux2_klein_4b.json`](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/scripts/_calibration_flux2_klein_4b.json)),
and the release notes report no skips at the default threshold. That calibration file is
summary-only, so this note does not claim a point-by-point reconstruction of the 4B fit.

FLUX.1-schnell is a related but weaker observation. Its
[variant page](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/docs/variants/flux1-schnell.md)
reports that the gate skips nothing at the 4-step default, where the skip-first and skip-last
windows leave only two gate-eligible steps. It has no calibration artifact comparable to the
Klein 9B raw sample, so it is not part of the quantitative Klein result.

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
To keep the gate's per-step decision live, `mlx-teacache` replaces that compiled function with an
eager Python closure (today built by
[`make_teacache_predict_factory`](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/src/mlx_teacache/integrations/mflux/forward_flux2.py#L395)):
a compiled function would bake the first step's gate decision into the trace. On chips where
mflux compiles, the side effect is that the wrapper also sidesteps the compiled execution path
entirely. In the v0.3.0 runs (4-bit quantization, 8-step schedule, on the M1 Max the release
notes name) the wrapper's eager path finished ahead of vanilla mflux's compiled path. Zero skips
rules out caching as the cause. Source inspection makes the eager-versus-compiled execution-path
difference the leading explanation, but the 8-step benchmark did not isolate that effect from
thermal state, condition order, allocator state, or other wrapper differences.

Two later corrections sharpened the attribution.

First, in v0.6.0 the project rebuilt its benchmark as subprocess-per-repetition: every (variant,
condition, repetition) triple runs in a fresh Python interpreter, so each timing starts from a
cold state. Under that harness, the eager-versus-compiled difference on 50-step non-distilled
Klein recipes contributes only **1.01–1.02×**
([`_artifacts/v0.6.0_bench_klein_base_4b.json`](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/_artifacts/v0.6.0_bench_klein_base_4b.json),
[`_artifacts/v0.6.0_bench_klein_base_9b.json`](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/_artifacts/v0.6.0_bench_klein_base_9b.json)).
This shows that the effect is small on those recipes. The rebuilt harness did not remeasure the
8-step distilled rows. Fixed per-step overhead taking a larger share of a short run is a plausible
explanation for their larger apparent difference, but the original measurements remain
single-interpreter and confounded, so that explanation is a hypothesis rather than a result.

Second, a headline elsewhere in the project was withdrawn. v0.5.0 advertised 2.68× for Klein
base 9B, the non-distilled 9B variant. Its tracked record,
[`_artifacts/validation_klein_base_9b.json`](https://github.com/IonDen/mlx-teacache/blob/v0.5.0/_artifacts/validation_klein_base_9b.json)
(first tagged in v0.5.0), is a single run with subprocess-per-condition isolation at 768 × 1024 on a
portrait prompt, with a 24 GB memory-cap setting recorded next to a vanilla peak of 25.21 GB. It
records the Darwin kernel release but neither an mflux version nor a quantization setting. The
later 1.36× measurement ran at 512 × 512 on a different prompt, three cold repetitions per
condition (`v0.6.0_bench_klein_base_9b.json`, above), and the v0.12.0 rerun of that recipe measured
1.37×
([`_artifacts/v0.12.0_bench_klein_base_9b.json`](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/_artifacts/v0.12.0_bench_klein_base_9b.json)).
Resolution, prompt, repetition count, harness and the memory-cap setting all differ between the
2.68× and the 1.36×, the 2.68× run's mflux version is unknown, and the cause of the difference
was not isolated. The record supports only that the 2.68× was not reproduced at a different,
smaller recipe.

The corrected pieces establish a narrower picture. In the v0.6.x three-way runs (mflux 0.17.5),
step-skipping is the dominant mechanism where it engages (a 1.47× step-skipping factor within
FLUX.1-dev's combined 1.46×; 1.22–1.34× within the non-distilled Klein results), while execution-path avoidance contributes only 1.00–1.02× on
those 25–50-step recipes. The v0.12.0 reruns of the Klein base recipes (mflux 0.18.0) gave the same
split: 1.18× and 1.35× from step-skipping, 1.01× and 1.02× from the execution path
([`_artifacts/v0.12.0_bench_klein_base_4b.json`](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/_artifacts/v0.12.0_bench_klein_base_4b.json),
`v0.12.0_bench_klein_base_9b.json`). For the earlier 8-step rows, the firm conclusion is only
that caching contributed nothing because the gate skipped no step; the magnitude and exclusivity
of the execution-path effect were not isolated.

## 5. Why every gate passed anyway

The v0.3.0 finding came from the new benchmark, not from the test suite. The checks then in place
answered different questions.

The image-quality gates (SSIM against vanilla output) pass *trivially* when the cache never
engages: with zero skips, the wrapper's output differs from vanilla only by kernel-dispatch
noise. A quality assertion cannot distinguish "the feature preserved quality" from "the feature
did nothing." An engagement assertion can; today's FLUX.2 image-quality tests carry one,
`skipped_count >= 1`
([`tests/test_image_quality_flux2.py`](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/tests/test_image_quality_flux2.py#L261)).

At v0.3.0 the library's no-benefit warning checked schedule shape (whether the step count minus
the protected windows left any skippable steps), not gate behavior. Klein 9B at 8 steps has five
skippable steps in principle, so no warning fired; the polynomial simply never chose to use them.
Since v0.10.0, `apply_teacache` warns at apply time on both distilled Klein variants
([`api.py` at v0.17.0](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/src/mlx_teacache/api.py#L117-L127)).

Until v0.3.0 there was no committed benchmark script, so the v0.2.0 figure could not be
re-derived, and it went into the documentation without one.

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
[ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) repository's
[issue #83](https://github.com/ali-vilab/TeaCache/issues/83) records an unresolved floating-point
exception; it does not establish non-engagement or the same failure mechanism. The
[TeaCache paper](https://arxiv.org/abs/2411.19108) evaluates video models rather than distilled
FLUX schedules. [NVIDIA's published FLUX.2-dev configuration](https://developer.nvidia.com/blog/scaling-nvfp4-inference-for-flux-2-on-nvidia-blackwell-data-center-gpus/)
uses 50 steps and `teacache_thresh=0.05`, skipping an average of 16 of 50 steps across 20 prompts;
it is not a Klein result. Runware exposes TeaCache controls for
[Klein 4B](https://runware.ai/docs/models/bfl-flux-2-klein-4b), but its page publishes no skip-rate
or quality calibration for the controls. None of these linked sources demonstrates step-level
caching engaging on a 4–8-step distilled FLUX schedule.

The project's resolution shipped across several releases. v0.3.0 corrected the v0.2.0 mechanism
framing in the README and changelog, published skip telemetry alongside every timing, and
declared distilled schedules out of scope for step-skipping. Subsequent releases redirected the
library to schedules where the premise holds: in the v0.12.0 cold runs the non-distilled Klein
base 4B and 9B variants skip 8 and 13 of 48 active steps at their 50-step recipes, for 1.20× and
1.37× (the two v0.12.0 reports above), and later non-FLUX variants engage as well. For the tested
Klein recipes, the implementation followed its decision rule; the shipped gate configuration and
trajectory did not produce a skip.

| Claim | Status | Source |
|---|---|---|
| No skips at the default threshold, Klein 4B and 9B at 8 steps | reported in the release notes; the raw v0.3.0 report is not public, so no count is quoted | [changelog](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/CHANGELOG.md); [README](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/README.md#benchmarks) |
| Klein 9B adjacent-step output change 0.250–0.880 | committed calibration data (raw arrays) | [`scripts/_calibration_flux2_klein_9b.json`](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/scripts/_calibration_flux2_klein_9b.json) |
| Klein 4B empirical output-change range 0.261–1.008 | committed summary; raw per-pair arrays are not public | [`scripts/_calibration_flux2_klein_4b.json`](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/scripts/_calibration_flux2_klein_4b.json) |
| Distilled-row wall-clock results | wrapper faster in the v0.3.0 runs; single-interpreter and confounded; no ratio quoted | [changelog](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/CHANGELOG.md) |
| Execution-path contribution 1.01–1.02× at 50 steps | v0.6.0 and v0.12.0 runs, cold subprocess per repetition | [v0.6.0 Klein base 9B report](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/_artifacts/v0.6.0_bench_klein_base_9b.json); [v0.12.0 Klein base 9B report](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/_artifacts/v0.12.0_bench_klein_base_9b.json) |
| Withdrawn 2.68× on Klein base 9B and the later 1.36× measurement | single run at 768 × 1024 on another prompt, with a memory-cap setting and no mflux version recorded, against three cold runs at 512 × 512; cause not isolated | [`validation_klein_base_9b.json`](https://github.com/IonDen/mlx-teacache/blob/v0.5.0/_artifacts/validation_klein_base_9b.json); [v0.6.0 report](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/_artifacts/v0.6.0_bench_klein_base_9b.json) |
| Short distilled schedules intrinsically defeat step-level caching | hypothesis consistent with these observations; not established by this study | bounded evidence above |

## 7. Lessons

**Measure whether the feature engaged, not only the wall clock.** If a library claims a
mechanism, its benchmark must count mechanism events (here, skipped steps) next to every
timing. A speedup number without an engagement count cannot attribute anything. An engagement
count is necessary but not sufficient: this project's v0.2.0 framing had no committed benchmark
and credited caching wrongly, and its v0.4.1 attribution for Klein base 4B carried skip counts
and still split the gain wrongly between step-skipping and the execution path until the v0.6.1
correction.

**Quality gates pass when the feature is dormant.** An output-similarity assertion is satisfied
by a no-op. Feature test suites need either an explicit engagement assertion or a documented
statement of why zero engagement is the expected outcome for that configuration.

**Time each condition cold before attributing.** When conditions run one after another in a
single interpreter, thermal state, condition order, compile caches, and allocator state are
shared between the arms, and the measurement cannot separate them from the feature. This note's
8-step rows were never separated from them, and the v0.4.1 decomposition moved when the project
reran it with each repetition in a fresh process. Cold subprocess runs per condition and
repetition gave Klein base 9B figures that held across releases: 1.36× at v0.6.0 and 1.37× at
v0.12.0, with 13 skips in each of the three repetitions of both. Cold isolation alone does not
make a figure stable, though: under the same harness FLUX.1-dev moved from 1.46× to 1.57× between
the v0.6.3 and v0.10.0 runs, on different mflux versions and with a wide vanilla spread in the
later run (section 1).

**Ship the correction in the product's own voice.** The corrected README states, per variant,
what mechanism produces the measured win and where the feature does not help. That includes
telling users to run distilled Klein through vanilla mflux. A negative result documented where users read
is worth more than a quiet fix.

## References and source notes

- Committed calibration files, cold-isolated benchmark reports, and the benchmark harness in the
  `mlx-teacache` repository, at v0.17.0:
  [`scripts/bench_speedup.py`](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/scripts/bench_speedup.py),
  [`scripts/_calibration_flux2_klein_9b.json`](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/scripts/_calibration_flux2_klein_9b.json),
  [`scripts/_calibration_flux2_klein_4b.json`](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/scripts/_calibration_flux2_klein_4b.json),
  and the versioned benchmark reports under [`_artifacts/`](https://github.com/IonDen/mlx-teacache/tree/v0.17.0/_artifacts).
  The `_artifacts/` reports support the later runs, not the historical distilled timing rows.
- Released documentation at v0.17.0: [README](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/README.md)
  ("Benchmarks" and "How the speedup happens"), [changelog](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/CHANGELOG.md) (v0.3.0
  correction of the v0.2.0 framing; v0.6.0 harness change; v0.6.1 correction of the v0.4.1
  decomposition; v0.10.0 apply-time warning),
  [`docs/calibration.md`](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/docs/calibration.md), and the
  [per-variant pages](https://github.com/IonDen/mlx-teacache/tree/v0.17.0/docs/variants).
- TeaCache: [arXiv:2411.19108](https://arxiv.org/abs/2411.19108) (CVPR 2025);
  [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) and
  [issue #83](https://github.com/ali-vilab/TeaCache/issues/83).
- [NVIDIA's FLUX.2 inference-scaling article](https://developer.nvidia.com/blog/scaling-nvfp4-inference-for-flux-2-on-nvidia-blackwell-data-center-gpus/)
  (TeaCache threshold on non-distilled FLUX.2-dev).
- [Hugging Face diffusers cache documentation](https://huggingface.co/docs/diffusers/api/cache)
  (FirstBlockCache and related block-level techniques).
- Runware's [Klein 4B](https://runware.ai/docs/models/bfl-flux-2-klein-4b) API documentation
  (exposed cache controls without published skip-rate or quality calibration).

---

*Prepared 2026-07-19. Last updated 2026-11-13. Denis Ineshin.*
