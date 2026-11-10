# Benchmarking Above the Recommended Working Set

*A research memo from `mlx-teacache`: a careful, process-isolated Qwen-Image benchmark ran above the GPU's recommended working set and published a speedup about 12% higher than a later run measured*

> 📄 [Read on the website](https://ineshin.space/papers/benchmarking-above-the-recommended-working-set/) — same paper, formatted for reading.

An earlier memo from this repository,
[Why the TeaCache Gate Did Not Engage on Short Distilled FLUX Schedules](https://github.com/IonDen/mlx-teacache/blob/main/docs/papers/why-teacache-does-not-engage-on-short-distilled-schedules.md),
described a speedup that was real but credited to the wrong cause. This one describes a speedup
that a later run measured lower. For its 0.10.0 release, `mlx-teacache` published two Qwen-Image
figures against plain mflux: a 3.02× speedup, and an unexplained 1.10× lead for the wrapper with
its gate off and no steps skipped. Both came from a run whose MLX active memory peaked above the
GPU's recommended working set. The 0.11.0 re-run changed
several things at once. Its post-load active peak sat 0.83 to 0.88 GiB below that line; adding a
1 GiB cache allowance gives a loose upper bound, not an observation, about 0.1 to 0.2 GiB above it.
The re-run's median ratio was 2.68×, though its per-repetition ratios overlap the earlier run's,
and the 1.10× became 1.04×, which three repetitions cannot tell apart from no difference.

The evidence is two committed benchmark reports,
[`v0.10.0_bench_qwen_image.json`](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/v0.10.0_bench_qwen_image.json)
and
[`v0.11.0_bench_qwen_image.json`](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/v0.11.0_bench_qwen_image.json),
each committed at its release tag (the links show the copies at v0.14.1, where a stored
image-directory path was made repository-relative and no number changed), cross-checked against the
[changelog](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/CHANGELOG.md), the
[README](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/README.md) and the
[Qwen-Image variant page](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/docs/variants/qwen-image.md)
at v0.11.0, the tag that also carries the 0.10.1 changelog entry.

Pre-release builds of the library (see Tables 1 and 2) recorded both reports on an Apple M1 Max
with 32 GB of unified memory, macOS 26 (Darwin 25.6.0), mflux 0.18.0 and a uniform 4-bit
Qwen-Image build computing in bf16. The 0.10.0 report has no commit field. It was committed on
the pull-request branch as 15250be on 2026-09-01 and merged to `main` as fe06b48 on 2026-09-02,
the squash merge that the v0.10.0 tag points to. Its version string, `0.9.4.dev8+g90fca33c9`,
names an older commit and is an install-time stamp, so it does not identify the code that ran.
The 0.11.0 report records commit 1536f28 with uncommitted changes. That commit, also on a
pull-request branch, was made about two minutes after the run's first chunk started, so the field
shows the checkout when the report was written. The report's version string,
`0.10.1.dev13+g05c05a1d9.d20260905`, names yet another commit, and the report was merged to
`main` as 3b86c8b on 2026-09-06. The reports reached the tagged history only through these squash
merges; the two branch commits and the two commits the version strings name are not in it. The
tagged scripts are therefore the released harness, not necessarily the code that produced these
numbers.
Neither report records the MLX version, so the MLX source cited below is a pinned reference, not
the build that ran. Everything here concerns the original Qwen/Qwen-Image checkpoint on mflux
0.18.0. Since release 0.15.0, mlx-teacache requires mflux 0.24 or newer. On those mflux releases,
as the [v0.16.0 variant page](https://github.com/IonDen/mlx-teacache/blob/v0.16.0/docs/variants/qwen-image.md)
explains, the `qwen-image` alias loads Qwen-Image-2512, a separate variant with a default threshold
of 0.25, and mflux's 4-bit Qwen build keeps the image-stream modulation layers at 8 bits, so a
re-run today is not this recipe. Nothing was rerun for this write-up.

## 1. A benchmark that did most things right

TeaCache speeds up a diffusion transformer by skipping steps it predicts will barely change the
result. At each denoising step a cheap signal from the first transformer block estimates how far
the output will move. The gate accumulates that estimate, and while the total stays under a
threshold it skips the block stack and reuses the residual cached from the last computed step. A
threshold of 0 skips nothing. `mlx-teacache` wraps mflux, a community MLX implementation of image
models such as FLUX and Qwen-Image. The first and last of Qwen-Image's 50 steps are always
computed, leaving 48 active steps the gate may skip.

The 0.10.0 number came from
[`scripts/bench_speedup.py`](https://github.com/IonDen/mlx-teacache/blob/v0.10.0/scripts/bench_speedup.py)
in its three-way mode. Every (condition, repetition) pair ran in a fresh Python subprocess, so
each timing started from a cold MLX allocator. There was no warmup: each process loaded the model
and timed one `generate_image` call, its first. Three conditions ran: plain mflux ("vanilla"), the
wrapper at the original Qwen-Image variant's default threshold of 0.30 ("gated"), and the wrapper
at threshold 0 ("no gate"). The no-gate condition separates the wrapper's own speed from the speed
gained by skipping. Every run used the same prompt ("a red apple on a wooden table"), seed 42, 50
steps, guidance 4.0 and 768×768 output.

| Condition | Repetition seconds | Median | Peak MLX active memory | Steps skipped (of 48 active) |
|---|---|---|---|---|
| vanilla | 851.5 / 850.3 / 849.0 | 850.3 s | 26.16 GiB | — |
| no gate | 770.7 / 769.5 / 780.8 | 770.7 s | 26.16 GiB | 0 (changelog) |
| gated (0.30) | 279.2 / 296.4 / 282.0 | 282.0 s | 26.24 GiB | 33 in every repetition |

*Table 1. The 0.10.0 three-way benchmark, from the committed report. Its memory fields say `GB`,
but the [v0.10.0 harness](https://github.com/IonDen/mlx-teacache/blob/v0.10.0/scripts/bench_speedup.py)
divides by 1024³, so the figures are GiB: MLX's own `mx.get_peak_memory()`, active buffers only
(no buffer cache, process or system memory), over the whole run including model load.
Reproducer at the tag:
`scripts/bench_speedup.py --variant qwen --three-way --reps 3` (the variant page's command). The
report's version string, `0.9.4.dev8+g90fca33c9`, does not identify the code that ran (see the
introduction).*

The report derives 3.02× for vanilla over gated, 2.73× for no-gate over gated and 1.10× for
vanilla over no-gate. By the project's own standards (cold processes, repeated runs, skip counts
beside every timing, a second baseline) this was a careful measurement.

## 2. A 1.10× without a known cause

On the obvious suspect, a different computation, the 0.10.0 changelog has one line: "the no-gate
output is pixel-identical to vanilla's". It does not name the images. The tagged harness saves one
image per condition, from the first repetition only, into an untracked directory: one pair at
most, and none a reader can check.

Two facts made the usual explanations less likely. The README and the variant page state that
mflux does not `mx.compile` Qwen-Image's prediction step, so unlike on FLUX.2 Klein and Z-Image
the wrapper had no compiled path to avoid. And at threshold 0 the wrapper runs the same
transformer arithmetic, through its own walk of mflux's forward.

The repository states that sameness twice, at different strengths. A later test, added in
[0.13.2](https://github.com/IonDen/mlx-teacache/blob/v0.13.2/CHANGELOG.md), finds a computed step
under TeaCache bit-identical to a two-block mflux transformer with random weights. For real
weights the v0.11.0 variant page sets a looser bound: "Threshold-0 parity vs vanilla is cosine
≥ 0.99 (`tests/test_parity_qwen.py`), not bit-exact — the eager re-walk diverges from mflux by
Metal-dispatch noise." Both can hold, since a small latent difference can vanish in an 8-bit
image. But the second means the wrapper dispatches work differently from mflux, and that could
change the time.

Within the run, the gap looked nothing like noise. The vanilla repetitions agreed to within 2.4
seconds and the no-gate ones to within 11.3, while the medians differed by 79.6 seconds, about 33
times the vanilla spread. The 0.10.0 changelog calls the mechanism "not established" and advises
reading "the true figure as sitting between the two", 2.73× and 3.02×.

## 3. An earlier sweep points the same way, but does not test the cause

The changelog adds: "the 0.9.0 sweep showed the same zero-skip wrapper 1.18–1.20× ahead of
vanilla." That June 2026 sweep,
[`sweep_threshold_qwen.py`](https://github.com/IonDen/mlx-teacache/blob/v0.9.0/scripts/sweep_threshold_qwen.py),
used a different harness: one vanilla unit first, then one wrapped unit per threshold, so each
speedup is a single repetition. It ran the 0.9.x gate on the mixed-precision build described in
[Qwen-Image Mixed Precision on a 32 GB Mac](https://github.com/IonDen/mlx-teacache/blob/main/docs/papers/qwen-image-mixed-precision-on-a-32-gb-mac.md)
and left no committed artifact and no peak reading. That build peaks at about 30.4 GiB of active
memory at 768×768 here, per the report that memo cites, so the sweep very likely ran above the
line too (an inference). The sweep changed the session, the gate and the build, but probably not
the regime or the grouped order. It agrees with the 1.10× without testing its cause.

## 4. The line the runs crossed

Apple describes
[`recommendedMaxWorkingSetSize`](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)
as "an approximation of how much memory, in bytes, this GPU device can allocate without affecting
its runtime performance", and advises "keeping the total memory footprint of its resources and
heaps less than this threshold value". MLX reports it as `max_recommended_working_set_size` in
`mx.device_info()`. The repository's
[comparison recipes](https://github.com/IonDen/mlx-teacache/blob/v0.14.0/scripts/_comparison_recipes.py)
hard-code 24.96 GiB as the default for a 32 GB M1 Max, and this memo uses that figure as the line;
neither benchmark report records the device's own value, and another Mac reports a different one.
Above the line, Apple's description no longer applies.

Apple's advice concerns the whole footprint of the device's allocations, and MLX's cached buffers
are still device allocations. This memo therefore reports two measures: the active peak, and
active plus cache where the cache can be bounded. For 0.10.0 only the first exists: 26.16 and 26.24
GiB, 1.20 and 1.28 GiB over the line on active memory alone.

None of MLX's three memory controls stops a run there, as the repository's
[cap helper](https://github.com/IonDen/mlx-teacache/blob/v0.14.0/scripts/_mlx_caps.py) and MLX
source at
[revision `ce916db`](https://github.com/ml-explore/mlx/blob/ce916dbbcaa88e433b6fd1e60a17f766d49c27fe/python/src/memory.cpp)
agree:

- `mx.set_wired_limit` bounds only the memory kept resident. The variant page says the pageable
  excess "pages rather than failing", which neither Apple nor MLX documents and nothing here
  measured.
- `mx.set_memory_limit` is "a guideline for the maximum amount of memory to use during graph
  evaluation". It leaves the cache limit alone but moves the reclamation threshold below.
- `mx.set_cache_limit` bounds the pool of freed buffers kept for reuse. Its default is the smaller
  of 1.5 times the recommended set and 0.95 times physical memory, 30.4 GiB here. A lower line
  also applies: when an allocation misses the cache and active plus cached memory would reach a
  reclamation threshold, MLX first releases cached buffers. The
  [Metal allocator](https://github.com/ml-explore/mlx/blob/ce916dbbcaa88e433b6fd1e60a17f766d49c27fe/mlx/backend/metal/allocator.cpp)
  puts that threshold at the smaller of the memory limit and 0.95 times the recommended set,
  about 23.7 GiB here by default.

The v0.10.0 scripts never call `mx.set_cache_limit`, and the variant page dates the cache bound to
0.10.1, so the 0.10.0 processes most likely ran with the default (an inference: the code that
ran is not identified). Both tagged harnesses pass Qwen-Image's 22 GiB memory hint to
`mx.set_memory_limit`, which puts the threshold at 22 GiB; that is a reading of source and
configuration, not of the runs. Nothing recorded the pool's size. With active memory above 22 GiB,
every cache miss releases cached buffers, so the pool need not have neared its limit.

What happens past the line is, here, the project's account rather than a measurement. The 0.10.0
changelog says the recipe "completes only under low host memory pressure". For a run that kept the
text encoders loaded, a comment in the
[v0.11.0 harness](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/scripts/bench_speedup.py)
records "compressed-memory thrash (a step went from ~20 s to ~230 s on 2026-09-06)". The 0.10.1
changelog logs a single vanilla unit in the same regime at 920 s, 8% slower than Table 1's median,
from a run log, not a committed artifact. If wall clock above the line depends on what else the
host is doing, host drift becomes a candidate for the 1.10×.

Drift becomes a difference between conditions when the run order lines it up with them. The
v0.10.0 harness orders its chunks condition by condition (`_pending_chunks`), so every vanilla
repetition ran before any no-gate one. 0.10.1 switched to "rep-outer (vanilla, no-gate, gated,
then the next rep) so slow host drift lands on every condition alike." The 0.10.0 report has no
timestamps and does not say whether it was one invocation, so drift as the source of the 1.10×
stays an inference. Isolating each process resets MLX's state, not the host's. And rep-outer order
is still fixed. Randomized order, as in
[Google Benchmark's random interleaving](https://google.github.io/benchmark/random_interleaving.html),
is the stronger standard, and
[Duplyakin et al. (USENIX ATC 2023)](https://www.usenix.org/conference/atc23/presentation/duplyakin)
show that running all trials of one setting before the next can bias results.

[When an MLX Memory Cap Is Not a Safety Boundary](https://github.com/IonDen/mlx-train-perf/blob/main/docs/papers/when-an-mlx-memory-cap-is-not-a-safety-boundary.md)
covers the safety side of running past the line on a 32 GB Mac.

## 5. The 0.11.0 re-run

The 0.11.0 benchmark ran on 2026-09-06 with changes from 0.10.1 and 0.11.0:

- mflux's `MemorySaver` freed the text encoders once the prompt was encoded. The 0.11.0 changelog
  credits it with moving what it calls the loop peak from 26.2 to 24.1 (the changelog writes GB;
  the harness measures GiB), under the recommended working set. That 24.1 GiB is the post-load
  active peak, which the report stores per condition as `vanilla_loop_peak_memory_gb`,
  `nogate_loop_peak_memory_gb` and `wrapper_loop_peak_memory_gb` (the gated condition), 24.08 to
  24.13 GiB; "under" refers to that measure.
  `MemorySaver` changes the workload, not only the harness: it also runs garbage collection and
  `mx.clear_cache()` when the denoising loop ends
  ([mflux source](https://github.com/mflux-community/mflux/blob/v.0.18.0/src/mflux/callbacks/instances/memory_saver.py#L60-L74)).
- The peak counter was reset after model load and read after the whole `generate_image` call.
  Despite its JSON name, the peak therefore spans prompt encoding, denoising and VAE decode.
  Since mflux materializes weights on first use, the load peak reads near zero and this peak
  compares with the 0.10.0 whole-run peak.
- Each benchmark process bounded MLX's cache pool at 1 GiB and armed a watchdog that aborts it if
  active plus cached memory exceeds physical memory minus 4 GiB. All nine retained chunks
  completed. The report cannot show whether earlier attempts were aborted: the harness files an
  abort separately and reruns the chunk on a later invocation.
- Chunks ran rep-outer, each timestamped.

| Condition | Rep 1 | Rep 2 | Rep 3 | Median | Active peak after load |
|---|---|---|---|---|---|
| vanilla | 610.9 s | 642.6 s | 715.9 s | 642.6 s | 24.08 GiB |
| no gate | 615.6 s | 611.5 s | 692.5 s | 615.6 s | 24.08 GiB |
| gated (0.30) | 229.3 s | 240.7 s | 239.4 s | 239.4 s | 24.13 GiB |

*Table 2. The 0.11.0 three-way benchmark, from the committed report. Peaks span the whole
generation after load, not the loop alone. The cache pool read 1.06 GiB in every chunk, after the
VAE decode. The gated condition again skipped 33 of 48 active steps, in the same pattern as in
0.10.0. Reproducer at the v0.11.0 tag:
`scripts/bench_speedup.py --variant qwen --three-way --reps 3` (variant page), with the same
caveat as Table 1.*

The report puts the active peaks 0.83 to 0.88 GiB below the line, in a phase it does not record.
Its cache reading does not describe the loop. The harness reads `mx.get_cache_memory()` only after
the VAE decode, and `MemorySaver` clears the cache when the denoising loop ends, so the 1.06 GiB
is cache refilled during decode. By the source, not by measurement, the 1 GiB limit held the pool
during the loop; MLX checks the limit before recycling a freed buffer, so the pool can overshoot it
slightly. A full allowance on top of the active peak (the 1 GiB limit, or the 1.06 GiB read after
decode) gives a loose upper bound of roughly 25.1 to 25.2 GiB, about 0.1 to 0.2 GiB over the
line. The reports record neither the loop's footprint nor the cache at the peak.

Since 0.12.0 the repository's comparison harness probes its two tight recipes, FLUX.2 Klein base
9B and Qwen-Image. A size passes only if the active peak plus the whole cache limit stays under the
working set (`probe_passes`); a failed full size falls back to a smaller one, and two failures stop
the run. 24.08 GiB plus 1 GiB would fail.

The ratios are now 2.68× for vanilla over gated, 2.57× for no-gate over gated, and 1.04× for
vanilla over no-gate. The README's table says the 1.04× "is inside the rep spread", and the
variant page calls it "noise, not a mechanism". Three repetitions support only a narrower
reading: the 1.04× is not distinguishable from no difference. Paired by repetition, vanilla over
the adjacent no-gate chunk gives 0.99, 1.05 and 1.03. The vanilla repetitions span 1.17×, so this run was about 43
times noisier in seconds (105 s against 2.4 s in 0.10.0), and that noise is part of what absorbed
the 1.04×.

The timestamps (08:50 to 10:38 UTC) show the drift that rep-outer order was meant to spread. Rep
1's chunks ran about 2 seconds apart; in rep 2 about 5 minutes passed before each later chunk, in
rep 3 about 10. The harness has no pause of its own; section 7 returns to those 30 minutes.
Vanilla slowed with each repetition, and the third no-gate repetition, after a 10-minute gap, ran
about 80 seconds slower than the first two. The drift was not monotone: the second no-gate
repetition (611.5 s) beat the vanilla chunk before it (642.6 s), and the third gated one did not
slow. Under rep-outer order the slowdown widened the spread within conditions;
under grouped order it would have become a gap between them.

## 6. The headline moved in one direction

With the skips unchanged, the timings still did not move together:

| Condition | 0.10.0 median | 0.11.0 median | Change |
|---|---|---|---|
| vanilla | 850.3 s | 642.6 s | −24% |
| no gate | 770.7 s | 615.6 s | −20% |
| gated (0.30) | 282.0 s | 239.4 s | −15% |

*Table 3. Median change between the two committed reports.*

The ratio went from 3.02× to 2.68×, so the published figure was about 12% above the later one
(12.4% from the unrounded medians), a difference of medians. Paired by repetition number,
vanilla over gated gives 3.05, 2.87 and 3.01 in 0.10.0 and 2.66, 2.67 and 2.99 in 0.11.0.
The two ranges overlap, and three repetitions each cannot separate 3.02× from 2.68× with
confidence.

Vanilla and no-gate run the block stack on all 50 steps; gated runs it on 17 (15 computed active
steps plus the two forced end steps). The asymmetry in Table 3 is consistent with a cost that the
over-the-line regime added and that grows with run length or full passes: the slower condition
pays more of it, and the ratio looks larger. Two details fit this better than the wrapper's own
code changes. Vanilla runs no wrapper code, yet it fell the most. And no-gate fell more than
gated, though both run the wrapper's forward on all 50 steps and only gated calls the gate, whose
code 0.10.1 changed (section 7). Neither isolates the regime.

Two safeguards did not catch the gap. The changelog's bracket, 2.73× to 3.02×, sits wholly above
the later 2.68×: both baselines ran in the same regime, so it could not catch a bias common to
both. The
[benchmark protocol](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/docs/m3-plus-tradeoff.md#benchmark-protocol-for-community-numbers)
pairs the median with the minimum, the reading least affected by background load. Computed from
the per-repetition times, the minimum moves the same way (3.04× then 2.66×, about 14%): it filters
spikes inside a run, not a shift of the whole run.

The reports store the vanilla-over-no-gate ratio, a plain ratio of medians, under the name
`compile_avoidance_ratio`. The name fits variants where the wrapper sidesteps an mflux compile,
such as FLUX.2 Klein (1.01× to 1.02× on the 50-step base recipes, per the
[README](https://github.com/IonDen/mlx-teacache/blob/v0.14.0/README.md)). Qwen-Image has no
compiled path, so its 1.103 and 1.044 hold any wrapper overhead plus whatever differed between
the runs of the two conditions.

## 7. What the evidence does and does not establish

Besides the harness changes in section 5, the library code changed. In 0.10.1 the cached residual
is materialized and released instead of left lazy, and the gate's two reductions are cast to
float32 and read back in one host sync. The changelog bounds what that does to skipping (Qwen's
predicted distance moves "by at most 0.16 percent per step", and a replay "still gives 32 to 33
skips"), but a replay bounds skips, not timing. Neither run is tied to a tagged build, so other
code differences cannot be ruled out.

Power went unrecorded: neither report stores a power state. The 0.11.0 gaps in section 5 were 5,
5, 10 and 10 minutes, each after a vanilla or no-gate chunk, never after a gated one. That pattern
fits pauses to let the battery recharge on a low-wattage adapter, which is the author's account of
the session. The repository's
[benchmark protocol](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/docs/m3-plus-tradeoff.md#benchmark-protocol-for-community-numbers)
says timings taken on a draining battery are not comparable to timings on a fully powered machine.
No single change can be credited with the movement.

| Claim | Status | Source |
|---|---|---|
| 0.10.0 active peak above the line (26.16–26.24 GiB against 24.96 GiB) | measured; cache unrecorded | [0.10.0 report](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/v0.10.0_bench_qwen_image.json) |
| 0.11.0 post-load active peak under it (24.08–24.13 GiB); with a cache allowance of 1.00 to 1.06 GiB, a loose upper bound of roughly 25.1–25.2 GiB | peak measured, phase unrecorded; bound computed, not observed | [0.11.0 report](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/v0.11.0_bench_qwen_image.json) |
| The later median is 2.68×; 3.02× is about 12% higher | measured, three repetitions each; per-repetition ratios overlap (2.87–3.05 and 2.66–2.99) | both reports |
| The cause of the 1.10× | unknown; a different computation made less likely, not ruled out (section 2) | [changelog](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/CHANGELOG.md); [variant page](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/docs/variants/qwen-image.md) |
| Running above the line inflated the ratio | consistent, not isolated | sections 4 to 6 |
| Grouped order turned host drift into the 1.10× | order verified in the tagged harness; drift inferred | [v0.10.0 harness](https://github.com/IonDen/mlx-teacache/blob/v0.10.0/scripts/bench_speedup.py) |
| June 1.18–1.20×, above the line | source-reported, single repetitions, no artifact; regime inferred | [v0.9.0 sweep script](https://github.com/IonDen/mlx-teacache/blob/v0.9.0/scripts/sweep_threshold_qwen.py); [mixed-precision paper](https://github.com/IonDen/mlx-teacache/blob/main/docs/papers/qwen-image-mixed-precision-on-a-32-gb-mac.md) |

## 8. Rules for a benchmark near the ceiling

The project's community benchmark protocol (at v0.11.0) asks for process isolation, a discarded
warmup, repetitions, skip counts beside timings, one recipe and mains power. The 0.10.0 run met the
parts `bench_speedup.py` implements, and its median ratio still came out about 12% above the later
run's. Running above the line is consistent with that gap, not isolated as its cause: none of those
safeguards varied the memory regime, and too much else changed between the runs. When two
conditions differ in work and the run sits near the device ceiling, the measured ratio can partly
measure the ceiling. This memo has one such case, and it tests only the first of the rules below.

1. Record the device's line beside both memory measures, and flag any ratio from a run that
   crossed it. Store `mx.device_info()["max_recommended_working_set_size"]` next to the active
   peak and the cache reading. `bench_speedup.py` does not enforce this; the comparison harness
   checks its two tight recipes against it (section 5). The `bench_speedup.py` watchdog fires at
   physical memory minus 4 GiB, 28 GiB here, a safety line above the performance line, so a run
   between the two completes normally with suspect timings. By this rule the 0.11.0 run is
   flagged, its loose upper bound about 0.1 to 0.2 GiB over the line.
2. Reset the peak counter after model load. Not tested here: for Qwen-Image, whose weights load
   lazily, it changed nothing, and the rule rests on general reasoning.
3. Interleave the conditions, preferably in random order, and timestamp every chunk. Not tested
   here: no run compared orders.
4. Bound the cache pool explicitly. Not tested here: this study did not measure how far the
   default pool grows under its 30.4 GiB limit, and the 1 GiB bound arrived together with
   `MemorySaver`.
5. Follow the protocol's power rule (a charger that out-supplies the load, a battery near full at
   the start) and record the adapter and battery state with the timings. Not tested here.

---
*Denis Ineshin · 2026-11-10 · [ineshin.space](https://ineshin.space)*
