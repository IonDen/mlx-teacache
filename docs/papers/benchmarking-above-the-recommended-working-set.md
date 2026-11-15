# Benchmarking Above the Recommended Working Set

*A research memo from `mlx-teacache`: a process-isolated Qwen-Image benchmark ran every condition above the GPU's recommended working set and published a speedup that a later run measured lower; its safeguards never varied the memory regime, so they could not detect whether that regime biased every condition alike*

> 📄 [Read on the website](https://ineshin.space/papers/benchmarking-above-the-recommended-working-set/) — same paper, formatted for reading.

## Update, 2026-11-15: what the two reports can support

In short: the 0.10.0 release published a 3.02× Qwen-Image speedup from a run whose active memory
peaked above the GPU's recommended working set, and a later run measured 2.68×; this memo credits
the difference to no single cause. The argument and the tables stay, and this update says what the
two reports behind them record and what they leave out.

Tables 1 and 2 below copy two committed benchmark reports, rounded; one cell is marked as the
changelog's. Table 3 is arithmetic on the two. Neither report names the MLX version. The project's
lock file resolves MLX 0.31.2 on macOS at every commit involved, but that is the lock file's
statement, not the reports'. The 0.10.0 report has no date and no commit, and the code that
produced it was not yet committed when the run began. The 0.11.0 report names a commit made during
the run and flags uncommitted changes. The committed code is the same before and after that run,
but an edit that was never committed cannot be ruled out. Three figures that the changelog states
with no results file behind them are now told in words: an earlier sweep's ratio, one slow vanilla
run and a per-step bound. Two figures that do have committed records are told in words as well,
because this memo cites them only in passing: the mixed-precision build's peak, whose record names
no MLX version and had its date and build label added by hand, and a FLUX.2 Klein ratio. A quoted
code comment about one slowed step is gone, and so is an account of why the later run paused. The
subtitle was reworded and no longer carries the 12% figure, which stays as an example in section 6.
The thesis, the mechanisms and the rules did not change.
The [previous version](https://github.com/IonDen/mlx-teacache/blob/bc02ac3475665327dea91a35da999a794a581906/docs/papers/benchmarking-above-the-recommended-working-set.md)
keeps the earlier text.

Safeguards that never vary the memory regime cannot see a bias that every condition shares. This
memo works through one case. An earlier memo from this repository,
[Why the TeaCache Gate Did Not Engage on Short Distilled FLUX Schedules](https://github.com/IonDen/mlx-teacache/blob/main/docs/papers/why-teacache-does-not-engage-on-short-distilled-schedules.md),
described a speedup that was real but credited to the wrong cause. This one describes a speedup
that a later run measured lower. For its 0.10.0 release, `mlx-teacache` published two Qwen-Image
figures against plain mflux (Table 1): a 3.02× speedup, and an unexplained 1.10× lead for the
wrapper with its gate off and no steps skipped. Both came from a run whose MLX active memory peaked
above the GPU's recommended working set. The 0.11.0 re-run (Table 2) changed several things at
once. Its active peak after model load (with a load phase near zero, also the whole-run peak) sat
0.83 to 0.88 GiB below that line; adding a 1 GiB cache allowance gives an estimate, computed rather
than observed, of about 0.1 to 0.2 GiB above it. The re-run's median ratio was 2.68×, though its
per-repetition ratios overlap the earlier run's, and the 1.10× became 1.04×, which three
repetitions cannot tell apart from no difference.

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

Pre-release builds of the library recorded both reports on an Apple M1 Max with 32 GB of unified
memory, Darwin 25.6.0 (macOS 26; the reports store the kernel version), mflux 0.18.0 and a
uniform 4-bit Qwen-Image build computing in bf16. Neither report records the MLX version. The
project's `uv.lock`, at the commits around both runs and at the
[v0.10.0](https://github.com/IonDen/mlx-teacache/blob/v0.10.0/uv.lock) and
[v0.11.0](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/uv.lock) tags, resolves MLX 0.31.2 on
macOS (its 0.31.1 entry is the Linux resolution) and mflux 0.18.0. That is what the lock file says;
neither report says it.

The 0.10.0 report has no date and no commit field. The
[v0.10.0 README](https://github.com/IonDen/mlx-teacache/blob/v0.10.0/README.md?plain=1#L238) dates
the run: "Measured 2026-09-01". The report was committed on the pull-request branch as 15250be on
2026-09-01 and merged to `main` as fe06b48 on 2026-09-02, the squash merge that the v0.10.0 tag
points to. The benchmark's Qwen-Image support entered the same
[pull-request branch](https://github.com/IonDen/mlx-teacache/pull/33) as 18c8feb, 24 seconds
before 15250be. The report's own timings add up to about 95 minutes, and the run had to end before
the report was committed as 15250be at 19:28 (UTC+2). It therefore began before 18c8feb at 19:27,
so the code that ran was not yet committed when it started, and the history does not show which
code ran.

The 0.11.0 report records commit 1536f28 and `git_dirty: true`. The harness writes both fields
when it writes the report, after the last chunk
([`bench_speedup.py` at 1536f28](https://github.com/IonDen/mlx-teacache/blob/1536f28eadad942f50273cb610825b3269598e84/scripts/bench_speedup.py#L931),
[lines 447–459](https://github.com/IonDen/mlx-teacache/blob/1536f28eadad942f50273cb610825b3269598e84/scripts/bench_speedup.py#L447-L459)),
and 1536f28 was made about 97 seconds after the first chunk began. Both fields therefore
describe the tree at report time, and the report does not list the uncommitted files. The
committed `scripts/` and `src/` are byte-identical at 46612df, the last commit before the first
chunk, at 1536f28, and at 97f4385, the commit that carries the report; all three are public in the
[v0.11.0 pull request](https://github.com/IonDen/mlx-teacache/pull/35). What stays open is an edit
to the code that sat in the working tree during the run and was never committed: nothing rules it
out. The report was merged to `main` as 3b86c8b on 2026-09-06.

Each report also carries a package version string, `0.9.4.dev8+g90fca33c9` and
`0.10.1.dev13+g05c05a1d9.d20260905`. That string is frozen when the package is installed, as
the harness's own docstring notes
([`bench_speedup.py` at 1536f28](https://github.com/IonDen/mlx-teacache/blob/1536f28eadad942f50273cb610825b3269598e84/scripts/bench_speedup.py#L448-L449)),
so neither describes the code at run time. The reports reached the tagged history only through the
squash merges; 15250be, 1536f28 and the two commits the version strings name are not in it. The
tagged scripts are therefore the released harness, not necessarily the code that produced these
numbers, and the MLX source cited below is a pinned reference, not the build that ran.

Everything here concerns the original Qwen/Qwen-Image checkpoint on mflux 0.18.0. Neither report
names the checkpoint or the gated threshold: the harness builds the model from mflux's `qwen-image`
configuration
([`bench_speedup.py` at 1536f28](https://github.com/IonDen/mlx-teacache/blob/1536f28eadad942f50273cb610825b3269598e84/scripts/bench_speedup.py#L226)),
and 0.30 is the variant's default in the tagged code. The gated skip pattern is identical in the
two reports. Since release 0.15.0, mlx-teacache requires mflux 0.24 or newer. On those mflux
releases, as the
[v0.16.0 variant page](https://github.com/IonDen/mlx-teacache/blob/v0.16.0/docs/variants/qwen-image.md)
explains, the `qwen-image` alias loads Qwen-Image-2512, a separate variant with a default threshold
of 0.25, and mflux's 4-bit Qwen build keeps the image-stream modulation layers at 8 bits, so a
re-run today is not this recipe. Nothing was rerun for this write-up.

## 1. The 0.10.0 benchmark and its safeguards

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
each timing started from a cold MLX allocator. The tagged harness has no warmup: each process loads
the model and times one `generate_image` call, its first. Three conditions ran: plain mflux
("vanilla"), the wrapper at the original Qwen-Image variant's default threshold of 0.30 ("gated"),
and the wrapper at threshold 0 ("no gate"). The no-gate condition separates the wrapper's own
speed from the speed gained by skipping. Every run used the same prompt ("a red apple on a wooden
table"), seed 42, 50 steps, guidance 4.0 and 768×768 output.

| Condition | Repetition seconds | Median | Peak MLX active memory | Steps skipped (of 48 active) |
|---|---|---|---|---|
| vanilla | 851.5 / 850.3 / 849.0 | 850.3 s | 26.16 GiB | — |
| no gate | 770.7 / 769.5 / 780.8 | 770.7 s | 26.16 GiB | 0 (changelog) |
| gated (0.30) | 279.2 / 296.4 / 282.0 | 282.0 s | 26.24 GiB | 33 in every repetition |

*Table 1. A transcription of the committed
[0.10.0 report](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/v0.10.0_bench_qwen_image.json)
(copy at the v0.14.1 tag). The report records no MLX version, no date and no commit, and the
history does not show which code produced it (see the introduction). The 0.30 in the row label is
the variant default in the tagged code, not a report field. The report's memory fields say `GB`,
but the [v0.10.0 harness](https://github.com/IonDen/mlx-teacache/blob/v0.10.0/scripts/bench_speedup.py)
divides by 1024³, so the figures are GiB: MLX's own `mx.get_peak_memory()`, active buffers only
(no buffer cache, process or system memory), over the whole run including model load. Reproducer
at the tag: `scripts/bench_speedup.py --variant qwen --three-way --reps 3` (the variant page's
command).*

The report derives 3.02× for vanilla over gated, 2.73× for no-gate over gated and 1.10× for
vanilla over no-gate. Cold processes, repeated runs, skip counts beside every timing and a second
baseline are what the project's own standards ask of a benchmark, and this run had all four.

## 2. A gate-off lead without a known cause

On the obvious suspect, a different computation, the 0.10.0 changelog has one line: "the no-gate
output is pixel-identical to vanilla's". It does not name the images. The tagged harness saves one
image per condition, from the first repetition only, into an untracked directory: one pair at
most, and none a reader can check.

The usual explanations look less likely here. The README and the variant page state that mflux
does not `mx.compile` Qwen-Image's prediction step, so unlike on FLUX.2 Klein and Z-Image the
wrapper had no compiled path to avoid. And at threshold 0 the wrapper runs the same transformer
arithmetic, through its own walk of mflux's forward.

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

The changelog adds that the 0.9.0 sweep had shown the same zero-skip wrapper ahead of vanilla, by
a wider margin. That sweep,
[`sweep_threshold_qwen.py`](https://github.com/IonDen/mlx-teacache/blob/v0.9.0/scripts/sweep_threshold_qwen.py),
used a different harness: one vanilla unit first, then one wrapped unit per threshold, so each
speedup is a single repetition. It ran the 0.9.x gate on the mixed-precision build described in
[Qwen-Image Mixed Precision on a 32 GB Mac](https://github.com/IonDen/mlx-teacache/blob/main/docs/papers/qwen-image-mixed-precision-on-a-32-gb-mac.md)
and left no committed artifact and no peak reading. That memo's section 5 lists the committed
memory records for the mixed and the stock 4-bit builds. The one committed record for the mixed
build at 768×768, the repository's
[comparison report](https://github.com/IonDen/mlx-teacache/blob/v0.17.0/_artifacts/comparison_report.json),
puts its peak several GiB above the line. It comes from a different run with a different prompt;
that row's date and build label were added by hand after its figures were committed, and it names
no MLX version. The sweep therefore probably ran above the line too. It changed the session, the
gate and the build, and kept the grouped order. It points the same way as the 1.10× and does not
test its cause.

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
- `mx.set_cache_limit` bounds the pool of freed buffers kept for reuse. Its default is the memory
  limit's, the smaller of 1.5 times the recommended set and 0.95 times physical memory, 30.4 GiB
  here. A lower line also applies: when an allocation misses the cache and active plus cached
  memory would reach a reclamation threshold, MLX first releases cached buffers. The
  [Metal allocator](https://github.com/ml-explore/mlx/blob/ce916dbbcaa88e433b6fd1e60a17f766d49c27fe/mlx/backend/metal/allocator.cpp)
  puts that threshold at the smaller of the memory limit and 0.95 times the recommended set,
  about 23.7 GiB here by default.

The v0.10.0 scripts never call `mx.set_cache_limit`, and the variant page dates the cache bound to
0.10.1, so the 0.10.0 processes most likely ran with the default; the history does not show which
code ran. Both tagged harnesses pass Qwen-Image's 22 GiB memory hint to `mx.set_memory_limit`,
which puts the threshold at 22 GiB; that is a reading of source and configuration, not of the runs.
Nothing recorded the pool's size. With active memory above 22 GiB, every cache miss releases cached
buffers, so the pool need not have neared its limit.

What happens past the line is, here, the project's account rather than a measurement. The 0.10.0
changelog says the recipe "completes only under low host memory pressure". The
[0.10.1 changelog entry](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/CHANGELOG.md?plain=1#L48)
describes a later sweep on the same recipe, under the guarded harness, with figures from that run's
log rather than a committed artifact. Its vanilla reference, again above the line on active memory,
ran slower than Table 1's vanilla median, and then "the host ran out of memory during the first
gated unit and the run was stopped from outside the process". If wall clock above the line depends
on what else the host is doing, host drift becomes a candidate for the 1.10×.

Drift becomes a difference between conditions when the run order lines it up with them. The
v0.10.0 harness orders its chunks condition by condition (`_pending_chunks`); if that is the code
that ran, every vanilla repetition ran before any no-gate one. 0.10.1 switched to "rep-outer
(vanilla, no-gate, gated, then the next rep) so slow host drift lands on every condition alike".
The 0.10.0 report has no timestamps and does not say whether it was one invocation, so drift as the
source of the 1.10× stays an inference. A fresh process resets MLX's allocator, while the host's
memory state carries over from one process to the next. Rep-outer order is also still fixed.
Randomized order, as in
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
  24.13 GiB; "under" refers to that measure. No run isolates that credit: the cache bound, the
  peak reset, the chunk order, the watchdog and the build changed in the same interval.
  `MemorySaver` changes the workload as well as the harness: with `keep_transformer=True`, which the
  harness sets, it also runs garbage collection and `mx.clear_cache()` when the denoising loop ends
  ([mflux source](https://github.com/mflux-community/mflux/blob/v.0.18.0/src/mflux/callbacks/instances/memory_saver.py#L60-L74)).
- The peak counter was reset after model load and read after the whole `generate_image` call.
  Despite its JSON name, the peak therefore spans prompt encoding, denoising and VAE decode.
  Since mflux materializes weights on first use, as the
  [variant page](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/docs/variants/qwen-image.md?plain=1#L85)
  notes, the load peak reads near zero and this peak compares with the 0.10.0 whole-run peak.
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

*Table 2. A transcription of the committed
[0.11.0 report](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/v0.11.0_bench_qwen_image.json)
(copy at the v0.14.1 tag). The report records no MLX version, and its commit field and
`git_dirty: true` describe the tree when the report was written: the committed code is the same
around the run, but an uncommitted edit cannot be ruled out (see the introduction). The 0.30 in
the row label is the variant default in the tagged code, not a report field. Peaks span the
whole generation after load, not the loop alone; the load phase reads near zero, so they are also
whole-run peaks. The cache pool read 1.06 GiB in every chunk, after the VAE decode. The gated
condition again skipped 33 of 48 active steps, in the same pattern as in 0.10.0. Reproducer at
the v0.11.0 tag:
`scripts/bench_speedup.py --variant qwen --three-way --reps 3` (variant page).*

The report puts the active peaks 0.83 to 0.88 GiB below the line, in a phase it does not record.
Its cache reading does not describe the loop. The harness reads `mx.get_cache_memory()` only after
the VAE decode, and `MemorySaver` clears the cache when the denoising loop ends, so the 1.06 GiB
is most likely cache refilled during decode; the reports do not show this. By the source, not by
measurement, the 1 GiB limit held the pool during the loop; MLX checks the limit before recycling
a freed buffer, so the pool can overshoot it slightly. A full allowance on top of the active peak
(the 1 GiB limit, or the 1.06 GiB read after decode) gives an estimate, computed rather than
observed, of roughly 25.1 to 25.2 GiB, about 0.1 to 0.2 GiB over the line. The reports record
neither the loop's footprint nor the cache at the peak.

Since 0.12.0 the repository's comparison harness probes its two tight recipes, FLUX.2 Klein base
9B and Qwen-Image. A size passes only if the active peak plus the whole cache limit stays under the
working set (`probe_passes`); a failed full size falls back to a smaller one, and two failures stop
the run. 24.08 GiB plus 1 GiB would fail.

The ratios are now 2.68× for vanilla over gated, 2.57× for no-gate over gated, and 1.04× for
vanilla over no-gate. The README's table says the 1.04× "is inside the rep spread", and the
variant page calls it "noise, not a mechanism". Three repetitions support only a narrower
reading: the 1.04× is not distinguishable from no difference. Paired by repetition, vanilla over
the adjacent no-gate chunk gives 0.99, 1.05 and 1.03. The vanilla repetitions span 1.17×, so this
run was about 43 times noisier in seconds (105 s against 2.4 s in 0.10.0), and that noise is part
of what absorbed the 1.04×.

The timestamps (08:50 to 10:38 UTC) show the gaps between chunks, and the timings show a slowdown
across repetitions, the kind of slow drift that rep-outer order was meant to spread. Rep 1's
chunks ran about 2 seconds apart; in rep 2 about 5 minutes passed before each later chunk, in
rep 3 about 10. The harness has no pause of its own; section 7 returns to those 30 minutes.
Vanilla slowed with each repetition, and the third no-gate repetition, after a 10-minute gap, ran
about 80 seconds slower than the first two. The drift was not monotone: the second no-gate
repetition (611.5 s) beat the vanilla chunk before it (642.6 s), and the third gated one did not
slow. Under rep-outer order the slowdown widened the spread within conditions; under grouped order
it would have become a gap between them.

## 6. The headline moved in one direction

With the skips unchanged, the timings still did not move together:

| Condition | 0.10.0 median | 0.11.0 median | Change |
|---|---|---|---|
| vanilla | 850.3 s | 642.6 s | −24% |
| no gate | 770.7 s | 615.6 s | −20% |
| gated (0.30) | 282.0 s | 239.4 s | −15% |

*Table 3. Median change between the two committed reports, computed from Tables 1 and 2. It
inherits the gaps of both reports: neither names the MLX version; the history does not show the
code behind the 0.10.0 report, and the 0.11.0 report was written from a tree with uncommitted
changes.*

The ratio went from 3.02× to 2.68×, so the published figure was about 12% above the later one
(12.4% from the unrounded medians), a difference of medians. Paired by repetition number,
vanilla over gated gives 3.05, 2.87 and 3.01 in 0.10.0 and 2.66, 2.67 and 2.99 in 0.11.0.
The two ranges overlap, and three repetitions each cannot separate 3.02× from 2.68× with
confidence.

Vanilla and no-gate run the block stack on all 50 steps; gated runs it on 17 (15 computed active
steps plus the two forced end steps). The asymmetry in Table 3 is consistent with a cost that the
over-the-line regime added and that grows with run length or full passes: the slower condition
pays more of it, and the ratio looks larger. Vanilla runs no wrapper code, yet it fell the most.
And no-gate fell more than gated, though both run the wrapper's forward on all 50 steps and only
gated calls the gate, whose code 0.10.1 changed (section 7). Both fit the regime better than the
wrapper's own code changes, and neither isolates it.

The changelog's bracket, 2.73× to 3.02×, sits wholly above the later 2.68×. Both baselines ran in
the same regime, so the bracket could not catch a bias common to both. The
[benchmark protocol](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/docs/m3-plus-tradeoff.md#benchmark-protocol-for-community-numbers)
pairs the median with the minimum, the reading least affected by background load. Computed from
the per-repetition times, the minimum moves the same way (3.04× then 2.66×, about 14%). It filters
spikes inside a run; a shift that moves the whole run passes through it.

The reports store the vanilla-over-no-gate ratio, a plain ratio of medians, under the name
`compile_avoidance_ratio`. The name fits variants where the wrapper sidesteps an mflux compile,
such as FLUX.2 Klein, where the
[README](https://github.com/IonDen/mlx-teacache/blob/v0.14.0/README.md) puts that ratio within
about two percent of 1 on the 50-step base recipes. Qwen-Image has no compiled path, so its 1.103
and 1.044 hold any wrapper overhead plus whatever differed between the runs of the two conditions.

## 7. What the evidence does and does not establish

Besides the harness changes in section 5, the library code changed. In 0.10.1 the cached residual
is materialized and released instead of left lazy, and the gate's two reductions are cast to
float32 and read back in one host sync. The changelog bounds what that does to skipping: it states
a per-step bound on how far the change moves Qwen's predicted distance, and it says the gate
"replayed over its committed calibration trace still gives 32 to 33 skips", which a
[replay test](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/tests/test_gate_replay.py)
asserts. A replay bounds skips, not timing. The history does not show which code produced the
0.10.0 report, so other code differences between the runs cannot be ruled out.

Power went unrecorded: neither report stores a power state. The 0.11.0 gaps in section 5 were 5,
5, 10 and 10 minutes, each after a vanilla or no-gate chunk, never after a gated one, and no record
says why the run waited. Power is one cause of the timing change that the reports cannot exclude:
the repository's
[benchmark protocol](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/docs/m3-plus-tradeoff.md#benchmark-protocol-for-community-numbers)
says timings taken on a draining battery are not comparable to timings on a fully powered machine.
No single change can be credited with the movement.

| Claim | Status | Source |
|---|---|---|
| 0.10.0 active peak above the line (26.16–26.24 GiB against 24.96 GiB) | recorded in the report, which names no MLX version, no date and no commit, and whose code the history does not show; cache unrecorded | [0.10.0 report](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/v0.10.0_bench_qwen_image.json) |
| 0.11.0 post-load active peak under it (24.08–24.13 GiB); with a cache allowance of 1.00 to 1.06 GiB, an estimate of roughly 25.1–25.2 GiB | peak recorded in the report, which names no MLX version and was written from a tree with unlisted uncommitted changes; phase unrecorded; estimate computed, not observed | [0.11.0 report](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/v0.11.0_bench_qwen_image.json) |
| The later median ratio, 2.68×, is below 3.02×; per-repetition ratios overlap (2.87–3.05 and 2.66–2.99) | recorded in the reports; neither names the MLX version, the history does not show the 0.10.0 code and the 0.11.0 report flags uncommitted changes; three repetitions each | both reports |
| The cause of the 1.10× | unknown; a different computation made less likely, not ruled out (section 2) | [changelog](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/CHANGELOG.md); [variant page](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/docs/variants/qwen-image.md) |
| Running above the line may have inflated the ratio | consistent, not isolated | sections 4 to 6 |
| Grouped order may have turned host drift into the 1.10× | order verified in the tagged harness; drift inferred | [v0.10.0 harness](https://github.com/IonDen/mlx-teacache/blob/v0.10.0/scripts/bench_speedup.py) |
| An earlier sweep also had the zero-skip wrapper ahead of vanilla, by a wider margin, probably above the line | changelog statement, single repetitions, no committed record; regime inferred | [changelog](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/CHANGELOG.md); [v0.9.0 sweep script](https://github.com/IonDen/mlx-teacache/blob/v0.9.0/scripts/sweep_threshold_qwen.py) |

## 8. Rules for a benchmark near the ceiling

The project's community benchmark protocol (at v0.11.0) asks for process isolation, a discarded
warmup, repetitions, skip counts beside timings, one recipe and mains power. Its
[first item](https://github.com/IonDen/mlx-teacache/blob/v0.11.0/docs/m3-plus-tradeoff.md?plain=1#L98-L100)
asks a report to name its environment, including the Python version and the installed `mlx`,
`mflux` and `mlx-teacache` versions; neither report recorded the Python or the MLX version. The
tagged harnesses have no discarded warmup: each process times its first generation. The 0.10.0
run met the parts `bench_speedup.py` implements, and its median ratio still came out higher than
the later run's (section 6). Running above the line is consistent with that gap, not isolated
as its cause: none of those safeguards varied the memory regime, and too much else changed between
the runs. When two conditions differ in work and the run sits near the device ceiling, the measured
ratio can partly measure the ceiling. This memo has one such case, and it tests only the first of
the rules below.

1. Record the device's line beside both memory measures, and flag any ratio from a run that
   crossed it. Store `mx.device_info()["max_recommended_working_set_size"]` next to the active
   peak and the cache reading. `bench_speedup.py` does not enforce this; the comparison harness
   checks its two tight recipes against it (section 5). The `bench_speedup.py` watchdog fires at
   physical memory minus 4 GiB, 28 GiB here, higher than the 24.96 GiB performance line, so a run
   between the two completes normally with suspect timings. By this rule the 0.11.0 run is
   flagged: its estimate, computed rather than observed, is about 0.1 to 0.2 GiB over the line.
2. Reset the peak counter after model load. Not tested here. For Qwen-Image, whose weights load
   lazily, the reset probably changes little; the rule rests on general reasoning.
3. Interleave the conditions, preferably in random order, and timestamp every chunk. Not tested
   here: no run compared orders.
4. Bound the cache pool explicitly. Not tested here: this study did not measure how far the
   default pool grows under its 30.4 GiB limit, and the 1 GiB bound arrived together with
   `MemorySaver`.
5. Follow the protocol's power rule (a charger that out-supplies the load, a battery near full at
   the start) and record the adapter and battery state with the timings. Not tested here.

---
*Denis Ineshin · 2026-11-10, updated 2026-11-15 · [ineshin.space](https://ineshin.space)*
