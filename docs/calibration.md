# Coefficient calibration

mlx-teacache's gate uses a degree-4 polynomial that maps the relative-L1
distance of the modulated block-0 input (`rel_l1(mod_in_t, mod_in_{t-1})`)
to the predicted relative-L1 distance of the transformer output. Coefficients
live per variant in `src/mlx_teacache/variants/<id>/config.py` (`COEFFICIENTS`),
with the matching `Provenance` in that variant's `integration.py`. The top-level
`coefficients.py` is a thin re-export shim, extracted in v0.6.0; editing it
changes nothing.

**Anchoring convention.** `t-1` is the immediately previous gated step,
whether that step was computed or skipped — the comparison anchor advances
on every gated step. This matches how the calibration script measures its
training pairs: `rel_l1(mod_in_t, mod_in_{t-1})` between consecutive steps,
never against an older, non-adjacent step.

## Calibrated range

The FLUX.2 Klein base, Z-Image, Qwen-Image and Qwen-Image-2512 fits are
origin-constrained: the
polynomial is forced through zero. Below the smallest delta the calibration
produced, that constraint decides the prediction, not the data. The base-4b
polynomial gives 0.027 at a delta of 0.0035, while the smallest output change
its calibration measured anywhere was 0.107. On an 864×1152, 50-step
klein-base-4b run the gate skipped three steps with deltas that small at every
threshold down to 0.08.

Since 0.12.0 each of these variants records the range its fit was made on as
`CALIBRATED_RANGE` in its `config.py`, copied from `x_min` / `x_max` of the
calibration JSON (signal B for Z-Image, signal A for both Qwen-Image checkpoints; klein-base-9b
takes base-4b's along with its coefficients). The gate clamps the measured
delta into that range before evaluating the polynomial, so a delta below it is
priced like the smallest calibrated one. A delta above it is a larger change
than the fit ever saw, and the gate always computes that step: on Klein base the
fit's value at the top of its range (0.164) is below the 0.17 default, so
pricing it there would still allow a skip.
The step's recorded `rel_l1` is still the measured delta, so the stats show
when a run went outside the range. FLUX.1 dev and schnell (upstream tuples with
no recorded range), Krea (a free-intercept fit, which predicts more change
below its data rather than less), the distilled Klein variants and
user-supplied coefficients are evaluated unclamped.

Every clamped polynomial stays positive over its range, and its lowest value is
at the lower end: 0.144 for Klein base, 0.082 for Z-Image, 0.067 for Qwen-Image
and 0.038 for Qwen-Image-2512.
Each skip therefore adds at least that much to the accumulator, which caps the
longest possible run of skips at the shipped thresholds at 1 for Klein base and
Z-Image, 4 for Qwen-Image and 6 for Qwen-Image-2512 (0.25 / 0.038 allows six).
The benches show the same streaks for the first three; the 2512 sweep observed
4 at its 0.25 default, under that bound.

At the committed bench recipes this changed one decision: klein-base-4b at
50 steps / g=4.0 now computes step 6 (8 skips instead of 9, 1.20× instead of
1.22×). klein-base-9b, Z-Image and Qwen-Image skip the same steps as before. If
you re-calibrate one of these variants, update `CALIBRATED_RANGE` together with
the coefficients; `tests/test_calibration_artifacts.py` fails when they drift
apart.

Qwen-Image-2512 joined in 0.14.0 with its own fit and range (input changes
from 0.0809 to 0.4706), so the gate clamps it the same way from its first
release.

## Runaway guard

The gate accumulates the polynomial's predicted change across consecutive
skips and resets it to zero on every real compute. The
origin-constrained in-repo fits are positive for small deltas but cross zero
at large ones (base-4b at x≈0.24, z-image at x≈0.29, qwen at x≈0.78,
qwen-image-2512 at x≈0.68), beyond the range they were fit on; there the clamp turns a large, real change into
a predicted change of zero, so the accumulator stops advancing. Without a
guard, that would let the wrapper reuse the same cached residual for an
unbounded number of steps. The calibrated range (above) now keeps the deltas of
those variants inside the fitted range, where their polynomials are positive,
so for them the guard is a backstop; it still matters for user-supplied
coefficients. `MAX_CONSECUTIVE_SKIPS` in `src/mlx_teacache/_kernel/gate.py`
forces a recompute after 8 consecutive skips regardless of the accumulated
total. This is an intentional divergence from upstream ali-vilab TeaCache,
which has no such cap — the same kind of deliberate departure as the gate's
`max(0.0, ...)` clamp on the polynomial output, which keeps the accumulator
monotonic instead of matching upstream's raw polynomial exactly.

### Observed max consecutive-skip streaks

Measured at each variant's default threshold on the committed bench recipes
(M1 Max 32 GB, mflux 0.18.0, three cold reps per condition; flux1-dev and
z-image-base on 2026-08-15, the two Klein base rows on 2026-09-28 with the
0.12.0 gate, qwen-image on 2026-09-06, flux1-krea-dev on 2026-09-05). Every row but qwen-image renders at
512×512; qwen-image uses its pinned 768×768. The bench reports carry a per-rep
`skip_patterns` string (`S` = skipped, `C` = computed) and
`max_consecutive_skips`; the streak below is the maximum across reps, and in
every case each rep produced the same pattern.

| Variant | Default threshold | Skips (active steps) | Max observed streak | Source |
|---|---|---|---|---|
| `flux1-dev` | 0.20 | 6 / 23 | 1 | `_artifacts/v0.10.0_bench_flux1_dev.json` (25 steps, g=3.5) |
| `flux2-klein-base-4b` | 0.17 | 8 / 48 | 1 | `_artifacts/v0.12.0_bench_klein_base_4b.json` (50 steps, g=4.0) |
| `flux2-klein-base-9b` | 0.17 | 13 / 48 | 1 | `_artifacts/v0.12.0_bench_klein_base_9b.json` (50 steps, g=4.0) |
| `z-image-base` | 0.12 | 15 / 48 | 1 | `_artifacts/v0.10.0_bench_z_image.json` (50 steps, g=4.0, q8) |
| `qwen-image` | 0.30 | 33 / 48 | 4 | `_artifacts/v0.11.0_bench_qwen_image.json` (50 steps, g=4.0, 768×768) |
| `flux1-krea-dev` | 0.30 | 10 / 26 | 1 | `_artifacts/v0.11.0_bench_krea_dev.json` (28 steps, g=4.5) |

Every row but qwen-image stays at 1: the gate alternates compute / skip and
never reuses a residual twice in a row. `qwen-image` runs longer, reaching 4,
because at its 0.30 default it skips roughly two-thirds of its active steps —
much the largest share in the table. Even there the `MAX_CONSECUTIVE_SKIPS = 8`
cap sits at twice the longest observed streak, so it engages at no shipped
default and mainly matters for degenerate settings (an all-zero polynomial, or
a threshold well above the sweep range).

Qwen's streak grew with the v0.10.0 anchoring change: 0.9.x documented 24
skips at this threshold and recipe, and the gate replayed over the committed
calibration trace under consecutive-delta anchoring gives 33 skips with a
streak of 4, which is exactly what the bench then measured. It is the variant
the anchoring fix moves most, because its accumulator sits nearest the
threshold. A threshold sweep under the current gate (stock q4, same recipe)
gave 24 / 26 / 30 / 33 skips at 0.15 / 0.20 / 0.25 / 0.30, SSIM 0.980 / 0.980 /
0.976 / 0.967, longest streaks 2 / 2 / 3 / 4, and 0.30 stays the default (see
the variant page). The record is
`_artifacts/sweeps/qwen-image_plain-q4_2026-09-06.json`.
What the record leaves out is [listed below](#threshold-sweep-records).

`qwen-image-2512` has no bench yet, so it is not in the table. Its threshold
sweep (red-apple prompt, seed 42, 768×768, 50 steps, guidance 4.0, 4-bit)
skipped 29 of 48 active steps at the 0.25 default with a longest run of 4, and
a weight-free replay of the gate over its calibration traces gives the same 29
skips and run of 4 (`tests/test_gate_replay.py` pins this). The longest run in
that sweep was 6, at 0.40, still under the cap. The record is
`_artifacts/sweeps/qwen-image-2512_plain-q4_2026-10-20.json`. What it leaves
out is [listed below](#threshold-sweep-records).

## Threshold-sweep records

Four threshold sweeps have a committed record, a JSON file with the skip count
and the SSIM against plain mflux for each threshold, and, in three of the four,
the longest streak (the Z-Image record has none). The
files are copies of what the runs wrote; no value was edited. Every similarity
score in them was computed from images that are not published, so a reader
cannot recompute a score from these files.

| Record | Model | Recipe | Date | MLX / mflux per `uv.lock` (macOS entry) | Script |
|---|---|---|---|---|---|
| `_artifacts/sweeps/qwen-image_plain-q4_2026-09-06.json` | Qwen-Image | stock 4-bit, 768×768, 50 steps, guidance 4.0, seed 42; thresholds 0.15 to 0.30 | 2026-09-06 | MLX 0.31.2, mflux 0.18.0 (`uv.lock` at 3b86c8b) | `scripts/sweep_threshold_qwen.py`, in the repository since June 2026; the version that frees the text encoders was committed during the run and reached main in 3b86c8b |
| `_artifacts/sweeps/qwen-image-2512_plain-q4_2026-10-20.json` | Qwen-Image-2512 | stock 4-bit, 768×768, 50 steps, guidance 4.0, seed 42; thresholds 0.15 to 0.40 | 2026-10-20 | MLX 0.32.2, mflux 0.23.0 (`uv.lock` at b883378) | `scripts/sweep_threshold_qwen.py`, as on main at b883378 (see below) |
| `_artifacts/sweeps/z-image-base_2026-05-31.json` | Z-Image base | 8-bit, 512×512, 50 steps, guidance 4.0, seed 42; thresholds 0.05 to 0.30 | 2026-05-31 | MLX 0.31.2, mflux 0.17.5 (`uv.lock` at c6a4109) | `scripts/sweep_threshold_z_image.py` at c6a4109 |
| `_artifacts/sweeps/flux1-krea-dev_2026-09-05.json` | FLUX.1 Krea [dev] | 4-bit, 512×512, 28 steps, guidance 4.5, seed 42; thresholds 0.30 to 0.80 | 2026-09-05 | MLX 0.31.2, mflux 0.18.0 (`uv.lock` at 3b86c8b) | `scripts/calibrate_flux1.py --model krea-dev --sweep`, committed on a working branch before the run; reached main in 3b86c8b |

What the records leave out, for all four: none holds a date, an MLX or mflux
version, the hardware, a commit or a flag saying whether the working tree had
uncommitted changes. The date in the table is the file time on the machine when the record was
made, and the file name carries it (git keeps no file times). The versions are the ones `uv.lock` pins at the
commit named, which may differ from what was installed. The hardware is not
recorded.

The speed columns are single timings. Each is one gated run against one plain
run, and that plain run was slower than the median of the three-rep bench for
the same model: 920 s against 643 s for Qwen-Image, 246 s against 227 s for
Z-Image, 129 s against 125 s for Krea. The Qwen-Image and Z-Image rows read
high because their speed is measured against the sweep's own slower plain run;
against the bench median, each of those rows sits within what its skipped steps
allow. One Z-Image row (0.30) shows a speedup below 1×. The Krea 0.60 and 0.80 rows are different: those single gated runs
were faster than their skip counts allow (at 0.80, 10 computed steps out of 28
at the plain 128.7 s give 45.96 s, and the gated run took 41.19 s). The record
does not say why. Read all of these as single timings, not findings. The 2512
page does not quote the timings in its record.

mflux's release notes changed the output of three of these models after the
runs. Version 0.22.0 loads the bias of the last normalization layer again for
FLUX.1 and Qwen-Image (slightly different images for FLUX.1, visibly different
ones for Qwen-Image) and runs Z-Image's transformer in bfloat16, and 0.24.0
changed Z-Image's VAE decode. So the Qwen-Image and Krea scores hold for
mflux 0.18.0 and the Z-Image scores for 0.17.5. The 2512 record was made on
0.23.0, after the 0.22.0 change.

For each record:

- Qwen-Image: the plain-mflux reference image comes from a first attempt that was stopped when the host ran out of memory (see the 0.10.1 notes in the CHANGELOG), and the gated runs followed hours later. The script has been in the repository since June 2026. The version that frees the text encoders after the prompt is encoded was committed during the run, and reached main in 3b86c8b. The reference and the 0.15 run finished before that change. The record's `memory_saver: true` flag says only that some run freed the encoders; the per-run files that would show which are not committed, so the 0.20 to 0.30 runs most likely did, but the record does not prove it.
- Qwen-Image-2512: the runs used a working commit that was not kept. The script on main (b883378) differs from it by three lines: how each run reads the variant name, and one line added in the step that wrote this file.
- Z-Image base: c6a4109, made shortly before the run, holds the script as it then stood; whether the working tree matched it is unknown. The sweep predates the gate change in 0.10.0.
- FLUX.1 Krea [dev]: the script was committed on a working branch before the run and reached main in 3b86c8b the next day. Whether the working tree matched that commit is not recorded. The Krea fit and its 0.30 default were committed about two minutes after the sweep's last run, and the sweep's 0.30 skip pattern equals the shipped bench's in all three reps, so the sweep most likely ran with that fit. The record names the model by the alias `krea-dev`.

Sweep figures that earlier versions of these pages printed without a record are
now described in words, without the number. Released CHANGELOG and ROADMAP lines
keep their figures and carry a dated note where no record exists.

## Built-in coefficient sources

| Variant | Source | Provenance |
|---|---|---|
| `flux1-dev`, `flux1-schnell` | Vendored from ali-vilab/TeaCache (`TeaCache4FLUX/teacache_flux.py`); Apache-2.0. | `COEFFICIENTS` in `variants/flux1_dev/config.py` (shared by `flux1_schnell` via cross-import). The FLUX dev/schnell architecture is shared, so both reuse the same set. |
| `flux1-krea-dev` | Derived in-repo on 2026-09-05 from 10 prompts × 28 steps × seed=42 on M1 Max 32GB, bf16, 512×512, guidance=4.5 (the model card's recipe). Free `numpy.polyfit(degree=4)` on 270 consecutive-step `(mod_in, body_out)` rel-L1 pairs; R² = 0.68 (the origin-constrained fit gave 0.66). FLUX.1-dev's vendored tuple was scored on the same pairs and rejected at R² = −496: Krea's per-step changes reach rel-L1 0.66 where dev's stay near 0.2, and dev's 499·x⁴ term explodes there. The fit's intercept (~0.40) makes the package fallback 0.20 skip nothing; the per-variant default 0.30 comes from the sweep on the variant page. | `COEFFICIENTS` in `variants/flux1_krea_dev/config.py`; `_PROVENANCE` in its `integration.py`. Calibration and sweep script: `scripts/calibrate_flux1.py --model krea-dev`. Full report with the scored tuple: `scripts/_calibration_flux1_krea_dev.json`. |
| `flux2-klein-4b` | Derived in-repo on 2026-05-15 from 10 prompts × 8 steps × seed=42 on M1 Max 32GB, bf16, 512×512, guidance=1.0. `numpy.polyfit(degree=4)` on 70 consecutive-step `(mod_in, body_out)` rel-L1 pairs; R² = 0.65. | `COEFFICIENTS` in `variants/flux2_klein_4b/config.py`; `_PROVENANCE` in its `integration.py`. Calibration script: `scripts/calibrate_flux2.py --variant klein-4b`. Full report: `scripts/_calibration_flux2_klein_4b.json` (summary only; unlike the klein-9b and base-4b reports it omits the raw `x_values`/`y_values` arrays, so reproducing this tuple offline means re-running the ~196 s calibration). |
| `flux2-klein-9b` | Derived in-repo on 2026-05-16 from 10 prompts × **8 steps** × seed=42 on M1 Max 32GB, bf16, 512×512, guidance=1.0. **Origin-constrained** least-squares fit (forces `poly(0) = 0`) on 70 consecutive-step `(mod_in, body_out)` rel-L1 pairs; R² = 0.4710. At the package default `rel_l1_thresh=0.20` these coefficients trigger **0 step-skips** on Klein 9B's 8-step schedule — empirical `y_min = 0.25` exceeds the threshold. Any wall-clock difference on Klein comes from `mx.compile`-path avoidance, and only on Macs where mflux compiles `_predict` (not base or Pro M1/M2), not from caching. | `COEFFICIENTS` in `variants/flux2_klein_9b/config.py`; `_PROVENANCE` in its `integration.py`. Calibration script: `scripts/calibrate_flux2.py --variant klein-9b --fit-mode origin`. Full report: `scripts/_calibration_flux2_klein_9b.json`. |
| `flux2-klein-base-4b` | Derived in-repo on 2026-05-17 from 10 prompts × **25 steps** × seed=42 on M1 Max 32GB, bf16, 512×512, guidance=1.0. **Origin-constrained** least-squares fit (forces `poly(0) = 0`); R² = 0.106. Low R² for the polynomial form — FLUX.2-family polynomials are noisier than FLUX.1-family. To compensate, this variant ships a per-variant default `rel_l1_thresh=0.17` via `Provenance.default_thresh` (set automatically). At that threshold the gate skips only a few of the 25 steps. The committed multi-run bench for this variant is at the 50-step CFG recipe (`_artifacts/v0.12.0_bench_klein_base_4b.json`). At the package-wide default 0.20 the gate over-fires and visibly degrades the image (an earlier sweep showed this; no record of it was committed); the per-variant default is the recommended setting. Polynomial calibrated at `guidance=1.0`; v0.4.1 reuses it under CFG only because the g=4.0 / 50-step release bench passed the skip and SSIM gates. The encoder-independent `mod_in` invariant justifies one shared branch decision per step; coefficient transfer remains empirical. | `COEFFICIENTS` in `variants/flux2_klein_base_4b/config.py`; `_PROVENANCE` in its `integration.py`. Calibration script: `scripts/calibrate_flux2.py --variant klein-base-4b --fit-mode origin`. Full report: `scripts/_calibration_flux2_klein_base_4b.json`. |
| `qwen-image-2512` | Derived in-repo on 2026-10-20 from 10 prompts (7 fit / 3 held-out) × 50 steps × seed=42 on M1 Max 32 GB, 4-bit, 768×768, guidance=4.0 (CFG), mflux 0.23.0, checkpoint `Qwen/Qwen-Image-2512`. **Origin-constrained** degree-4 fit on Signal A, the modulated block-0 image input the original Qwen-Image variant also gates on; R² = 0.818, held-out 0.785, over input changes from 0.0809 to 0.4706. The original checkpoint's tuple was scored on the same pairs and rejected: R² = −0.35 on the fit prompts and −0.20 on the held-out ones, worse than a constant. The default 0.25 comes from the sweep on the variant page. | `COEFFICIENTS` and `CALIBRATED_RANGE` in `variants/qwen_image_2512/config.py`; `_PROVENANCE` in its `integration.py`. Calibration script: `scripts/calibrate_qwen.py --model Qwen/Qwen-Image-2512`. Full report: `scripts/_calibration_qwen_2512.json`. |

## Producing new coefficients

Run the calibration script with the model loaded locally:

```bash
# Klein 4B (free polyfit; shipped since v0.2.0)
uv run python scripts/calibrate_flux2.py --variant klein-4b

# Klein 9B (origin-constrained polyfit; shipped since v0.3.0).
# IMPORTANT: do not omit --fit-mode origin — the default (free) fit gives
# a polynomial with poly(0) ≈ 5.36, which is physically nonsensical and
# was rejected during the v0.3.0 calibration audit.
uv run python scripts/calibrate_flux2.py --variant klein-9b --fit-mode origin

# Klein base-4B (origin-constrained polyfit; shipped since v0.4.0). Non-distilled
# variant designed for 20-50 step generation; this calibration uses 25 steps.
# The per-variant default rel_l1_thresh lives in variants/flux2_klein_base_4b/
# (DEFAULT_THRESH in config.py + Provenance.default_thresh in integration.py); if
# you re-calibrate, re-run scripts/sweep_threshold_klein_base_4b.py afterwards to
# confirm or re-tune that default.
uv run python scripts/calibrate_flux2.py --variant klein-base-4b --fit-mode origin
uv run python scripts/sweep_threshold_klein_base_4b.py   # tune the per-variant default

# CFG-aware calibration (v0.4.1+). Use when the g=1.0 polynomial does not
# engage acceptably at the canonical upstream recipe. Step count MUST match
# the release recipe (50 for base-4b under CFG) — calibrating at 25 steps
# and shipping at 50 would put off-trajectory coefficients in production.
uv run python scripts/calibrate_flux2.py \
  --variant klein-base-4b \
  --fit-mode origin \
  --guidance 4.0 \
  --num-inference-steps 50 \
  --fit-branch-policy worst

# klein-base-9b reuses base-4b's polynomial verbatim (shipped v0.5.0); run this
# only to override the reuse — see "Reusing coefficients across model sizes" below.
uv run python scripts/calibrate_flux2.py --variant klein-base-9b --fit-mode origin
```

The script monkeypatches `flux._predict` with a capturing factory that
reproduces the vanilla computation and records `(mod_in, body_out_concat)` per
step. It runs 10 prompts at seed=42 (8 steps for the distilled klein variants,
25 for the non-distilled base variants), computes per-step `rel_l1(t, t-1)` for
both signals, fits a degree-4 polynomial, and writes
`scripts/_calibration_flux2_<variant>.json` with the full report.

To bake the result back, replace the `COEFFICIENTS` tuple in the variant's
`variants/<id>/config.py` and update the `_PROVENANCE` record in that variant's
`integration.py` with the new `revision`, `calibration_dataset`, and
`fit_metric_value`.

## Custom user-supplied coefficients

Users can pass their own coefficients at integration time:

```python
my_coeffs = (236.92, -201.47, 66.91, -11.15, 1.27)
with apply_teacache(flux, coefficients=my_coeffs, rel_l1_thresh=0.20):  # your own threshold
    ...
```

Pass a `rel_l1_thresh` with your coefficients. A threshold is tuned for one polynomial, so without one `apply_teacache` picks 0.30 on FLUX.1 Krea [dev] and 0.20 on every other model, and warns with a `FutureWarning`. From 1.0 that call raises an error. After you calibrate a new polynomial, a threshold sweep finds the value to pass.

Where each family's calibration lives:

- FLUX.1 (dev, Krea): `scripts/calibrate_flux1.py`, which also runs the threshold sweep with `--sweep`.
- FLUX.2 Klein: `scripts/calibrate_flux2.py`; `scripts/sweep_threshold_klein_base_4b.py` sweeps the threshold for Klein base 4B.
- Qwen-Image (and 2512): `scripts/calibrate_qwen.py`; `scripts/sweep_threshold_qwen.py` sweeps the threshold.
- Z-Image: `scripts/calibrate_z_image.py`; `scripts/sweep_threshold_z_image.py` sweeps the threshold.

The handle's `provenance` field will record `source="user"` so calls to
`handle.provenance` make the origin clear.

Your polynomial is evaluated on the raw measured delta. The calibrated-range
clamp applies only to a variant's own built-in tuple.

## img2img coefficient reuse

v0.2.0 reuses txt2img coefficients for img2img generations. The polynomial
captures an architectural property (per-block residual sensitivity), so the
schedule slice should not change the fit much. A dedicated img2img
calibration may follow in v0.2.x if SSIM gates on real img2img workloads
show drift.

## Reusing coefficients across model sizes within an architecture family

v0.5.0 ships `flux2-klein-base-9b` reusing the polynomial calibrated for
`flux2-klein-base-4b` verbatim. Two conditions made this defensible:

1. **Same architecture family.** Both variants use the FLUX.2 Klein
   transformer (same block layout; different depth and hidden size). The
   polynomial maps cumulative input-modulation rel-L1 onto output rel-L1,
   a per-step property of the block structure rather than the parameter
   count.
2. **Same calibration recipe.** Both were calibrated (or in 9B's case,
   would be calibrated) at 25 inference steps, guidance=1.0,
   origin-constrained polyfit. A different recipe — say 50 steps with
   guidance=4.0 — traces a different trajectory through the latent space
   and produces a different polynomial; do not transfer fits across
   recipes (this is the v0.4.1 plan-audit Finding 3 lesson).

The reuse is converted from assumption to fact by a one-shot validation
pass before merge: `scripts/validate_klein_base_9b.py` generates one
fixed prompt at the canonical shipping recipe (50 steps + guidance=4.0)
both vanilla and wrapped, decodes through the VAE, and asserts SSIM ≥
0.95. The evidence lives at `_artifacts/validation_klein_base_9b.json`.

If the validation fails (SSIM too low, or wrapper skips zero steps at the
shared default threshold), the release holds and a fresh calibration runs
through `scripts/calibrate_flux2.py --variant klein-base-9b --fit-mode origin`
in a follow-up branch.

**When this pattern applies.** Same architecture + same recipe + empirical
validation. Reasonable for sibling variants within a generation (4B / 9B
within FLUX.2 Klein base). Not reasonable across different recipes
(g=1.0 → g=4.0 needs its own fit) or across families (FLUX.1 → FLUX.2
needs its own fit). When in doubt, calibrate fresh; the script is fast
relative to release timelines.
