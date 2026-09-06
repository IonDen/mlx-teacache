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

## Runaway guard

The gate accumulates the polynomial's predicted change across consecutive
skips and resets it to zero on every real compute. The
origin-constrained in-repo fits are positive for small deltas but cross zero
at large ones (base-4b at x≈0.24, z-image at x≈0.29, qwen at x≈0.78), beyond
the range they were fit on; there the clamp turns a large, real change into
a predicted change of zero, so the accumulator stops advancing — without a
guard, that would let the wrapper reuse the same cached residual for an
unbounded number of steps. `MAX_CONSECUTIVE_SKIPS` in `src/mlx_teacache/_kernel/gate.py`
forces a recompute after 8 consecutive skips regardless of the accumulated
total. This is an intentional divergence from upstream ali-vilab TeaCache,
which has no such cap — the same kind of deliberate departure as the gate's
`max(0.0, ...)` clamp on the polynomial output, which keeps the accumulator
monotonic instead of matching upstream's raw polynomial exactly.

### Observed max consecutive-skip streaks

Measured at each variant's default threshold on the committed bench recipes
(M1 Max 32 GB, mflux 0.18.0, three cold reps per condition; the first four on
2026-08-15, qwen-image on 2026-09-01). Every row but qwen-image renders at
512×512; qwen-image uses its pinned 768×768. The bench reports carry a per-rep
`skip_patterns` string (`S` = skipped, `C` = computed) and
`max_consecutive_skips`; the streak below is the maximum across reps, and in
every case each rep produced the same pattern.

| Variant | Default threshold | Skips (active steps) | Max observed streak | Source |
|---|---|---|---|---|
| `flux1-dev` | 0.20 | 6 / 23 | 1 | `_artifacts/v0.10.0_bench_flux1_dev.json` (25 steps, g=3.5) |
| `flux2-klein-base-4b` | 0.17 | 9 / 48 | 2 | `_artifacts/v0.10.0_bench_klein_base_4b.json` (50 steps, g=4.0) |
| `flux2-klein-base-9b` | 0.17 | 13 / 48 | 1 | `_artifacts/v0.10.0_bench_klein_base_9b.json` (50 steps, g=4.0) |
| `z-image-base` | 0.12 | 15 / 48 | 1 | `_artifacts/v0.10.0_bench_z_image.json` (50 steps, g=4.0, q8) |
| `qwen-image` | 0.30 | 33 / 48 | 4 | `_artifacts/v0.10.0_bench_qwen_image.json` (50 steps, g=4.0, 768×768) |
| `flux1-krea-dev` | 0.30 | 10 / 26 | 1 | `_artifacts/v0.11.0_bench_krea_dev.json` (28 steps, g=4.5) |

Four of the five stay at 1 or 2: the gate mostly alternates compute / skip and
reuses a residual at most twice in a row. `qwen-image` runs longer, reaching 4,
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
threshold. A threshold sweep on 2026-09-06 under the current gate (stock q4,
same recipe) put the curve on record: 24 / 26 / 30 / 33 skips at 0.15 / 0.20 /
0.25 / 0.30, SSIM 0.980 / 0.980 / 0.976 / 0.967, longest streaks 2 / 2 / 3 / 4,
and 0.30 stays the default (see the variant page).

## Built-in coefficient sources

| Variant | Source | Provenance |
|---|---|---|
| `flux1-dev`, `flux1-schnell` | Vendored from ali-vilab/TeaCache (`TeaCache4FLUX/teacache_flux.py`); Apache-2.0. | `COEFFICIENTS` in `variants/flux1_dev/config.py` (shared by `flux1_schnell` via cross-import). The FLUX dev/schnell architecture is shared, so both reuse the same set. |
| `flux1-krea-dev` | Derived in-repo on 2026-09-05 from 10 prompts × 28 steps × seed=42 on M1 Max 32GB, bf16, 512×512, guidance=4.5 (the model card's recipe). Free `numpy.polyfit(degree=4)` on 270 consecutive-step `(mod_in, body_out)` rel-L1 pairs; R² = 0.68 (the origin-constrained fit gave 0.66). FLUX.1-dev's vendored tuple was scored on the same pairs and rejected at R² = −496: Krea's per-step changes reach rel-L1 0.66 where dev's stay near 0.2, and dev's 499·x⁴ term explodes there. The fit's intercept (~0.40) makes the package fallback 0.20 skip nothing; the per-variant default 0.30 comes from the sweep on the variant page. | `COEFFICIENTS` in `variants/flux1_krea_dev/config.py`; `_PROVENANCE` in its `integration.py`. Calibration and sweep script: `scripts/calibrate_flux1.py --model krea-dev`. Full report with the scored tuple: `scripts/_calibration_flux1_krea_dev.json`. |
| `flux2-klein-4b` | Derived in-repo on 2026-05-15 from 10 prompts × 8 steps × seed=42 on M1 Max 32GB, bf16, 512×512, guidance=1.0. `numpy.polyfit(degree=4)` on 70 consecutive-step `(mod_in, body_out)` rel-L1 pairs; R² = 0.65. | `COEFFICIENTS` in `variants/flux2_klein_4b/config.py`; `_PROVENANCE` in its `integration.py`. Calibration script: `scripts/calibrate_flux2.py --variant klein-4b`. Full report: `scripts/_calibration_flux2_klein_4b.json` (summary only; unlike the klein-9b and base-4b reports it omits the raw `x_values`/`y_values` arrays, so reproducing this tuple offline means re-running the ~196 s calibration). |
| `flux2-klein-9b` | Derived in-repo on 2026-05-16 from 10 prompts × **8 steps** × seed=42 on M1 Max 32GB, bf16, 512×512, guidance=1.0. **Origin-constrained** least-squares fit (forces `poly(0) = 0`) on 70 consecutive-step `(mod_in, body_out)` rel-L1 pairs; R² = 0.4710. At the package default `rel_l1_thresh=0.20` these coefficients trigger **0 step-skips** on Klein 9B's 8-step schedule — empirical `y_min = 0.25` exceeds the threshold. Wall-clock benefit on Klein comes from `mx.compile`-path avoidance, not from caching. | `COEFFICIENTS` in `variants/flux2_klein_9b/config.py`; `_PROVENANCE` in its `integration.py`. Calibration script: `scripts/calibrate_flux2.py --variant klein-9b --fit-mode origin`. Full report: `scripts/_calibration_flux2_klein_9b.json`. |
| `flux2-klein-base-4b` | Derived in-repo on 2026-05-17 from 10 prompts × **25 steps** × seed=42 on M1 Max 32GB, bf16, 512×512, guidance=1.0. **Origin-constrained** least-squares fit (forces `poly(0) = 0`); R² = 0.106. Low R² for the polynomial form — FLUX.2-family polynomials are noisier than FLUX.1-family. To compensate, this variant ships a per-variant default `rel_l1_thresh=0.17` via `Provenance.default_thresh` (set automatically). At that threshold the gate fires 3/25 skips for 1.41× speedup with SSIM ≥ 0.99 vs vanilla. At the package-wide default 0.20 the gate over-fires (19/25 skips, SSIM=0.76); the per-variant default is the recommended setting. Polynomial calibrated at `guidance=1.0`; v0.4.1 reuses it under CFG only because the g=4.0 / 50-step release bench passed the skip and SSIM gates. The encoder-independent `mod_in` invariant justifies one shared branch decision per step; coefficient transfer remains empirical. | `COEFFICIENTS` in `variants/flux2_klein_base_4b/config.py`; `_PROVENANCE` in its `integration.py`. Calibration script: `scripts/calibrate_flux2.py --variant klein-base-4b --fit-mode origin`. Full report: `scripts/_calibration_flux2_klein_base_4b.json`. |

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
with apply_teacache(flux, coefficients=my_coeffs):
    ...
```

The handle's `provenance` field will record `source="user"` so calls to
`handle.provenance` make the origin clear.

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
