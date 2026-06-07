# Coefficient calibration

mlx-teacache's gate uses a degree-4 polynomial that maps the relative-L1
distance of the modulated block-0 input (`rel_l1(mod_in_t, mod_in_{t-1})`)
to the predicted relative-L1 distance of the transformer output. Coefficients
live per variant in `src/mlx_teacache/variants/<id>/config.py` (`COEFFICIENTS`),
with the matching `Provenance` in that variant's `integration.py`. The top-level
`coefficients.py` is a thin re-export shim, extracted in v0.6.0; editing it
changes nothing.

## Built-in coefficient sources

| Variant | Source | Provenance |
|---|---|---|
| `flux1-dev`, `flux1-schnell` | Vendored from ali-vilab/TeaCache (`TeaCache4FLUX/teacache_flux.py`); Apache-2.0. | `COEFFICIENTS` in `variants/flux1_dev/config.py` (shared by `flux1_schnell` via cross-import). The FLUX dev/schnell architecture is shared, so both reuse the same set. |
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
# The per-variant default rel_l1_thresh is shipped via Provenance.default_thresh
# in coefficients.py — if you re-calibrate, re-run scripts/sweep_threshold_klein_base_4b.py
# afterwards to confirm or re-tune that default.
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

To bake the result back into `coefficients.py`, replace the tuple value
under the appropriate variant id and update the Provenance dataclass with
the new `revision`, `calibration_dataset`, and `fit_metric_value`.

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
