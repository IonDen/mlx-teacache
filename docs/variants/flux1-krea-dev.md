# flux1-krea-dev

FLUX.1 Krea [dev] (Black Forest Labs with Krea) — a FLUX.1-dev-architecture finetune that mflux loads through its FLUX.1 path. It shares the transformer and the gate signal with `flux1-dev`, so the variant reuses the FLUX.1 proxy strategy verbatim; what it does not share is the polynomial. Krea changes far more per denoising step than dev, and FLUX.1-dev's coefficients do not transfer, so this variant carries its own calibration and its own default threshold.

## Facts

- Model: [`black-forest-labs/FLUX.1-Krea-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev), gated (accept the licence on the model page; needs an HF token).
- License: FLUX.1-dev Non-Commercial License, same as `flux1-dev`.
- Recipe: 28 steps, guidance 4.5 (the model card's), 512×512 for the committed numbers; non-distilled.
- Detection: mflux's `krea-dev` / `dev-krea` aliases. `flux1-dev`'s detector matches only `dev`, so the two never collide.
- Default `rel_l1_thresh`: **0.30**, set at the sweep knee (below).
- Patch strategy: FLUX.1 proxy transformer (`ProxyFlux1Transformer` + `flux1_forward_with_gate`), so the `flux.parameters()` caveat from the README applies here too.

## Coefficients: why FLUX.1-dev's tuple was rejected

`scripts/calibrate_flux1.py --model krea-dev` captured per-step pairs of gate-signal change and body-output change for ten calibration prompts at the recipe above (270 pairs). Scored on those pairs, FLUX.1-dev's vendored tuple gives **R² −496**: its dominant 499·x⁴ term, fit on dev's small per-step changes (relative L1 around 0.1–0.25), explodes at Krea's (0.06–0.66). A fresh degree-4 fit on Krea's own pairs gives **R² 0.68** (free fit; the origin-constrained fit gives 0.66 and was not chosen). That is between the FLUX.2 family (0.11–0.65) and Qwen-Image (0.85), and the sweep, not the R², sets the quality bar. The fit and the score are committed in `scripts/_calibration_flux1_krea_dev.json`, and a test pins the config tuple to it.

One property of the fit matters for users: its intercept is about 0.40, and the smallest predicted change per step over the calibration trajectories is about 0.26. The package fallback `rel_l1_thresh=0.20` would therefore never skip on Krea; the per-variant default exists because of that.

## Threshold sweep

`scripts/calibrate_flux1.py --model krea-dev --sweep 0.30,0.35,0.40,0.60,0.80`, red-apple prompt, seed 42, 28 steps, guidance 4.5, 512×512, q4, one gated generation per threshold against a vanilla reference (M1 Max 32 GB, 2026-09-05; single-rep timings are thermal noise, the multi-rep bench below is the headline):

| `rel_l1_thresh` | Skipped (of 26 active) | Longest streak | SSIM vs vanilla | Single-rep speedup |
|---|---|---|---|---|
| 0.30 | 10 | 1 | 0.990 | 1.52× |
| 0.35 | 12 | 1 | 0.890 | 1.80× |
| 0.40 | 13 | 1 | 0.888 | 1.91× |
| 0.60 | 16 | 2 | 0.866 | 2.47× |
| 0.80 | 18 | 3 | 0.863 | 3.12× |

The knee is sharp: 0.30 holds SSIM 0.99 while skipping 10 of the 26 active steps in a strict alternation, and 0.35 already drops under the 0.90 bar the FLUX.1 quality gate uses. Raising the threshold buys speed at a visible cost on this model, so 0.30 ships and the parity lane pins a 9–11 skip band with an SSIM floor of 0.95 at that default.

## Benchmark

`scripts/bench_speedup.py --variant krea-dev --three-way --reps 3`, red-apple prompt, seed 42, 28 steps, guidance 4.5, 512×512, q4, one fresh subprocess per (condition, rep) so every timing starts cold, chunks interleaved rep-outer, medians of three (M1 Max 32 GB, macOS 26, mflux 0.18.0, 2026-09-05; `_artifacts/v0.11.0_bench_krea_dev.json`):

| Condition | Rep seconds | Median | Skipped (of 26 active) |
|---|---|---|---|
| vanilla | 125.3 / 129.3 / 123.9 | 125.3 s | — |
| wrapper, no gate (`rel_l1_thresh=0.0`) | 121.4 / 127.1 / 119.5 | 121.4 s | 0 |
| wrapper, default 0.30 | 77.2 / 79.6 / 66.5 | 77.2 s | 10 in every rep, longest streak 1 |

**1.62× combined** against vanilla, of which **1.57×** is gating (no-gate 121.4 s → gated 77.2 s). The no-gate wrapper's 1.03× over vanilla is run-to-run noise: mflux does not `mx.compile` the FLUX.1 predict step, so there is no compiled path for the eager proxy to avoid. Every rep produced the same skip pattern (`CCCSCSCSCSCSCSCSCSCSCSCCCCCC`: the forced first step, the seed step, then a strict alternation until the trailing forced steps). Peak memory 11.1 GB in every condition; the report's load-time peak reads 0 because mflux materialises weights on first use, so the loop peak is the whole story here.

## Reproduce

```bash
uv run python scripts/calibrate_flux1.py --model krea-dev --max-prompts 1        # capture, one prompt per invocation
uv run python scripts/calibrate_flux1.py --model krea-dev --score-coefficients 498.651651244,-283.781631,55.8554382,-3.82021401,0.264230861
uv run python scripts/calibrate_flux1.py --model krea-dev --sweep 0.30,0.35,0.40,0.60,0.80 --max-units 1
uv run python scripts/bench_speedup.py --variant krea-dev --three-way --reps 3 --max-chunks 1 --report out.json
uv run pytest tests/test_parity_krea.py -m parity
```
