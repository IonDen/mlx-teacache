# flux1-dev

FLUX.1 dev — the non-distilled FLUX.1 base model, 25-step canonical schedule.

## Construct via mflux

```python
from mflux.models.flux.variants.txt2img.flux import Flux1

flux = Flux1.from_name("dev", quantize=4)
```

## Recipe + defaults

- Default recipe: 25 steps, `guidance=3.5`
- Default `rel_l1_thresh`: 0.20 (the package fallback)
- skip-window defaults: `skip_first_n_steps=1`, `skip_last_n_steps=1`

At the default threshold and 25 steps the gate skips 6/25 steps (never two in a row) and produces a measured 1.30× wall-clock speedup on M1 Max (median of three cold reps, 2026-11-16; mflux 0.24.0, MLX 0.32.2) with SSIM ≥ 0.80 on the 5-prompt suite (≥ 0.90 on the PR-gate red-apple prompt). The record is `_artifacts/v0.18.0_bench_flux1_dev.json`: plain mflux 89.46 s, no-gate wrapper 89.46 s, gated wrapper 68.74 s, commit e7cb468, clean tree. The no-gate run computes every step and took the same time as plain mflux (89.46 s both), so the wrapper alone gains nothing on mflux 0.24, and all of the 1.30× comes from the 6 skipped steps. Of the 25 steps, 19 were computed: 17 the gate decided, plus the first and last step, which always run. Six skipped steps can therefore give at most about 1.32× (25/19), and the 1.30× is under that limit. Each rep is a fresh process with no warmup generation, so each timing includes first-call setup.

The record names commit e7cb468, a pre-release build of 0.18.0 on the development branch; that commit is not in the published history. Between it and the 0.18.0 release three files differ: the warning texts in `src/mlx_teacache/api.py`, a docstring in `src/mlx_teacache/support.py`, and the benchmark script's git and resume handling (a timeout, an explicit untracked-files flag, and reading back only the start keys on resume). The generation path the bench timed is the same, and `pyproject.toml` and `uv.lock` are identical. The record's version string, `v0.16.0-20-ge7cb468`, names that same commit, counted from the v0.16.0 tag.

History: an older run on mflux 0.18.0 (median of three cold reps, 2026-08-15) measured 1.57×, and 1.50× fastest to fastest. Both are above the 1.32× limit worked out above. In that run the plain-mflux reps varied between 104.7 and 119.8 s, while the 0.24.0 run, with steady plain times, measured 1.30×. The record (`_artifacts/v0.10.0_bench_flux1_dev.json`) has no commit or dirty field, no timestamp and no MLX version; its install-time version string names commit 90fca33. The date comes from the README at the v0.10.0 release. MLX 0.31.2 per the lock file at c33e8d5, the commit that added the record. The gate measured is the one as of the v0.10.0 work, which differs slightly from the release's. mflux 0.22.0 loads the bias of FLUX.1's last normalization layer again, which changes its images slightly, so that figure holds for mflux 0.18.0. See [`docs/benchmarks.md` → Benchmarks](../benchmarks.md#benchmarks).

## Coefficient provenance

Vendored verbatim from [`ali-vilab/TeaCache`](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py). License: Apache-2.0. See `NOTICE` for attribution.

The polynomial fits the relative-L1 distance of the modulated block-0 input to the predicted relative-L1 distance of the transformer output. FLUX.1 dev and schnell share the same transformer architecture; `flux1-schnell.config.COEFFICIENTS` cross-imports from this variant so the tuple is identity-equal.

## License

[`FLUX.1-dev Non-Commercial License`](https://huggingface.co/black-forest-labs/FLUX.1-dev) — accept on the Hugging Face model page before downloading.

## Quirks

None worth flagging. This is the reference variant.
