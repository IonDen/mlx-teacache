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

At the default threshold and 25 steps the gate skips 6/25 steps and produces a measured 1.46× wall-clock speedup on M1 Max with SSIM ≥ 0.80 on the 5-prompt suite (≥ 0.90 on the PR-gate red-apple prompt). See [`README.md` → Benchmarks](../../README.md#benchmarks).

## Coefficient provenance

Vendored verbatim from [`ali-vilab/TeaCache`](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py). License: Apache-2.0. See `NOTICE` for attribution.

The polynomial fits the relative-L1 distance of the modulated block-0 input to the predicted relative-L1 distance of the transformer output. FLUX.1 dev and schnell share the same transformer architecture; `flux1-schnell.config.COEFFICIENTS` cross-imports from this variant so the tuple is identity-equal.

## License

[`FLUX.1-dev Non-Commercial License`](https://huggingface.co/black-forest-labs/FLUX.1-dev) — accept on the Hugging Face model page before downloading.

## Quirks

None worth flagging. This is the reference variant.
