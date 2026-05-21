# flux1-schnell

FLUX.1 schnell — distilled FLUX.1, 4-step default. **Not a good fit for TeaCache.**

## Construct via mflux

```python
from mflux.models.flux.variants.txt2img.flux import Flux1

flux = Flux1.from_name("schnell", quantize=4)
```

## Recipe + defaults

- Default recipe: 4 steps, `guidance=1.0`
- Default `rel_l1_thresh`: 0.20 (package fallback)
- skip-window defaults: `skip_first_n_steps=1`, `skip_last_n_steps=1`

At 4 steps with `skip_first=1 + skip_last=1`, only 2 steps are gate-eligible. The polynomial output exceeds the 0.20 threshold on every adjacent pair (the distilled schedule trains the model to make large strides per step), so the gate produces 0 skips. The wrapper adds 1-2% gating overhead with no benefit. Run schnell through vanilla mflux.

## Coefficient provenance

Cross-imported from `flux1-dev`. FLUX.1 dev and schnell share the same transformer architecture; the upstream `ali-vilab/TeaCache` registry uses the same coefficient set for both. `flux1_schnell.config.COEFFICIENTS is flux1_dev.config.COEFFICIENTS`.

## License

[`Apache-2.0`](https://huggingface.co/black-forest-labs/FLUX.1-schnell). No usage restrictions beyond the standard Apache obligations.

## Quirks

- **Gate never engages at the distilled default.** Use vanilla mflux for production schnell workloads.
- Apply detects via `"schnell" in model_config.aliases`.
