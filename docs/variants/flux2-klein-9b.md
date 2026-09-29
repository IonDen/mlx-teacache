# flux2-klein-9b

FLUX.2 Klein 9B — distilled, 8-step default. Larger model, same engagement story as `flux2-klein-4b`.

## Construct via mflux

```python
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.common.config.model_config import ModelConfig

flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_9b())
```

## Recipe + defaults

- Default recipe: 8 steps, `guidance=1.0`
- Default `rel_l1_thresh`: `None` (caller passes one, or the package fallback 0.20 is used)
- skip-window defaults: `skip_first_n_steps=1`, `skip_last_n_steps=1`

Gate engagement is the same shape as klein-4b: the distilled 8-step schedule produces adjacent-step body-output rel-L1 above the threshold, so 0 skips at the package default. Any wall-clock difference comes from `mx.compile`-path avoidance, and only on Macs where mflux compiles `_predict` (Max and Ultra chips, and M3 and newer; not base or Pro M1/M2). This repository has no committed multi-run measurement of that effect on the 8-step schedule, so no figure is quoted. The same mechanism measures 1.02× on the 50-step klein-base-9b CFG recipe.

## Coefficient provenance

Derived in-repo by `scripts/calibrate_flux2.py --variant klein-9b --fit-mode origin` on 2026-05-16:
- 10 prompts × 8 steps × seed=42 on M1 Max 32 GB, bf16, 512×512, guidance=1.0
- 70 consecutive-step pairs of `(rel_l1(mod_in_t, mod_in_{t-1}), rel_l1(body_out_t, body_out_{t-1}))`
- Origin-constrained least-squares fit (forces `poly(0)=0` for physical sensibility at small input rel-L1), R² = 0.4710
- Stored verbatim in `src/mlx_teacache/variants/flux2_klein_9b/config.py::COEFFICIENTS`

See `scripts/_calibration_flux2_klein_9b.json` for the full report. The distilled gate doesn't engage because the empirical adjacent-step body-output rel-L1 (≈0.25) exceeds the default `rel_l1_thresh`, so any wall-clock difference comes from `mx.compile`-path avoidance, where mflux compiles `_predict`, rather than caching.

## License

[`FLUX Non-Commercial`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) — accept on the Hugging Face model page before downloading. **For commercial use, see Black Forest Labs licensing.**

## Quirks

- **Gate never engages at the distilled default**; a change in wall-clock time is `mx.compile` avoidance, where mflux compiles `_predict`, and nothing else.
- The integration cross-imports the forward + factory from `flux2_klein_base_4b`. Same architecture family.
