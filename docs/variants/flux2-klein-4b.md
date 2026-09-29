# flux2-klein-4b

FLUX.2 Klein 4B — distilled, 8-step default. Same gate-engagement story as `flux1-schnell`, with a different mechanism still providing wall-clock benefit.

## Construct via mflux

```python
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.common.config.model_config import ModelConfig

flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())
```

## Recipe + defaults

- Default recipe: 8 steps, `guidance=1.0`
- Default `rel_l1_thresh`: `None` (caller passes one, or the package fallback 0.20 is used)
- skip-window defaults: `skip_first_n_steps=1`, `skip_last_n_steps=1`

At 8 steps the empirical adjacent-step body-output rel-L1 starts at 0.25 — above the 0.20 fallback threshold. Gate produces 0 step-skips. On Macs where mflux compiles `_predict` (Max and Ultra chips, and M3 and newer; not base or Pro M1/M2), the wrapper can still change wall-clock time by sidestepping that compiled path. This repository has no committed multi-run measurement of that effect on the 8-step schedule, so no figure is quoted. On the 50-step klein-base CFG recipe the same mechanism measures 1.01×. Where mflux already runs eager, there is nothing to sidestep and the wrapper only adds gate overhead.

## Coefficient provenance

Derived in-repo by `scripts/calibrate_flux2.py --variant klein-4b` on 2026-05-15:
- 10 prompts × 8 steps × seed=42 on M1 Max 32 GB, bf16, 512×512, guidance=1.0
- 70 consecutive-step pairs of `(rel_l1(mod_in_t, mod_in_{t-1}), rel_l1(body_out_t, body_out_{t-1}))`
- `numpy.polyfit` degree=4, R² = 0.6530
- Stored verbatim in `src/mlx_teacache/variants/flux2_klein_4b/config.py::COEFFICIENTS`

See `scripts/_calibration_flux2_klein_4b.json` for the full report.

## License

[`Apache-2.0`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B). No usage restrictions beyond the standard Apache obligations.

## Quirks

- **Gate never engages at the distilled default.** If wall-clock time changes at all, the cause is `mx.compile` avoidance on the FLUX.2 `_predict` path (where mflux compiles it), not step-skipping.
- The integration cross-imports `make_teacache_predict_factory` and the forward functions from `flux2_klein_base_4b`. Same architecture family.
