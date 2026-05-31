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

Gate engagement is the same shape as klein-4b: the distilled 8-step schedule produces adjacent-step body-output rel-L1 above the threshold, so 0 skips at the package default. Wrapper benefit on M1 Max comes from `mx.compile`-path avoidance. The v0.4-era same-process bench reported a wide 1.5-2.0× range with high thermal variance on the 8-step schedule (one rep combined a thermally-throttled 227 s vanilla with a recovered 46 s wrapper to land 1.93× median; steady-state is closer to 1.5×). This row is pending a re-bench under v0.6.0's subprocess-per-rep harness. v0.6.0's re-measurement of the same mechanism on the 50-step klein-base-9b CFG recipe came out at 1.02× — so the wide distilled-row figure is specific to short schedules where per-step dispatch overhead dominates wall-clock.

## Coefficient provenance

Derived in-repo by `scripts/calibrate_flux2.py --variant klein-9b --fit-mode origin` on 2026-05-16:
- 10 prompts × 8 steps × seed=42 on M1 Max 32 GB, bf16, 512×512, guidance=1.0
- 70 consecutive-step pairs of `(rel_l1(mod_in_t, mod_in_{t-1}), rel_l1(body_out_t, body_out_{t-1}))`
- Origin-constrained least-squares fit (forces `poly(0)=0` for physical sensibility at small input rel-L1), R² = 0.4710
- Stored verbatim in `src/mlx_teacache/variants/flux2_klein_9b/config.py::COEFFICIENTS`

See `scripts/_calibration_flux2_klein_9b.json` for the full report. The distilled gate doesn't engage because the empirical adjacent-step body-output rel-L1 (≈0.25) exceeds the default `rel_l1_thresh`, so Klein's wall-clock win comes from `mx.compile`-path avoidance rather than caching.

## License

[`FLUX Non-Commercial`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) — accept on the Hugging Face model page before downloading. **For commercial use, see Black Forest Labs licensing.**

## Quirks

- **Gate never engages at the distilled default**; wall-clock improvement is from `mx.compile` avoidance.
- The integration cross-imports the forward + factory from `flux2_klein_base_4b`. Same architecture family.
