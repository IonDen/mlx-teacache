# Coefficient calibration

mlx-teacache's gate uses a degree-4 polynomial that maps the relative-L1
distance of the modulated block-0 input (`rel_l1(mod_in_t, mod_in_{t-1})`)
to the predicted relative-L1 distance of the transformer output. Coefficients
are stored per variant in `src/mlx_teacache/coefficients.py`.

## Built-in coefficient sources

| Variant | Source | Provenance |
|---|---|---|
| `flux1-dev`, `flux1-schnell` | Vendored from ali-vilab/TeaCache (`TeaCache4FLUX/teacache_flux.py`); Apache-2.0. | `_UPSTREAM_FLUX_COEFFS` in `coefficients.py`. The FLUX dev/schnell architecture is shared, so both reuse the same set. |
| `flux2-klein-4b` | Derived in-repo on 2026-05-15 from 10 prompts × 8 steps × seed=42 on M1 Max 32GB, bf16, 512×512, guidance=1.0. `numpy.polyfit(degree=4)` on 70 consecutive-step `(mod_in, body_out)` rel-L1 pairs; R² = 0.65. | `_REGISTRY["flux2-klein-4b"]` in `coefficients.py`. Calibration script: `scripts/calibrate_flux2.py --variant klein-4b`. Full report: `scripts/_calibration_flux2_klein_4b.json`. |
| `flux2-klein-9b` | Derived in-repo on 2026-05-16 from 10 prompts × **8 steps** × seed=42 on M1 Max 32GB, bf16, 512×512, guidance=1.0. `numpy.polyfit(degree=4)` on 70 consecutive-step `(mod_in, body_out)` rel-L1 pairs; R² = 0.5421. **Target schedule: `num_inference_steps=8`.** At the mflux default of 4 steps with default skip windows the gate has only one possible skip per generation; ship with that bound in mind. | `_REGISTRY["flux2-klein-9b"]` in `coefficients.py`. Calibration script: `scripts/calibrate_flux2.py --variant klein-9b`. Full report: `scripts/_calibration_flux2_klein_9b.json`. |

## Producing new coefficients

Run the calibration script with the model loaded locally:

```bash
# Klein 4B
uv run python scripts/calibrate_flux2.py --variant klein-4b
# Klein 9B
uv run python scripts/calibrate_flux2.py --variant klein-9b
# klein-base-4b and klein-base-9b are declared but raise
# NotImplementedError until v0.4.0 and v0.5.0 respectively.
```

The script monkeypatches `flux._predict` with a capturing factory that
mirrors the vanilla math while recording `(mod_in, body_out_concat)` per
step. It runs 10 prompts × 8 steps at seed=42, computes per-step
`rel_l1(t, t-1)` for both signals, fits a degree-4 polynomial, and writes
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
