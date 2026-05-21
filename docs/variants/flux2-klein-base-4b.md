# flux2-klein-base-4b

FLUX.2 Klein base 4B — non-distilled, 50-step canonical CFG recipe. **Primary use case for TeaCache on the FLUX.2 family.**

## Construct via mflux

```python
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.common.config.model_config import ModelConfig

flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
```

## Recipe + defaults

- Default recipe: 50 steps, `guidance=4.0` (canonical CFG)
- Alternative recipe `low_step`: 25 steps, `guidance=1.0` (v0.4.0 row)
- Default `rel_l1_thresh`: **0.17** (per-variant default, tuned via the v0.4.0 sweep)
- skip-window defaults: `skip_first_n_steps=1`, `skip_last_n_steps=1`

At the canonical 50-step CFG recipe the gate skips 9/50 steps and the wrapper measures **1.26× wall-clock** vs vanilla mflux. At the 25-step `low_step` recipe (`guidance=1.0`) the gate skips 3/25 with **1.41× wall-clock** and SSIM > 0.99.

The 0.17 default was chosen because the polynomial R² is low (0.106) and the engagement window is narrow:
- At 0.20 (package default): 19/25 skips, SSIM ~0.76 — visibly degraded.
- At **0.17**: 3/25 skips, SSIM 0.99 — indistinguishable from vanilla.
- At 0.175: 14/25 skips, SSIM ~0.78 — sharp cliff.

See `scripts/sweep_threshold_klein_base_4b.py` for the sweep methodology.

## CFG (guidance > 1.0)

CFG runs through a per-branch gated path (`flux2_cfg_forward_with_gate`, v0.4.1+). Both positive and negative branches go through the same cached-residual logic; each branch maintains its own `cached_residual`. The eager-Python wrapper diverges from vanilla mflux's compiled `_predict` by ~1 ULP per branch per step, compounding across steps to a cosine similarity ≥ 0.97 — well above the parity gate but not bit-exact.

## Coefficient provenance

Origin-constrained polyfit derived in-repo on 2026-05-17 from 25-step (non-distilled) trajectories:
- `scripts/calibrate_flux2.py --variant klein-base-4b --fit-mode origin`
- 10 prompts × 25 steps × seed=42 on M1 Max 32 GB, bf16, 512×512, guidance=1.0
- Origin-constrained least-squares fit (forces `poly(0)=0`), R² = 0.106
- Stored verbatim in `src/mlx_teacache/variants/flux2_klein_base_4b/config.py::COEFFICIENTS`

The polynomial output range `[0.144, 0.233]` straddles 0.20 — gate is structurally capable of engaging (unlike distilled klein-4b/9b where the polynomial never dips below 0.20).

See `scripts/_calibration_flux2_klein_base_4b.json` for the full report.

## License

[`Apache-2.0`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B). No usage restrictions beyond the standard Apache obligations.

## Quirks

- **Default threshold is 0.17, not the package fallback 0.20.** This is set via `Provenance.default_thresh` in the variant's `_PROVENANCE` and resolved at `apply_teacache` time.
- The CFG path's cosine ≥ 0.97 tolerance is documented in `tests/test_parity_flux2.py::_FLUX2_COSINE_GATE`. Compile-vs-eager dispatch noise compounds across steps — see the test module docstring.
- `flux2_klein_base_9b` cross-imports COEFFICIENTS from this variant — same architecture family, same calibration recipe.
