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

At the canonical 50-step CFG recipe on M1 Max 32 GB (subprocess-per-rep, 3 reps, bf16, q4, v0.6.0 bench harness):

| Condition | Median wall-clock | Peak memory |
|---|---|---|
| vanilla | 236.2 s | ~10.7 GB |
| wrapper, no gate (compile-avoidance only) | 233.4 s | ~5.9 GB |
| wrapper, gated (full TeaCache) | 191.8 s | ~5.9 GB |

- **Combined speedup: 1.23×**
- **Gating contribution (v0.4.1 effect): 1.22×**
- **`mx.compile`-path avoidance (v0.4 effect): 1.01×** — effectively noise on M1 Max at this recipe; the wrapper's main `mx.compile` benefit is memory (~45% peak drop), not wall-clock
- Skip count stable across reps: 9 of 48 active steps skipped at `rel_l1_thresh=0.17` (byte-identical to v0.4.1's algorithmic skip count)

At the 25-step `low_step` recipe (`guidance=1.0`) the gate skips 3/25 with **1.41× wall-clock** and SSIM > 0.99 (v0.4.0 same-process measurement; not re-bench'd under subprocess-per-rep yet).

> **Note on the v0.4.1 claim.** v0.4.1 reported 1.26× combined on this variant, decomposed as 1.16× gating × 1.09× compile-avoidance. The 1.26× combined number was honest within day-to-day noise; v0.6.0's subprocess-per-rep harness lands 1.23×, a 2.4% difference well inside any reasonable measurement band. The *decomposition* is what needed correcting. With subprocess isolation, gating is doing essentially all the work (1.22×) and compile-avoidance is at noise level (1.01×). The v0.4.1 same-process harness had the wrapper inheriting warm allocator state, which got attributed to compile-avoidance.

Reproduce with `uv run python scripts/bench_speedup.py --variant klein-base-4b --three-way --reps 3 --report out.json`. Full report at `_artifacts/v0.6.0_bench_klein_base_4b.json`; side-by-side images at `tests/_artifacts/bench_images/klein-base-4b/`.

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
