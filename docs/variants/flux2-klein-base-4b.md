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

At the canonical 50-step CFG recipe on M1 Max 32 GB (subprocess-per-rep, 3 reps interleaved rep-outer, bf16, q4, mflux 0.18.0, v0.12.0 bench, 2026-09-28):

| Condition | Median wall-clock | Peak memory |
|---|---|---|
| vanilla | 237.5 s | ~10.7 GiB |
| wrapper, no gate (compile-avoidance only) | 234.2 s | ~5.5 GiB |
| wrapper, gated (full TeaCache) | 197.8 s | ~5.5 GiB |

The record (`_artifacts/v0.12.0_bench_klein_base_4b.json`) names commit ec4d110 with a clean tree and has no date field or MLX version field; MLX 0.31.2 per the lock file at ec4d110. The date comes from the record's chunk timestamps, and `scripts/` and `src/` are identical at ec4d110 and at 1290c01, the commit that added the record, on the pull-request branch (public in [PR #42](https://github.com/IonDen/mlx-teacache/pull/42), squashed into main as 0a3482c; that merge also added to `gate.py` the rule that a change above the calibrated range always computes). The record stores the Darwin kernel version, 27.0.0, which maps to macOS 27.0. mflux 0.24.0 changed the default image decode for FLUX.2 Klein to use far less memory, and the bench timer includes the decode, so the timings and the memory peaks in the table hold for mflux 0.18.0.

- **Combined speedup: 1.20×**. The 8 skipped steps of 50 alone allow 1.19× (50/42); the rest is the 1.01× compile avoidance. Each factor below is rounded on its own from the unrounded medians, so the rounded factors need not multiply to the rounded total. (v0.10.0 measured 1.22× and v0.6.0 1.23×, with one more skip; the 1.23× is above its 1.22× limit, 50/41; see below)
- **Gating contribution (v0.4.1 effect): 1.18×**
- **`mx.compile`-path avoidance (v0.4 effect): 1.01×** — effectively noise on M1 Max at this recipe; the wrapper's main `mx.compile` benefit is memory (~48% peak drop), not wall-clock
- Skip count stable across reps: 8 of 48 active steps skipped at `rel_l1_thresh=0.17`, never two in a row
- SSIM 0.985 vs vanilla at this recipe (`_artifacts/v0.12.0_klein_base_ssim.json`)

Until 0.12.0 the gate also skipped step 6 here (9 of 48, one run of 2). That step's input change was smaller than any step in this variant's calibration, and the polynomial, which is forced through zero, priced it as almost no change. 0.12.0 holds the change to the calibrated range (0.0283 to 0.2198) before evaluating the polynomial, so a change that small is priced at 0.144, the lowest the fit gives inside that range. With a floor that high, at the 0.17 default the gate can never skip two steps in a row. On an 864×1152 run the same effect had been skipping three early steps at every threshold down to 0.08.

> **Note on the v0.4.1 claim.** v0.4.1 reported 1.26× combined on this variant, decomposed as 1.16× gating × 1.09× compile-avoidance. The 1.26× combined number was honest within day-to-day noise, though it is above the 1.22× limit that 9 skipped steps of 50 allow (50/41); the subprocess-per-rep harness landed 1.23× (v0.6.0, above its 1.22× limit of 9 skipped steps in 50, 50/41) and 1.22× (v0.10.0), a difference of a few percent that sits inside any reasonable measurement band. The *decomposition* is what needed correcting. With subprocess isolation, gating is doing essentially all the work (1.18–1.22×) and compile-avoidance is at noise level (1.01×). The 1.09× stays withdrawn. The v0.4.1 bench script ran an untimed plain warmup first, then plain mflux, the no-gate wrapper and the gated wrapper in one Python process. Its record, `_artifacts/_bench_report_klein_base_4b_v0_4_1.json`, has medians of 254.0 s plain, 233.5 s no-gate and 201.2 s gated. When v0.6.0 gave each rep its own process (`_artifacts/v0.6.0_bench_klein_base_4b.json`), the medians were 236.2 s, 233.4 s and 191.8 s. The no-gate wrapper took the same time in both. So the change in the compile-avoidance ratio came from the plain run. The gated run was also slower in v0.4.1 (201.2 s against 191.8 s), and the records do not say why either run was slower. The v0.10.0 record (`_artifacts/v0.10.0_bench_klein_base_4b.json`, the 1.22×, mflux 0.18.0) has no commit or dirty field, no date and no MLX version, and its install-time version string names commit 90fca33. The v0.6.0 record (mflux 0.17.5) has no commit, dirty flag, date or MLX version, and its version string names no commit. The v0.4.1 record names no hardware, versions, date or commit at all. Both runs used 512×512, the red-apple prompt and 3 reps.
>
> (Corrected 2026-11-13: an earlier version of this note said the wrapper inherited a warm allocator. That explanation was wrong.)

Reproduce with `uv run python scripts/bench_speedup.py --variant klein-base-4b --three-way --reps 3 --report out.json`. Full report at `_artifacts/v0.12.0_bench_klein_base_4b.json` (v0.10.0's and v0.6.0's sit alongside it); regenerate side-by-side images with `scripts/bench_comparison.py`.

The 0.17 default was chosen because the polynomial R² is low (0.106) and the engagement window is narrow. The v0.4.0 sweep (before the 0.10 gate change) at 25 steps showed it:
- At 0.20 (package default) the image was visibly degraded.
- At **0.17** the image was indistinguishable from vanilla.
- Just above 0.17 (0.175) the quality fell off sharply.

No record of that sweep was committed, so its skip counts and SSIM scores are not given here. See `scripts/sweep_threshold_klein_base_4b.py` for the sweep methodology.

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

- **Default threshold is 0.17, not the package fallback 0.20.** The default comes from the variant's `DEFAULT_THRESH`.
- The CFG path's cosine ≥ 0.97 tolerance is documented in `tests/test_parity_flux2.py::_FLUX2_COSINE_GATE`. Compile-vs-eager dispatch noise compounds across steps — see the test module docstring.
- `flux2_klein_base_9b` cross-imports COEFFICIENTS from this variant — same architecture family, same calibration recipe.
