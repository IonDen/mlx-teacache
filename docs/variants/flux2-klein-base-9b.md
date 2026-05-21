# flux2-klein-base-9b

FLUX.2 Klein base 9B — non-distilled, 50-step canonical CFG recipe. The headline FLUX.2 result, shipped in v0.5.0.

## Construct via mflux

```python
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.common.config.model_config import ModelConfig

flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_9b())
```

## Recipe + defaults

- Default recipe: 50 steps, `guidance=4.0` (canonical CFG, matches upstream)
- Alternative recipe `low_step`: 25 steps, `guidance=1.0`
- Default `rel_l1_thresh`: **0.17** (cross-import from `flux2_klein_base_4b`)
- skip-window defaults: `skip_first_n_steps=1`, `skip_last_n_steps=1`
- `memory_cap_hint_gb: 24` — 32 GB unified memory headroom

At the canonical 50-step CFG recipe on M1 Max 32 GB (subprocess-isolated cold rep, bf16, q4):
- Vanilla: 2744 s
- Wrapper: 1025 s
- **2.68× wall-clock speedup**, 12 of 48 active steps skipped at `rel_l1_thresh=0.17`
- SSIM 0.986 vs vanilla — visually equivalent

The 2.68× combines step-skipping with `mx.compile`-path avoidance; clean attribution between the two mechanisms is deferred to a future release.

See `_artifacts/validation_klein_base_9b.json` for the full evidence and `_artifacts/validation_klein_base_9b_images/` for side-by-side images.

## Coefficient provenance — intentional reuse

Coefficients are **cross-imported from `flux2_klein_base_4b`** (object-identity-equal — `BASE_9B is BASE_4B`). The reuse is justified by:
- Same architecture family (FLUX.2 Klein, non-distilled).
- Same calibration recipe (25-step, guidance=1.0, origin-constrained polyfit).
- Validated empirically before each release: `tests/variants/flux2_klein_base_9b/test_shared_coefficients.py` enforces the identity invariant.

If a future release recalibrates 9B-specific coefficients, that test will need to flip from `assert BASE_9B is BASE_4B` to a value-equal check (or move 9B to its own literal tuple).

## License

[`FLUX Non-Commercial`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) — accept on the Hugging Face model page before downloading. **For commercial use, see Black Forest Labs licensing.**

## Quirks

- **Memory cap hint = 24 GB.** The bench harness reads `META["memory_cap_hint_gb"]` and sets `mx.set_wired_limit((cap - 2) * 1024**3) = 22 GB` plus `mx.set_memory_limit(cap * 1024**3) = 24 GB` in each worker before model load. Lower these if you have less than 32 GB or are competing with other memory-heavy processes; see [Memory guardrails](#memory-guardrails) below.
- Default threshold 0.17 (cross-imported from base-4b), not the package fallback 0.20.
- The 2.68× headline number is reproducible via `scripts/bench_speedup.py --variant klein-base-9b --three-way --reps 3`.

## Memory guardrails

On 32 GB Apple Silicon machines running the canonical 50-step + g=4.0 recipe, peak wired memory has crossed the system limit twice (2026-05-17 jetsam, 2026-05-19 kernel watchdog panic). The mitigations now in place:

1. `tests/conftest.py` installs a session-level `mx.set_wired_limit(20 GB)` + `mx.set_memory_limit(22 GB)` for the test suite.
2. `scripts/bench_speedup.py` workers apply the same caps before model load, derived from this variant's `memory_cap_hint_gb`.

If you allocate over the wired cap, MLX raises an exception rather than the kernel panicking. See [`CLAUDE.md` "Memory guardrails for heavy generations on 32 GB"](../../docs/superpowers/standards/) and [ml-explore/mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883) for upstream context.
