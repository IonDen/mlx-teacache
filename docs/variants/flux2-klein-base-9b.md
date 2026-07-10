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

At the canonical 50-step CFG recipe on M1 Max 32 GB (subprocess-per-rep, 3 reps, bf16, q4, v0.6.0 bench harness):

| Condition | Median wall-clock | Peak memory |
|---|---|---|
| vanilla | 517.6 s | ~22 GB |
| wrapper, no gate (compile-avoidance only) | 509.3 s | ~10 GB |
| wrapper, gated (full TeaCache) | 380.6 s | ~10 GB |

- **Combined speedup: 1.36×**
- **Gating contribution (v0.4.1 effect): 1.34×**
- **`mx.compile`-path avoidance (v0.4 effect): 1.02×** — small on this recipe/chip
- Skip count stable across reps: 13 of 48 active steps skipped at `rel_l1_thresh=0.17`
- SSIM 0.986 vs vanilla (carried over from v0.5.0 validation; visually equivalent)

> **Correction to v0.5.0.** v0.5.0 reported 2.68× combined on this variant. That measurement was inflated by same-process MLX state leakage in the v0.5.x bench harness: vanilla ran cold while the wrapper inherited warm MLX allocator state from it, so the wall-clock difference conflated the variant difference with the cold-vs-warm gap. v0.6.0's subprocess-per-rep harness makes every condition cold and exposes the honest 1.36×.

Reproduce with `uv run python scripts/bench_speedup.py --variant klein-base-9b --three-way --reps 3 --report out.json`. Full report at `_artifacts/v0.6.0_bench_klein_base_9b.json`; regenerate side-by-side images with `scripts/bench_comparison.py`.

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
- The 1.36× headline number is reproducible via `scripts/bench_speedup.py --variant klein-base-9b --three-way --reps 3`.

## Memory guardrails

On 32 GB Apple Silicon machines running the canonical 50-step + g=4.0 recipe, peak wired memory has crossed the system limit twice (2026-05-17 jetsam, 2026-05-19 kernel watchdog panic). The mitigations now in place:

1. `tests/conftest.py` installs a session-level `mx.set_wired_limit(20 GB)` + `mx.set_memory_limit(22 GB)` for the test suite.
2. `scripts/bench_speedup.py` workers apply the same caps before model load, derived from this variant's `memory_cap_hint_gb`.

If you allocate over the wired cap, MLX raises an exception rather than the kernel panicking. See [ml-explore/mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883) for upstream context on why the wired limit (not the soft memory limit) is what prevents the panic.
