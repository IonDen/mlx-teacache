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

At the canonical 50-step CFG recipe on M1 Max 32 GB (subprocess-per-rep, 3 reps, bf16, q4, mflux 0.18.0, v0.10.0 bench, 2026-08-15):

| Condition | Median wall-clock | Peak memory |
|---|---|---|
| vanilla | 520.6 s | ~22 GB |
| wrapper, no gate (compile-avoidance only) | 509.8 s | ~9.5 GB |
| wrapper, gated (full TeaCache) | 379.1 s | ~9.5 GB |

- **Combined speedup: 1.37×** (v0.6.0 measured 1.36× at the same recipe; unchanged to within noise)
- **Gating contribution (v0.4.1 effect): 1.34×**
- **`mx.compile`-path avoidance (v0.4 effect): 1.02×** — small on this recipe/chip
- Skip count stable across reps: 13 of 48 active steps skipped at `rel_l1_thresh=0.17`, never two in a row (max consecutive-skip streak 1)
- SSIM 0.986 vs vanilla (carried over from v0.5.0 validation; visually equivalent)

> **Correction to v0.5.0.** v0.5.0 reported 2.68× combined on this variant. That measurement was inflated by same-process MLX state leakage in the v0.5.x bench harness: vanilla ran cold while the wrapper inherited warm MLX allocator state from it, so the wall-clock difference conflated the variant difference with the cold-vs-warm gap. v0.6.0's subprocess-per-rep harness makes every condition cold and exposes the honest 1.36×; the v0.10.0 re-measurement lands at 1.37×.

Reproduce with `uv run python scripts/bench_speedup.py --variant klein-base-9b --three-way --reps 3 --report out.json`. Full report at `_artifacts/v0.10.0_bench_klein_base_9b.json` (v0.6.0's at `_artifacts/v0.6.0_bench_klein_base_9b.json`); regenerate side-by-side images with `scripts/bench_comparison.py`.

## Coefficient provenance — intentional reuse

Coefficients are **cross-imported from `flux2_klein_base_4b`** (object-identity-equal — `BASE_9B is BASE_4B`). The reuse is justified by:
- Same architecture family (FLUX.2 Klein, non-distilled).
- Same calibration recipe (25-step, guidance=1.0, origin-constrained polyfit).
- Validated empirically before each release: `tests/variants/flux2_klein_base_9b/test_shared_coefficients.py` enforces the identity invariant.

If a future release recalibrates 9B-specific coefficients, that test will need to flip from `assert BASE_9B is BASE_4B` to a value-equal check (or move 9B to its own literal tuple).

## License

[`FLUX Non-Commercial`](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) — accept on the Hugging Face model page before downloading. **For commercial use, see Black Forest Labs licensing.**

## Quirks

- **Memory cap hint = 24 GB.** `META["memory_cap_hint_gb"]` is the advisory soft cap the heavy workers pass to `scripts/_mlx_caps.py::install_caps` before model load. `install_caps` sets three limits: a device-clamped wired cap (the request, capped at 0.85× the machine's recommended working set, so it can never exceed the system wired limit and still works on a 16/24 GB Mac), the 24 GB advisory soft limit, and a bounded MLX cache pool. On a 32 GB M1 Max the clamp puts the wired cap near 21 GB regardless of the request. Lower the hint if you have less than 32 GB or are competing with other memory-heavy processes; see [Memory guardrails](#memory-guardrails) below.
- Default threshold 0.17 (cross-imported from base-4b), not the package fallback 0.20.
- The 1.37× headline number is reproducible via `scripts/bench_speedup.py --variant klein-base-9b --three-way --reps 3`.

## Memory guardrails

On 32 GB Apple Silicon running the canonical 50-step + g=4.0 recipe, peak memory has crossed the system limit twice (2026-05-17 jetsam, 2026-05-19 kernel watchdog panic). The current mitigations:

1. Every heavy worker calls `scripts/_mlx_caps.py::install_caps` before model load — a device-clamped wired cap, an advisory soft cap (this variant's `memory_cap_hint_gb`, 24 GB), and a bounded MLX cache pool (dropped buffers pool there instead of returning to the OS, and MLX's default pool sits near physical RAM).
2. The same worker arms `scripts/_mlx_watchdog.py::arm_mlx_watchdog`, a daemon thread that aborts the run — writing an artifact that says why — the moment `active + cache` memory exceeds physical memory minus a headroom (4 GiB by default).
3. `tests/conftest.py` installs the same device-derived caps and bounds the cache pool for the test session.

The wired cap is the only hard ceiling, but it bounds only non-pageable (wired) allocations — it does not stop a run from over-allocating pageable memory, which pages instead of failing, and a sustained paging storm is what panicked the machine in 2026-05. So the watchdog, not the wired cap, is what makes a heavy run on this recipe safe. See [ml-explore/mlx-lm#883](https://github.com/ml-explore/mlx-lm/issues/883) for upstream context on the wired-limit mechanism.
