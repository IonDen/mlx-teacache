# z-image-base

Z-Image base (Tongyi-MAI) — non-distilled, 50-step CFG recipe. The first non-FLUX model with a TeaCache mini-kernel; unlike the FLUX variants, its gate is calibrated on a latent-dependent internal signal rather than a cheap modulation input.

## Construct via mflux

```python
from mflux.models.z_image.variants.z_image import ZImage
from mflux.models.common.config.model_config import ModelConfig

flux = ZImage(quantize=8, model_config=ModelConfig.z_image())
```

Z-Image base and Z-Image Turbo are the same `ZImage` class, distinguished by `model_config.aliases` (base: `["z-image", "zimage"]`; turbo: `["z-image-turbo", "zimage-turbo"]`). The detector matches base only; Turbo is not a registered variant and falls through to `IncompatibleModelError`.

## Recipe + defaults

- Default recipe: 50 steps, `guidance=4.0` (CFG), quantize=8
- Default `rel_l1_thresh`: **0.12** (per-variant default, set at the SSIM knee from the sweep)
- skip-window defaults: `skip_first_n_steps=1`, `skip_last_n_steps=1`

At the 512×512 red-apple bench recipe on M1 Max 32 GB (subprocess-per-rep, 3 reps, q8, mflux 0.18.0, v0.10.0 bench, 2026-08-15):

| Condition | Median wall-clock | Peak memory |
|---|---|---|
| vanilla | 227.4 s | 17.2 GB |
| wrapper, no gate (compile path bypass only) | 228.9 s | 11.5 GB |
| wrapper, gated (full TeaCache) | 174.2 s | 11.5 GB |

- **Combined speedup: 1.31×** (gating 1.31×, compile-avoidance 0.99×)
- The wall-clock win is entirely gating. `mx.compile`-path avoidance is not a tailwind on Z-Image — the no-gate wrapper runs at vanilla speed (228.9 s vs 227.4 s; the eager re-walk neither beats nor loses to mflux's compiled `_predict` on this model), so the whole 1.31× is step-skipping.
- Peak memory drops from 17.2 GB to 11.5 GB. This is the eager wrapper bypassing mflux's compiled `_predict`, not the skip path: the no-gate wrapper shows the same ~11.5 GB peak. Same effect documented on klein-base-9b.
- Skip count stable across reps: 15 of 48 active steps skipped at `rel_l1_thresh=0.12`, never two in a row (max consecutive-skip streak 1). SSIM 0.991 vs vanilla on this prompt.
- v0.7.0 reported 1.17× at this recipe (245.3 s → 209.4 s, `scripts/_bench_z_image_v0_7_0.json`) with a thermally confounded three-way split. The skip count and pattern are identical, so the gap is host state in that session, not the gate.

Reproduce with `uv run python scripts/bench_speedup.py --variant z-image --three-way --reps 3 --report out.json`. Full report at `_artifacts/v0.10.0_bench_z_image.json`.

The portrait row in [COMPARISON.md](../../COMPARISON.md) is a separate generation at 640×896 q8 (the shared comparison prompt): 1.33× warm, 14/48 skips, SSIM 0.957, peak 18.7 GB → 13.1 GB. The speedup is higher at 640×896 than at 512² because each skipped step saves more absolute compute, so the per-step gating overhead amortizes better.

## Threshold sweep

`scripts/sweep_threshold_z_image.py` sweeps `rel_l1_thresh` at the 512² recipe and records skip count + SSIM vs vanilla per threshold. SSIM holds ≥ 0.99 through 0.12 (15/48 skips, SSIM 0.991), then drops to ~0.974 at 0.15 and plateaus. 0.12 is the quality-first knee — the last threshold before the cliff.

## CFG (guidance > 1.0)

CFG runs through a per-branch gated path (`zimage_cfg_forward_with_gate`). The transformer runs twice per step (positive and negative caption); the gate decision is driven by the positive branch's Signal B and applied to both branches, each maintaining its own cached residual. CFG combine matches mflux's `z_image.py`: `noise + guidance * (noise - negative_noise)` (note this differs from the FLUX.2 form `neg + g*(pos - neg)`). The eager-Python wrapper diverges from vanilla mflux's compiled `_predict` by dispatch noise that compounds to cosine ≥ 0.99 at `rel_l1_thresh=0` — above the parity gate (`tests/test_parity_z_image.py`), not bit-exact.

## Coefficient provenance

Z-Image's adaLN modulation is timestep-only (content-independent), so there is no cheap caption-independent modulation input to gate on, the way the FLUX variants use. The calibration tapped two candidate latent-dependent signals and fit a degree-4 origin-constrained polynomial for each:

- **Signal A** — noise-refiner output rel-L1 (image-only, caption-independent). Rejected: R² = 0.069. Its consecutive-step rel-L1 sits in a compressed [0.01, 0.12] range that cannot track the body change.
- **Signal B** — first-main-layer residual rel-L1 (caption-dependent). **Selected**: R² = 0.400, held-out R² = 0.179.

Signal B's R² is in line with the shipped FLUX.2 variants (klein-base-4b ships at 0.106, klein-9b at 0.471); per-step fit R² is not the arbiter of caching efficacy — the rescale-polynomial plus the accumulator threshold is, and the sweep confirmed a usable skip-vs-SSIM knee.

- `scripts/calibrate_z_image.py --fit-mode origin`
- 10 prompts (7 fit / 3 held-out) × 50 steps × seed=42 on M1 Max 32 GB, q8, 512×512, guidance=4.0 (CFG)
- A first-step self-check asserts the capturing re-walk's CFG noise matches `transformer(...)` at cosine ≥ 0.999 (faithful-port guard)
- Stored verbatim in `src/mlx_teacache/variants/z_image_base/config.py::COEFFICIENTS`

Full report: `scripts/_calibration_z_image.json`.

## Cache contract

The mini-kernel re-walks `ZImageTransformer.__call__`: patchify → noise refiner → caption embed → context refiner → concat (`unified_in`) → 30 main layers (`main_out`) → final layer → unpatchify → negate. The gate signal is the first-main-layer residual; on a compute step it caches `cached_residual = main_out - unified_in` per CFG branch, and on a skip it reconstructs `main_out = unified_in + cached_residual` from the current step's freshly computed prelude. A skip runs the prelude plus one of the 30 main layers (to produce the gate signal), then reuses the cached residual for the other 29.

## License

[`Apache-2.0`](https://huggingface.co/Tongyi-MAI/Z-Image). Permissive, not gated — no acceptance flow required (unlike the FLUX Non-Commercial variants).

## Quirks

- **Default threshold is 0.12, not the package fallback 0.20.** Set via `Provenance.default_thresh` in the variant's `_PROVENANCE`, resolved at `apply_teacache` time.
- **q8, not q4.** Z-Image's pinned recipe is 8-bit. The bench and comparison harnesses carry a per-variant quantize for this reason.
- **Self-contained mini-kernel.** The variant defines its own internal handle and forward and imports no sibling variant — only the model-agnostic `_kernel/`, the public handle, and the shared mflux lifecycle helpers.
- **Repeatability across a compiled/eager switch needs one line on MLX 0.32.** mflux builds Z-Image's rotary tables (`transformer.rope_embedder.freqs_cis`) as lazy arrays it never evaluates. On MLX 0.32 a compiled vanilla `_predict` traces that pending graph into the compiled function, while the first eager pass through the transformer (TeaCache's proxy, or any direct call) materialises the tables with the eager kernels, and every later compiled run then captures those values. Measured on an M1 Max with mflux 0.19.1 and MLX 0.32.2: a bare `mx.eval` of the tables, with no TeaCache in the process, moves the next vanilla latent by max-abs 6.25e-2 (cosine 0.999977) on the 8-step parity recipe; on MLX 0.31.2 nothing moves. `restore()` puts back every attribute it changed; the materialised tables are what persists, and any eager use of the transformer produces them. If you compare vanilla runs before and after using TeaCache and want them bit-identical, run `mx.eval(*flux.transformer.rope_embedder.freqs_cis)` once after loading the model; the parity tests do exactly that.
- The comparison harness clears the MLX buffer cache between reps for this variant (`clear_cache_between_reps=True`); a single 640×896 q8 generation peaks ~18.7 GB, but the cache accumulates across reps in one process and OOMs the Metal command buffer without it.
