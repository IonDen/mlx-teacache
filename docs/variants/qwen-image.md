# qwen-image

Qwen-Image base (Alibaba) — a ~20B dual-stream MMDiT, non-distilled, run at a 20-step CFG recipe. It's the first variant on the FLUX.1 proxy-transformer pattern that also runs true two-pass CFG, and at the package default it gets the largest measured speedup of any supported model (1.41× warm-median).

## Construct via mflux

```python
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.common.config.model_config import ModelConfig

flux = QwenImage(quantize=4, model_config=ModelConfig.qwen_image())
```

The detector matches `model_config.aliases` containing `"qwen-image"` or `"qwen"`. Qwen-Image-Edit (aliases `"qwen-image-edit"` / `"qwen-edit"` / `"qwen-edit-plus"` / `"qwen-edit-2509"`) is a separate pipeline, not a registered variant; it falls through to `IncompatibleModelError`.

## Recipe + defaults

- Default recipe: 20 steps, `guidance=4.0` (CFG), quantize=4
- Default `rel_l1_thresh`: **0.25** (per-variant default, set at the SSIM knee from the sweep)
- skip-window defaults: `skip_first_n_steps=1`, `skip_last_n_steps=1`

At the 512×512 shared portrait recipe on M1 Max 32 GB (`bench_comparison.py`, subprocess-per-condition, 3 reps, q4):

| Condition | Warm-median wall-clock | Peak memory |
|---|---|---|
| vanilla | 117.27 s | 27.61 GB |
| wrapper (full TeaCache) | 83.36 s | 27.78 GB |

- **Warm speedup: 1.41×** (cold 1.39×), 6 of 18 active steps skipped at `rel_l1_thresh=0.25`, stable across reps.
- The win is entirely step-skipping. Qwen-Image's `_predict` is **not** `mx.compile`-wrapped in mflux (the FLUX.2/Z-Image compile-avoidance effect does not exist here), so there is no separate compile-path tailwind to attribute — the speedup comes from reusing the cached transformer-body residual on skipped steps.
- Peak memory is ~27.6 GB, near the 32 GB ceiling. The wired cap (device-derived, strictly below `max_recommended_working_set_size`) bounds the non-pageable allocation; the remainder is pageable. See [Quirks](#quirks) for the 768→512 resolution fallback.

Reproduce with `uv run python scripts/bench_comparison.py --only qwen-image`. Full report at `_artifacts/comparison_report.json`; images under `_artifacts/comparison/qwen-image/`.

There is no `bench_speedup.py` three-way row for Qwen-Image: with no `mx.compile` path to bypass, the vanilla / no-gate / gated decomposition collapses to "all step-skipping", so the subprocess-per-condition comparison bench above is the committed source for the number.

## Threshold sweep

`scripts/sweep_threshold_qwen.py` sweeps `rel_l1_thresh` at the 512² recipe and records skip count + SSIM vs vanilla per threshold. SSIM holds ≥ 0.99 through 0.30, then drops off a cliff to 0.908 at 0.40. The skip count plateaus at 6 of 18 active steps by 0.25 (SSIM 0.9918), so 0.25 is the quality-first default: it captures the full visually-lossless skip benefit with margin from the cliff, and a higher threshold buys no extra skips on this recipe. Full sweep: `tests/_artifacts/sweep_qwen/results_qwen.json`.

## CFG (guidance > 1.0)

Qwen-Image always runs CFG: `QwenImage.generate_image` calls the transformer twice per step (positive then negative caption) and combines the two predictions **outside** the transformer. So the proxy forward fires once per branch, and a small state machine threads one shared gate decision and two cached residuals across the pair. The decision is computed on the positive call's gate signal and reused on the negative call; each branch keeps its own cached residual. The gate signal is branch-independent (it depends on the latents and timestep, not the caption), so sharing one decision across both branches is exact, not an approximation. Threshold-0 parity vs vanilla is cosine ≥ 0.99 (`tests/test_parity_qwen.py`), not bit-exact — the eager re-walk diverges from mflux by Metal-dispatch noise.

## Coefficient provenance

Qwen-Image is FLUX-shaped: each block applies a 6×dim adaLN modulation, so the FLUX-canonical TeaCache signal is available — the modulated block-0 image input. The calibration captured two candidate signals and fit a degree-4 origin-constrained polynomial for each:

- **Signal A** — modulated block-0 image input rel-L1 (the FLUX-canonical signal; caption-independent). **Selected**: R² = 0.9464, held-out 0.9439.
- **Signal B** — first-block image-stream residual rel-L1 (caption-dependent). R² = 0.9516, held-out 0.9553.

Signal B's R² is marginally higher, but Signal A is the better choice on two counts: it's caption-independent (the shared CFG decision is exact rather than approximate), and it's cheaper on a skip step (Signal A comes from the prelude; Signal B would require running block 0). The fit quality here (~0.95) is far above Z-Image's 0.40 and the FLUX.2 family's 0.11–0.47, because Qwen's modulation structure matches the FLUX models the polynomial form was designed for.

- `scripts/calibrate_qwen.py --fit-mode origin`
- 10 prompts (7 fit / 3 held-out) × 20 steps × seed=42 on M1 Max 32 GB, q4, 512×512, guidance=4.0 (CFG)
- A first-step self-check asserts the capturing re-walk's per-branch noise matches the unwrapped `QwenTransformer.__call__` at cosine ≥ 0.999 (faithful-port guard)
- Stored verbatim in `src/mlx_teacache/variants/qwen_image/config.py::COEFFICIENTS`

Full report: `scripts/_calibration_qwen.json`.

## Cache contract

The integration replaces `flux.transformer` with a proxy whose forward re-walks `QwenTransformer.__call__` in four parts: a prelude (`img_in` + the timestep-only text embedding), the gate signal (the modulated block-0 image input), the body (the 60 dual-stream blocks over the image stream), and the tail (`norm_out` + `proj_out`). On a compute step it caches `cached_residual = body_out − img_in(latents)` per CFG branch; on a skip it reconstructs `body_out = img_in(latents) + cached_residual` from the current step's freshly computed prelude, then runs the tail. A skip runs the prelude, the gate signal, and the tail — but none of the 60 blocks, and not the rotary-embedding computation (which only the blocks need).

## License

[`Apache-2.0`](https://huggingface.co/Qwen/Qwen-Image). Permissive, not gated — no acceptance flow required.

## Quirks

- **Default threshold is 0.25, not the package fallback 0.20.** Set via `Provenance.default_thresh` in the variant's `_PROVENANCE`, resolved at `apply_teacache` time.
- **512×512, a memory fallback from the nominal 768×768.** At 768² the model peaks 28.3 GB; at 512² it still peaks 27.6 GB — the peak is weights-dominated (20B at q4 plus the Qwen2.5-VL text encoder), so dropping resolution barely moves it. 512² is the recipe that stays survivable on a 32 GB M1 Max. The calibration, sweep, parity, and comparison all use 512²; the COMPARISON portrait row keeps the shared prompt and seed and changes only the resolution.
- **Proxy-transformer pattern, not the `_predict` closure.** Qwen-Image has no `_predict` factory and no `mx.compile`, so the variant proxies `flux.transformer` (like FLUX.1) rather than replacing `_predict` (like FLUX.2 / Z-Image). As with FLUX.1, calling `flux.parameters()` at the parent level can miss transformer parameters while the wrapper is active — use `flux.transformer.parameters()` or `handle.restore()` first.
- **Self-contained.** The variant defines its own internal handle, branch-pairing state machine, and forward, importing no sibling variant — only the model-agnostic `_kernel/`, the public handle, and the shared mflux lifecycle helpers.
- The comparison harness clears the MLX buffer cache between reps for this variant (`clear_cache_between_reps=True`) and sets a soft memory limit above the peak (`soft_cap_gb`); without the cache clear, accumulation across reps in one process would OOM the second rep.
