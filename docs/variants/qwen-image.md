# qwen-image

Qwen-Image base (Alibaba) — a ~20B dual-stream MMDiT, non-distilled, run at the official 50-step CFG recipe. It's the first variant on the FLUX.1 proxy-transformer pattern that also runs true two-pass CFG, and TeaCache skips about half its denoising steps at visually-lossless quality.

## Construct via mflux

```python
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.common.config.model_config import ModelConfig

flux = QwenImage(quantize=4, model_config=ModelConfig.qwen_image())
```

The detector matches `model_config.aliases` containing `"qwen-image"` or `"qwen"`. Qwen-Image-Edit (aliases `"qwen-image-edit"` / `"qwen-edit"` / `"qwen-edit-plus"` / `"qwen-edit-2509"`) is a separate pipeline, not a registered variant; it falls through to `IncompatibleModelError`.

## Recipe + defaults

- Default recipe: 50 steps, `guidance=4.0` (CFG), quantize=4, 768×768
- Default `rel_l1_thresh`: **0.30** (per-variant default, set from the threshold sweep)
- skip-window defaults: `skip_first_n_steps=1`, `skip_last_n_steps=1`

At the 768×768 shared portrait recipe on M1 Max 32 GB (`bench_comparison.py`, subprocess-per-condition, 3 reps):

| Condition | Warm-median wall-clock | Peak memory |
|---|---|---|
| vanilla | 1207.2 s | 30.4 GB |
| wrapper (full TeaCache) | 695.1 s | 30.8 GB |

- **Warm speedup: 1.74×** (1.63× cold), with 25 of 48 active steps skipped at `rel_l1_thresh=0.30`.
- The win is step-skipping. Qwen-Image's `_predict` is **not** `mx.compile`-wrapped in mflux (the FLUX.2/Z-Image compile-avoidance effect does not exist here), so there is no separate compile-path tailwind to attribute — the speedup comes from reusing the cached transformer-body residual on skipped steps, each skip avoiding both CFG branches' 60-block bodies.

Reproduce with `uv run python scripts/bench_comparison.py --only qwen-image`. Full report at `_artifacts/comparison_report.json`; images under `_artifacts/comparison/qwen-image/`.

## Image quality on consumer memory

Stock uniform 4-bit quantization over-quantizes Qwen-Image's quantization-sensitive layers and produces a grainy, low-detail skin texture on a 32 GB Mac — a Qwen + q4 limitation, independent of TeaCache (the wrapper faithfully reproduces whatever the base model generates). The comparison portraits here were rendered with a mixed-precision build that keeps the first/last transformer blocks at 8-bit and the embeddings + final projection at bf16, which clears the artifact and still fits 32 GB (~30.4 GB peak). mlx-teacache stays quantization-agnostic; this is a model-construction choice. To reproduce the showcase quality, install the predicate before building the model:

```python
import mlx.nn as nn
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

_PROTECT = set(range(6)) | set(range(54, 60))           # first 6 + last 6 of 60 blocks -> q8
_BF16 = ("img_in", "txt_in", "time_text_embed", "proj_out", "norm_out")  # -> bf16

def _mixed(path, module):
    if not hasattr(module, "to_quantized"):
        return False
    if any(path == p or path.startswith(p + ".") for p in _BF16):
        return False                                    # keep bf16
    if path.startswith("transformer_blocks."):
        idx = int(path.split(".")[1])
        if idx in _PROTECT:
            return {"group_size": 64, "bits": 8}        # q8
    return True                                         # default q4

QwenWeightDefinition.quantization_predicate = staticmethod(_mixed)
flux = QwenImage(quantize=4, model_config=ModelConfig.qwen_image())
```

The gate signal (Signal A) is unaffected enough by this that the shipped coefficients — calibrated on stock q4 — transfer cleanly to the mixed-precision build (the threshold sweep confirmed a sensible skip ramp at high SSIM on it).

## Threshold sweep

`scripts/sweep_threshold_qwen.py` sweeps `rel_l1_thresh` at the 768×768/50-step recipe and records skip count + SSIM vs vanilla per threshold. SSIM degrades **gracefully — no cliff**: 0.9951 at 0.20, 0.9883 at 0.25, 0.9873 at 0.30, 0.9809 at 0.40, 0.9783 at 0.50. The default 0.30 takes ~50% of the active steps (24 of 48) at SSIM 0.987 — visually identical to vanilla — while leaving margin. The sweep's single-rep wall-clock is thermal noise (subprocess-per-threshold, cold each); the headline speedup comes from the multi-rep bench.

## CFG (guidance > 1.0)

Qwen-Image always runs CFG: `QwenImage.generate_image` calls the transformer twice per step (positive then negative caption) and combines the two predictions **outside** the transformer. So the proxy forward fires once per branch, and a small state machine threads one shared gate decision and two cached residuals across the pair. The decision is computed on the positive call's gate signal and reused on the negative call; each branch keeps its own cached residual. The gate signal is branch-independent (it depends on the latents and timestep, not the caption), so sharing one decision across both branches is exact, not an approximation. Threshold-0 parity vs vanilla is cosine ≥ 0.99 (`tests/test_parity_qwen.py`), not bit-exact — the eager re-walk diverges from mflux by Metal-dispatch noise.

## Coefficient provenance

Qwen-Image is FLUX-shaped: each block applies a 6×dim adaLN modulation, so the FLUX-canonical TeaCache signal is available — the modulated block-0 image input. The calibration captured two candidate signals and fit a degree-4 origin-constrained polynomial for each:

- **Signal A** — modulated block-0 image input rel-L1 (the FLUX-canonical signal; caption-independent). **Selected**: R² = 0.849, held-out 0.845.
- **Signal B** — first-block image-stream residual rel-L1 (caption-dependent). R² = 0.881, held-out 0.890.

Signal B's R² is marginally higher, but Signal A is the better choice on two counts: it's caption-independent (the shared CFG decision is exact rather than approximate), and it's cheaper on a skip step (Signal A comes from the prelude; Signal B would require running block 0). The R² (~0.85) is lower than a 512×512/20-step fit would give because the heavier 768×768/50-step recipe samples a finer, noisier per-step rel-L1 distribution — still well above Z-Image's 0.40 and the FLUX.2 family's 0.11–0.47, and the sweep confirms it skips strongly at high SSIM.

- `scripts/calibrate_qwen.py --fit-mode origin` (chunked: one worker subprocess per prompt, resumable)
- 10 prompts (7 fit / 3 held-out) × 50 steps × seed=42 on M1 Max 32 GB, q4, 768×768, guidance=4.0 (CFG)
- A first-step self-check asserts the capturing re-walk's per-branch noise matches the unwrapped `QwenTransformer.__call__` at cosine ≥ 0.999 (faithful-port guard)
- Stored verbatim in `src/mlx_teacache/variants/qwen_image/config.py::COEFFICIENTS`

Full report: `scripts/_calibration_qwen.json`.

## Cache contract

The integration replaces `flux.transformer` with a proxy whose forward re-walks `QwenTransformer.__call__` in four parts: a prelude (`img_in` + the timestep-only text embedding), the gate signal (the modulated block-0 image input), the body (the 60 dual-stream blocks over the image stream), and the tail (`norm_out` + `proj_out`). On a compute step it caches `cached_residual = body_out − img_in(latents)` per CFG branch; on a skip it reconstructs `body_out = img_in(latents) + cached_residual` from the current step's freshly computed prelude, then runs the tail. A skip runs the prelude, the gate signal, and the tail — but none of the 60 blocks, and not the rotary-embedding computation (which only the blocks need).

## License

[`Apache-2.0`](https://huggingface.co/Qwen/Qwen-Image). Permissive, not gated — no acceptance flow required.

## Quirks

- **Default threshold is 0.30, not the package fallback 0.20.** Set via `Provenance.default_thresh` in the variant's `_PROVENANCE`, resolved at `apply_teacache` time.
- **768×768, the official Qwen recipe.** The 20B model peaks ~26 GB at stock q4 on the 768×768 bench recipe and ~30.4 GB at the mixed-precision showcase build, both above the ~25 GB recommended working set of a 32 GB M1 Max. They complete only while host memory pressure stays low: the wired cap bounds only the non-pageable allocation, and the pageable excess pages rather than failing, which under sustained pressure can stall the run or panic the machine instead of raising a clean error. Treat 32 GB as the floor, confirm swap is near empty before a run, and expect the showcase build to be the tighter of the two. 1024×1024 needs tiled VAE decode to fit; 768×768 is the comfortable default.
- **Proxy-transformer pattern, not the `_predict` closure.** Qwen-Image has no `_predict` factory and no `mx.compile`, so the variant proxies `flux.transformer` (like FLUX.1) rather than replacing `_predict` (like FLUX.2 / Z-Image). As with FLUX.1, calling `flux.parameters()` at the parent level can miss transformer parameters while the wrapper is active — use `flux.transformer.parameters()` or `handle.restore()` first.
- **Self-contained.** The variant defines its own internal handle, branch-pairing state machine, and forward, importing no sibling variant — only the model-agnostic `_kernel/`, the public handle, and the shared mflux lifecycle helpers.
- The comparison harness clears the MLX buffer cache between reps for this variant and sets a soft memory limit above the peak; without the cache clear, accumulation across reps in one process would OOM the second rep.
