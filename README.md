# mlx-teacache

[![PyPI version](https://img.shields.io/pypi/v/mlx-teacache.svg)](https://pypi.org/project/mlx-teacache/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlx-teacache.svg)](https://pypi.org/project/mlx-teacache/)
[![License: Apache 2.0](https://img.shields.io/pypi/l/mlx-teacache.svg)](https://github.com/IonDen/mlx-teacache/blob/main/LICENSE)
[![CI](https://github.com/IonDen/mlx-teacache/actions/workflows/ci.yml/badge.svg)](https://github.com/IonDen/mlx-teacache/actions/workflows/ci.yml)

**TeaCache step-skipping for FLUX diffusion on Apple Silicon, in pure MLX.**

`mlx-teacache` is the first MLX port of TeaCache, a training-free inference optimization that skips 20-50% of denoising steps in FLUX-family diffusion models by predicting which steps add little to the final image. On FLUX.1-dev / 25 steps at the default threshold we measure 1.48× wall-clock speedup with output that stays visually equivalent (SSIM ≥ 0.80 across a 5-prompt suite, ≥ 0.90 on the PR-gate prompt).

## What it does

Diffusion models run the same big transformer 20-50 times in a loop. Between consecutive steps the output changes very little, and TeaCache uses a tiny polynomial fit to predict which steps can reuse the previous step's output. On M1 Max with FLUX.1-dev at 25 steps the default threshold (`rel_l1_thresh=0.20`) skips 6 of 25 steps and produces a 1.48× speedup.

```python
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.common.config.model_config import ModelConfig
from mlx_teacache import apply_teacache

flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())
with apply_teacache(flux):  # default rel_l1_thresh=0.20
    flux.generate_image(prompt="a red apple", seed=42, num_inference_steps=25)
```

## Install

```bash
pip install "mlx-teacache[mflux]"
# or with uv:
uv add "mlx-teacache[mflux]"
```

Requires Python ≥ 3.11 and Apple Silicon. The `[mflux]` extra pulls in `mflux>=0.17,<0.18`.

```bash
pip install "mlx-teacache==0.2.0[mflux]"  # pin for reproducibility
```

## Quick start

FLUX.1 dev:

```python
from mflux.models.flux.variants.txt2img.flux import Flux1
from mlx_teacache import apply_teacache

flux = Flux1.from_name("dev", quantize=4)
with apply_teacache(flux) as handle:  # default rel_l1_thresh=0.20
    flux.generate_image(prompt="...", seed=42, num_inference_steps=25, guidance=3.5)
    print(f"Speedup: {handle.stats.speedup_estimate:.2f}×")
```

## img2img

mlx-teacache supports mflux's image-to-image generation starting with v0.2.0. Pass `image_path` and `image_strength > 0` to `flux.generate_image()` with TeaCache active:

```python
from mlx_teacache import apply_teacache

with apply_teacache(flux):
    flux.generate_image(
        prompt="a red apple on a wooden table",
        image_path="/path/to/init.png",
        image_strength=0.7,
        num_inference_steps=25,
        seed=42,
        height=512,
        width=512,
    )
```

Caching engages on the active denoising window only, which mflux computes as `num_inference_steps - init_time_step`. So `image_strength=0.7` with `num_inference_steps=25` gives 8 predict calls available for caching. At `image_strength=1.0`, mflux skips denoising entirely and runs only VAE reconstruction, so TeaCache becomes a no-op.

Txt2img and img2img use the same polynomial coefficients. Image-quality is verified by SSIM gates over a fixed init-image suite in `tests/test_image_quality_*.py`.

## Threshold guide

Measured on M1 Max 32GB, FLUX.1-dev @ 25 steps, bf16, `seed=42`, `guidance=3.5`, red-apple prompt:

| `rel_l1_thresh` | Skipped steps | Speedup | SSIM vs vanilla | Recommended use |
|---|---|---|---|---|
| 0.10 | 0 / 25 | 1.07× | 1.0000 | Cache never engages |
| 0.15 | 0 / 25 | 1.13× | 1.0000 | Cache never engages |
| **0.20 (default)** | **6 / 25** | **1.48×** | **≥ 0.80 (5-prompt suite)** | **Visually-lossless sweet spot** |
| 0.25 | 11 / 25 | 1.96× | 0.57-0.93 | Visible style changes on text/synthetic prompts |

0.20 was picked after side-by-side visual comparison. At 0.25, text prompts that vanilla renders as neon tubes can come out as dot-matrix. At 0.20, the output is indistinguishable from vanilla and the cache still skips around 25% of steps. SSIM is conservative on high-frequency-detail prompts like text and synthetic patterns, which is why the suite floor (0.80) is lower than the PR-gate floor (0.90) on the red-apple prompt.

## Supported models

| Variant id | mflux class + config | Coefficient source |
|---|---|---|
| `flux1-dev` | `Flux1(model_config=ModelConfig.dev())` | upstream ali-vilab/TeaCache |
| `flux1-schnell` | `Flux1(model_config=ModelConfig.schnell())` | upstream (shared with dev) |
| `flux2-klein-4b` | `Flux2Klein(model_config=ModelConfig.flux2_klein_4b())` | in-repo (see `docs/calibration.md`) |
| `flux2-klein-9b`¹ | `Flux2Klein(model_config=ModelConfig.flux2_klein_9b())` | in-repo (see [`docs/calibration.md`](docs/calibration.md)) — see [License obligations](#license-obligations) |

¹ `flux2-klein-9b` coefficients are calibrated and validated at `num_inference_steps=8`. The mflux/BFL default of 4 steps leaves only a single skip opportunity per generation (`eligible=2`, `possible_skips=1`); TeaCache will still run but the wall-clock benefit at 4 steps is bounded by that single skip. For meaningful speedup pass `num_inference_steps=8` (or higher).

## Combining with mlx-taef

```python
from mlx_taef.integrations.mflux import LivePreviewCallback
from mlx_teacache import apply_teacache

preview = LivePreviewCallback(variant="taef2", every=5, save_to="preview.png",
                              latent_height=32, latent_width=32)
flux.callbacks.register(preview)

with apply_teacache(flux):  # default rel_l1_thresh=0.20
    flux.generate_image(prompt="...", seed=42, num_inference_steps=25)
```

## Inspecting stats

```python
handle = apply_teacache(flux)
flux.generate_image(prompt="apple", seed=42, num_inference_steps=25)
print(handle.stats.computed_count, handle.stats.skipped_count)
print(handle.stats.last_generation.decisions[5])  # per-step record
```

## Custom coefficients

```python
custom = [...]  # length 5, all finite
apply_teacache(flux, coefficients=custom)
```

## How it works

TeaCache observes that in diffusion denoising, consecutive transformer outputs change very little between most pairs of adjacent steps. The expensive transformer body (all the joint and single attention blocks) produces a residual that's added to the input, and that residual stays roughly stable for stretches of the denoising trajectory.

TeaCache trains a tiny polynomial that predicts how much the output will change given how much the input has changed, measured as the relative L1 distance of the modulated block-0 input. When the predicted accumulated change since the last real compute step is small enough, TeaCache reuses the cached residual and skips the transformer body. Only the cheap prelude (embeddings) and tail (norm + projection) still run on a skipped step.

mlx-teacache implements this for mflux on Apple Silicon. For FLUX.1 we replace `flux.transformer` with a per-instance proxy; for FLUX.2 we replace `flux._predict` with an instance-level closure, which keeps gating live on chips where mflux would otherwise wrap `_predict` in `mx.compile`. The FLUX.1 polynomial coefficients are vendored from upstream. The FLUX.2 Klein 4B coefficients are derived in-repo by `scripts/calibrate_flux2_klein.py`; see [`docs/calibration.md`](docs/calibration.md) for procedure and provenance. The original method is described in the TeaCache paper at https://liewfeng.github.io/TeaCache/.

## Benchmarks

M1 Max 32GB, bf16, 512×512, `seed=42`, default threshold (`rel_l1_thresh=0.20`):

| Model | Steps | Mode | Vanilla | With TeaCache | Speedup | Quality gate |
|---|---|---|---|---|---|---|
| FLUX.1-dev | 25 | txt2img | ~5:18 | ~3:35 | 1.48× | SSIM ≥ 0.90 (PR-gate prompt) |
| FLUX.1-dev | 25 | img2img, strength 0.7 | — | — | ≈ same skip fraction | SSIM ≥ 0.80 (parametrized suite) |
| FLUX.2 Klein 4B | 8 | txt2img | ~37s | ~30s | ~1.2× | SSIM ≥ 0.85 (PR-gate prompt) |
| FLUX.2 Klein 4B | 8 | img2img, strength 0.5 | — | — | ≈ same skip fraction | SSIM ≥ 0.85 (parametrized suite) |
| FLUX.2 Klein 9B | 8 | txt2img | — | — | tracked-only | SSIM ≥ 0.85 (PR-gate prompt) |

Klein's 8-step speedup is smaller than FLUX.1-dev because Klein is distilled to converge in fewer steps, so the trajectory has fewer adjacent-step redundancies to exploit. Longer schedules (`num_inference_steps >= 15`) give larger gains. The wall-clock columns for img2img are blank because mflux's active window shrinks with `image_strength`, so absolute timings stop being apples-to-apples; the per-step speedup tracks the txt2img skip fraction.

The five prompts in the SSIM suite are defined at `tests/test_image_quality_flux1.py:45` and reused at `tests/test_image_quality_flux2.py:28`:

- "a red apple on a wooden table"
- "mountain landscape at sunset"
- "portrait of a woman"
- "abstract pattern with circles"
- "text saying HELLO"

The PR-gate prompt is the red-apple one. Reproduce these numbers with `uv run pytest tests/test_image_quality_flux1.py -m parity` (requires real model weights).

## Performance by chip

mflux 0.17.5 wraps `_predict` in `mx.compile` on every Apple Silicon chip except base M1 and base M2 (the `is_m1_or_m2()` predicate excludes Max and Ultra variants). mlx-teacache replaces `_predict` with an eager closure so per-step gating stays live. On compiled chips that means giving up the compile gain to get the skip gain. See `docs/m3-plus-tradeoff.md` for a benchmark recipe.

| Chip | Vanilla `_predict` in mflux 0.17.5 | Expected speedup |
|---|---|---|
| Apple M1 / M2 (base) | eager | ≈ pure skip fraction (~1.5–1.6×) |
| M1 Pro / M2 Pro | eager | ≈ pure skip fraction — same as base |
| M1 Max / Ultra, M2 Max / Ultra | compiled | **1.48× measured** on M1 Max FLUX.1-dev / 25 steps |
| M3 / M3 Pro / M3 Max / Ultra | compiled | Likely 1.1–1.3× — untested |
| M4 / M4 Pro / M4 Max | compiled | Likely 1.1–1.3× — untested |
| M5+ (Neural Accelerators / TensorOps) | compiled + accelerator | May approach 1.0×. The eager wrapper can lose some or all of the M5 TensorOps advantage. Confirm with a profiler before treating as fact. |

## Limitations

img2img reuses the txt2img calibration. A dedicated img2img calibration may follow in v0.2.x if SSIM gates flag drift on specific schedules.

FLUX.2 with CFG (`guidance > 1.0`) falls back to vanilla mflux automatically. Output is bit-exact; per-branch caching is planned for v0.3.

Very short schedules cannot benefit. FLUX.1 schnell at 4 steps and Klein at 4 steps have too few non-forced steps for the gate to engage. Use `num_inference_steps >= 8` for Klein, `>= 10` for schnell, or stick to dev for big wins.

Klein variants other than `flux2-klein-4b` are not supported yet. 9B and base configs are planned for v0.3.0.

The wrapper runs eager, which gives up mflux's `mx.compile` of `_predict` in exchange for live per-step gating. Vanilla mflux compiles `_predict` on every chip except base M1 and base M2. The 1.48× measurement is from M1 Max / FLUX.1-dev / 25 steps; speedup on M2 Pro and newer is plausible but untested locally. On M5, the GPU Neural Accelerators (Metal 4 TensorOps) are only reachable through the compiled path, so the eager wrapper can lose some or all of that advantage. Output stays correct either way. See `docs/m3-plus-tradeoff.md` for the per-chip recipe; PRs with measurements welcome.

FLUX.2 parity is numerical, not bit-exact. Replacing a function that mflux wraps in `mx.compile` produces about 1 ULP per element of divergence from Metal kernel-dispatch noise, which compounds across steps but keeps cosine similarity ≥ 0.99. The CFG-fallback path stays bit-exact. The user-facing guarantee is end-to-end image quality (SSIM ≥ 0.85 on Klein 4B).

The mflux pin is strict at `>=0.17,<0.18`. Bumping it is a deliberate release.

Calling `flux.parameters()` at the parent level can miss transformer parameters while the wrapper is active. Use `flux.transformer.parameters()` directly, or call `handle.restore()` first.

## License obligations

The FLUX.1 variants (`flux1-dev`, `flux1-schnell`) and `flux2-klein-4b` come with their own upstream weight licenses; the wrapper this library applies does not change those terms.

`flux2-klein-9b` is distributed under the FLUX.2 Klein license (non-commercial use + BFL safety-filter obligations). These terms flow with the weights, not with mlx-teacache. If you call `apply_teacache(Flux2Klein(model_config=ModelConfig.flux2_klein_9b()))`, you are responsible for ensuring your use complies with the upstream license — including the safety-filter requirements that the BFL model card describes. See the official model card at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B for the full terms.

## Contributing

Open an issue at https://github.com/IonDen/mlx-teacache/issues.

## License + acknowledgements

Apache-2.0. See `LICENSE` and `NOTICE`.

- [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) — upstream method and FLUX.1 coefficients.
- [filipstrand/mflux](https://github.com/filipstrand/mflux) — MLX FLUX runner this library integrates with.
- [Apple ML Explore](https://github.com/ml-explore/mlx) — MLX.
