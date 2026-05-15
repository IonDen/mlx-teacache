# mlx-teacache

[![PyPI version](https://img.shields.io/pypi/v/mlx-teacache.svg)](https://pypi.org/project/mlx-teacache/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlx-teacache.svg)](https://pypi.org/project/mlx-teacache/)
[![License: Apache 2.0](https://img.shields.io/pypi/l/mlx-teacache.svg)](https://github.com/IonDen/mlx-teacache/blob/main/LICENSE)
[![CI](https://github.com/IonDen/mlx-teacache/actions/workflows/ci.yml/badge.svg)](https://github.com/IonDen/mlx-teacache/actions/workflows/ci.yml)

**TeaCache step-skipping for FLUX diffusion on Apple Silicon, in pure MLX.**

`mlx-teacache` is the first MLX port of TeaCache — a training-free inference optimization that skips ~20-50% of denoising steps in FLUX-family diffusion models by predicting which steps contribute little to the final image. Measured ~1.48× wall-clock speedup at the default threshold on FLUX.1-dev / 25 steps with visually-equivalent output (SSIM ≥ 0.80 on a 5-prompt suite).

## What it does

Diffusion models run the same big transformer 20-50 times in a loop. Between consecutive steps the output changes very little; TeaCache uses a tiny polynomial fit to predict which steps can reuse the previous step's output. On M1 Max FLUX.1-dev @ 25 steps the default threshold (`rel_l1_thresh=0.20`) skips 6 of 25 steps for a ~1.48× speedup.

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
pip install "mlx-teacache==0.1.0[mflux]"  # pin for reproducibility
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

## Threshold guide

Measured on M1 Max 32GB, FLUX.1-dev @ 25 steps, bf16, `seed=42`, `guidance=3.5`, red-apple prompt:

| `rel_l1_thresh` | Skipped steps | Speedup | SSIM vs vanilla | Recommended use |
|---|---|---|---|---|
| 0.10 | 0 / 25 | 1.07× | 1.0000 | Cache never engages |
| 0.15 | 0 / 25 | 1.13× | 1.0000 | Cache never engages |
| **0.20 (default)** | **6 / 25** | **1.48×** | **≥ 0.80 (5-prompt suite)** | **Visually-lossless sweet spot** |
| 0.25 | 11 / 25 | 1.96× | 0.57-0.93 | Visible style changes on text/synthetic prompts |

The 0.20 default was chosen after side-by-side visual comparison: at 0.25 a text prompt that vanilla renders as neon tubes can come out as dot-matrix; at 0.20 the output is indistinguishable from vanilla while still skipping ~25% of steps. SSIM is a conservative metric on high-frequency-detail prompts (text, synthetic patterns).

## Supported models

| Variant id | mflux class + config | Coefficient source |
|---|---|---|
| `flux1-dev` | `Flux1(model_config=ModelConfig.dev())` | upstream ali-vilab/TeaCache |
| `flux1-schnell` | `Flux1(model_config=ModelConfig.schnell())` | upstream (shared with dev) |
| `flux2-klein-4b` | `Flux2Klein(model_config=ModelConfig.flux2_klein_4b())` | in-repo (see `docs/calibration.md`) |

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

TeaCache observes that in diffusion denoising, consecutive transformer outputs change very little between most pairs of adjacent steps. The expensive transformer body (all the joint + single attention blocks) produces a residual that's added to the input — and that residual stays roughly stable for stretches of the denoising trajectory.

TeaCache trains a tiny polynomial that predicts how much the output will change given how much the input has changed (measured as relative L1 distance of the modulated block-0 input). When the predicted accumulated change since the last actual compute step is small, TeaCache reuses the cached residual instead of running the full transformer body — only the cheap prelude (embeddings) and tail (norm + projection) still run.

mlx-teacache implements this for mflux on Apple Silicon. We replace `flux.transformer` with a per-instance proxy for FLUX.1, and `flux._predict` with an instance-level closure for FLUX.2 (to keep gating live even when `mx.compile` is normally used). The polynomial coefficients for FLUX.1 are vendored from upstream; the FLUX.2 Klein 4b coefficients are derived in-repo via `scripts/calibrate.py`. See the TeaCache paper at https://liewfeng.github.io/TeaCache/ and `docs/superpowers/spikes/2026-05-14-mlx-teacache-phase-0-spike.md` for the Apple-Silicon-specific design notes.

## Benchmarks

M1 Max 32GB, default threshold (`rel_l1_thresh=0.20`):

| Model | Steps | Vanilla | With TeaCache | Speedup | SSIM (PR-gate prompt) |
|---|---|---|---|---|---|
| FLUX.1-dev @ 512² | 25 | ~5:18 | ~3:35 | 1.48× | ≥ 0.90 |
| FLUX.2 Klein 4b @ 512² | 8 | ~37s | ~30s | ~1.2× | ≥ 0.85 |

FLUX.2's smaller speedup at 8 steps reflects the short non-distilled denoising trajectory at Klein's recommended step count. Longer schedules (`num_inference_steps>=15`) give larger gains.

## Limitations

- **v0.1 is txt2img only.** img2img inputs raise `Img2ImgNotSupportedError`. v0.2 planned.
- **FLUX.2 CFG (`guidance > 1.0`) auto-falls-back to vanilla mflux.** Output is bit-exact. v0.2 will add per-branch caching.
- **Distilled schedules see no speedup.** FLUX.1 schnell 4-step and Klein 4-step defaults have too few non-forced steps. Use `num_inference_steps >= 10` for benefit.
- **Klein variants other than `flux2-klein-4b` are not supported in v0.1.** 9b and base configs planned for v0.2.
- **M3+ users lose mflux's `mx.compile` of `_predict`.** v0.1 provides a manual benchmark recipe (`docs/m3-plus-tradeoff.md`) and does not claim a measured M3+ speedup.
- **FLUX.2 parity is numerical, not bit-exact.** Because the wrapper replaces a function mflux wraps in `mx.compile`, vanilla-compiled vs wrapper-eager differ by ~1 ULP per element from Metal kernel-dispatch noise (compounds across steps; cosine similarity stays ≥ 0.99). The CFG-fallback path remains bit-exact. End-to-end image quality (SSIM ≥ 0.85 on Klein 4b) is the user-facing guarantee.
- **mflux pin is strict (`>=0.17,<0.18`).** Bumping requires a deliberate release.
- **Parent-level `flux.parameters()` may miss transformer parameters while patched.** Use `flux.transformer.parameters()` directly, or `handle.restore()` first.

## Contributing

Open an issue at https://github.com/IonDen/mlx-teacache/issues.

## License + acknowledgements

Apache-2.0. See `LICENSE` and `NOTICE`.

- [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) — upstream method and FLUX.1 coefficients.
- [filipstrand/mflux](https://github.com/filipstrand/mflux) — MLX FLUX runner this library integrates with.
- [Apple ML Explore](https://github.com/ml-explore/mlx) — MLX.
