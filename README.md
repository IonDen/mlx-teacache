# mlx-teacache

<p align="center">
  <img src="https://raw.githubusercontent.com/IonDen/mlx-teacache/main/docs/assets/mlx-teacache-logo.png" alt="mlx-teacache" width="100%">
</p>

[![PyPI version](https://img.shields.io/pypi/v/mlx-teacache.svg)](https://pypi.org/project/mlx-teacache/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/mlx-teacache/)
[![License: Apache 2.0](https://img.shields.io/pypi/l/mlx-teacache.svg)](https://github.com/IonDen/mlx-teacache/blob/main/LICENSE)
[![CI](https://github.com/IonDen/mlx-teacache/actions/workflows/ci.yml/badge.svg)](https://github.com/IonDen/mlx-teacache/actions/workflows/ci.yml)

**Faster image generation on Apple Silicon: TeaCache skips the denoising steps that barely change the picture. For Qwen-Image, FLUX.2 Klein base, Z-Image and FLUX.1 on mflux.**

mlx-teacache makes image generation on an Apple Silicon Mac faster. It plugs into [mflux](https://github.com/mflux-community/mflux), which runs open image models on MLX, Apple's machine-learning framework. You wrap the model in one call, `apply_teacache(flux)`, and keep your prompt, seed and settings. At each model's default setting the image stays very close to what plain mflux makes; it just arrives sooner. Nothing is retrained and nothing extra is downloaded.

Measured on an M1 Max with 32 GB and mflux 0.18.0, against plain mflux, at each model's default threshold. Qwen-Image ran at 768×768 with 4-bit weights, Z-Image base at 512×512 with 8-bit weights, the others at 512×512 with 4-bit weights:

| Model | Speedup | Steps skipped |
|---|---|---|
| Qwen-Image (original checkpoint) | 2.68× | 33 of 50 |
| FLUX.2 Klein base 9B | 1.37× | 13 of 50 |
| FLUX.2 Klein base 4B | 1.20× | 8 of 50 |
| Z-Image base | 1.31× | 15 of 50 |
| FLUX.1 Krea [dev] | 1.62× | 10 of 28 |
| FLUX.1 dev | 1.57× | 6 of 25 |

Each speedup is the median of three runs, each in a fresh process; [Benchmarks](#benchmarks) has the full recipe and a command to reproduce every row. Image similarity to plain mflux (SSIM, where 1.0 means identical) was 0.985 and 0.993 for the two Klein base runs, 0.967 to 0.991 for Qwen-Image, Z-Image and Krea in their threshold sweeps, and FLUX.1 dev passes the test suite's SSIM floor of 0.80 across five prompts. On mflux 0.19 and later the `qwen-image` name loads Qwen-Image-2512, which has its own calibration but no speed measurement yet.

It does not help *distilled* models, which are trained to finish in 4 to 8 steps: FLUX.1 schnell and FLUX.2 Klein 4B and 9B (the ones without "base" in the name). Each of their few steps changes the picture a lot, so there is nothing safe to skip. Run those with plain mflux. On Max, Ultra and M3-or-newer chips the wrapper for FLUX.2 Klein and Z-Image also sidesteps mflux's compiled prediction step, a separate small effect on speed; [How the speedup happens](#how-the-speedup-happens) explains it.

## Which library do I need?

**You want image generation to run faster on a Mac?** mlx-teacache skips steps on Qwen-Image, FLUX.2 Klein base, Z-Image and FLUX.1, with one line added to an mflux script.

**You want live previews while generating, or low-memory latent decode?** You want [`mlx-taef`](https://github.com/IonDen/mlx-taef), tiny preview decoders (the TAESD family) in MLX.

**You want both?** They work together. TeaCache skips steps, and mlx-taef's TAEF2 decoder gives cheap previews.

## What it does

An image model starts from random noise and cleans it up over 20 to 50 steps. Each step runs the same large neural network (a transformer), and that is where most of the time goes. Between neighboring steps, the network's output often changes very little.

TeaCache estimates how much the output will change before running the full network. It measures how much a cheap early part of the network changed since the previous step and feeds that number into a small formula fitted in advance for the model. While the estimated changes stay small, TeaCache skips the expensive part of the network and reuses what that part added on the last full step. When the estimates add up past a threshold, `rel_l1_thresh`, it runs the network again and starts counting from zero. A lower threshold skips fewer steps and stays closer to the plain image. Most models ship their own default threshold, chosen by comparing images at several values; FLUX.1 dev and schnell and the distilled Klein models use the package default, 0.20.

mlx-teacache is the first MLX port of TeaCache. [How it works](#how-it-works) has the details and a link to the original paper.

## Install

```bash
pip install "mlx-teacache[mflux]"
# or with uv:
uv add "mlx-teacache[mflux]"
```

```bash
pip install "mlx-teacache[mflux]==0.14.1"  # pin for reproducibility
```

Requires Python ≥ 3.10 and Apple Silicon. The `[mflux]` extra pulls in `mflux>=0.17.5,<0.26`. Version notes:

- With pillow 13.0.0 on Python 3.11 to 3.14, every mflux release up to 0.23.0 fails when you import its model classes or run its commands, because mflux reads a private pillow module that pillow 13 broke. The fix is merged in mflux ([PR 890](https://github.com/mflux-community/mflux/pull/890)) but not yet released; until then, install `pillow<13` alongside.
- mflux 0.19 and newer need MLX 0.32 and torch 2.13 or newer (torch and opencv have been mflux dependencies since before 0.18).
- For mlx-taef's live preview on mflux 0.19 or newer, use mlx-taef 0.8.1 or later. mlx-taef 0.8.3's own `[mflux]` extra still pins `mflux<0.21`, so installing `mlx-teacache[mflux]` and `mlx-taef[mflux]` together resolves to mflux 0.20. To use the live preview with mflux 0.21 or newer, install `mlx-taef` without its `[mflux]` extra next to `mlx-teacache[mflux]`; mlx-taef's mflux integration imports on 0.21 through 0.23, and only the extra's version bound blocks the resolver.
- From mflux 0.19 on, the `qwen-image` alias loads `Qwen/Qwen-Image-2512`. Since mlx-teacache 0.14.0 that checkpoint has its own variant and calibration, so `apply_teacache` no longer warns about it. The original `Qwen/Qwen-Image` keeps its own variant only when you build it in Python with `ModelConfig.from_name("Qwen/Qwen-Image")` (see the Qwen-Image notes under [Supported models](#supported-models)). The separate Qwen-Image-2.1 model (mflux 0.20 and later, Qwen Research License, non-commercial) is not a supported variant.

## From the mflux command line

If you run mflux from the terminal rather than from Python, the `mflux-teacache` plugin adds a TeaCache version of the Z-Image command. It takes every option of `mflux-generate-z-image` and prints how many steps it skipped.

Run `pip install mflux-teacache` in an environment that already has mflux. For a fresh setup, follow the [Install section of the plugin's README](https://github.com/mflux-community/mflux-plugins/tree/main/teacache#install).

```bash
pip install mflux-teacache
mflux-generate-z-image-teacache --prompt "a lighthouse on a cliff at dusk" --steps 50 --guidance 4.0 --seed 42 -q 8
```

That is the only command today, and it covers Z-Image base only. Before it loads weights, the command refuses an unsupported or distilled model and a run with too few steps. The Z-Image command has no step-cache option, so mflux's own step cache never comes into play there. It needs mflux 0.22 or newer and mlx-teacache 0.14.0 or newer. The [plugin's README](https://github.com/mflux-community/mflux-plugins/tree/main/teacache#readme) lists the options and versions.

## Quick start

FLUX.2 Klein base 4B makes a good first test: it is small, Apache-2.0 and needs no license click-through.

```python
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mlx_teacache import apply_teacache

flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
with apply_teacache(flux) as handle:  # this model's default rel_l1_thresh=0.17
    generated = flux.generate_image(
        prompt="a red apple on a wooden table",
        seed=42,
        num_inference_steps=50,
        guidance=4.0,
        width=512,
        height=512,
    )
    print(f"Skipped {handle.stats.skipped_count} steps, speedup ≈ {handle.stats.speedup_estimate:.2f}×")
generated.image.save("apple.png")
```

On an M1 Max this takes about three to four minutes (Benchmarks footnote ³). Every supported model is wrapped the same way: build it with mflux, then call `apply_teacache`. Each page under [`docs/variants/`](docs/variants) gives the mflux constructor and the recipe. Qwen-Image is the heavy one: at 4-bit and 768×768 it uses about all the memory a 32 GB Mac can give the GPU (the Qwen-Image-2512 note under Supported models gives the peak).

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

`rel_l1_thresh` trades speed against closeness to the plain image. The sweep below is for FLUX.1 dev; the variant pages under [`docs/variants/`](docs/variants) cover the other models.

Measured on M1 Max 32GB, FLUX.1-dev @ 25 steps, bf16, `seed=42`, `guidance=3.5`, red-apple prompt:

| `rel_l1_thresh` | Skipped steps | Speedup | SSIM vs vanilla | Recommended use |
|---|---|---|---|---|
| 0.10 | 0 / 25 | 1.07× | 1.0000 | Cache never engages |
| 0.15 | 0 / 25 | 1.13× | 1.0000 | Cache never engages |
| **0.20 (default)** | **6 / 25** | **1.57×** | **≥ 0.80 (5-prompt suite)** | **Visually-lossless sweet spot** |
| 0.25 | 11 / 25 | 1.96× | 0.57-0.93 | Visible style changes on text/synthetic prompts |

The committed benchmark `_artifacts/v0.10.0_bench_flux1_dev.json` validates the 0.20 default row (median of three cold reps, 2026-08-15); the other rows are single-run measurements from an earlier session with a faster vanilla baseline (~104 s), so read them for the trend, not as a committed multi-rep sweep.

0.20 was picked after side-by-side visual comparison. At 0.25, text prompts that vanilla renders as neon tubes can come out as dot-matrix. At 0.20, the output is indistinguishable from vanilla and the cache still skips around 25% of steps. SSIM is conservative on high-frequency-detail prompts like text and synthetic patterns, which is why the suite floor (0.80) is lower than the PR-gate floor (0.90) on the red-apple prompt.

## Supported models

Only the models below are supported; `apply_teacache` refuses any other. They are grouped by family, and the families come in order of how widely they are used on Macs: Hugging Face downloads of the MLX and mflux copies of the models TeaCache speeds up, plus web search interest over three months, checked in October 2026. The older FLUX.1 family comes last, even though its official checkpoint is still the most downloaded overall. Inside a family, the models TeaCache speeds up come first and the distilled ones after, each group ordered by MLX-copy downloads. Distilled models finish in 4 to 8 steps and get nothing from step-skipping; see [When to use mlx-teacache](#when-to-use-mlx-teacache). In the table, g is the guidance scale. For Qwen-Image, FLUX.2 Klein base and Z-Image, g above 1.0 adds a negative-prompt pass each step (classifier-free guidance, CFG); FLUX.1 dev and Krea take g as an input to the network and still run one pass.

The table is generated from the variant registry by `docs/_generate_supported_models.py`. Per-variant detail (mflux constructor, coefficient provenance, quirks) lives in `docs/variants/<id>.md`.

<!-- SUPPORTED_MODELS_START -->
| Variant id | Display name | Distilled? | Default recipe | License |
|---|---|---|---|---|
| `qwen-image-2512` | Qwen-Image-2512 | no | 50 steps, g=4.0 | [Apache-2.0](https://huggingface.co/Qwen/Qwen-Image-2512) |
| `qwen-image` | Qwen-Image | no | 50 steps, g=4.0 | [Apache-2.0](https://huggingface.co/Qwen/Qwen-Image) |
| `flux2-klein-base-9b` | FLUX.2 Klein base 9B | no | 50 steps, g=4.0 | [FLUX Non-Commercial](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) |
| `flux2-klein-base-4b` | FLUX.2 Klein base 4B | no | 50 steps, g=4.0 | [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) |
| `flux2-klein-4b` | FLUX.2 Klein 4B | yes | 8 steps, g=1.0 | [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| `flux2-klein-9b` | FLUX.2 Klein 9B | yes | 8 steps, g=1.0 | [FLUX Non-Commercial](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |
| `z-image-base` | Z-Image base | no | 50 steps, g=4.0 | [Apache-2.0](https://huggingface.co/Tongyi-MAI/Z-Image) |
| `flux1-dev` | FLUX.1 dev | no | 25 steps, g=3.5 | [FLUX.1-dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| `flux1-krea-dev` | FLUX.1 Krea [dev] | no | 28 steps, g=4.5 | [FLUX.1-dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev) |
| `flux1-schnell` | FLUX.1 schnell | yes | 4 steps, g=1.0 | [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |
<!-- SUPPORTED_MODELS_END -->

### Per-variant notes

Each variant has its own page under [`docs/variants/`](docs/variants) — mflux constructor, recipe, license obligations, coefficient provenance, quirks. The highlights for the variants where behavior diverges from the default story:

**[`qwen-image-2512`](docs/variants/qwen-image-2512.md)** — Qwen-Image-2512 (Alibaba, Apache-2.0), the newer checkpoint that mflux 0.19 and later load under the `qwen-image` alias. It runs the same proxy forward and gate signal as `qwen-image` (next entry), but the original checkpoint's coefficients do not fit it (R² −0.35 on 2512's calibration data), so it has its own fit: R² 0.818, held-out 0.785, ten prompts at the 768×768, 50-step, guidance 4.0, 4-bit recipe on mflux 0.23.0. Per-variant default `rel_l1_thresh=0.25`, set from its threshold sweep on the red-apple prompt at that recipe: 29 of 48 active steps skipped at SSIM 0.969 against plain mflux, longest run of skips 4. The original checkpoint's 0.30 falls to SSIM 0.947 on 2512. There is no speed benchmark for this variant yet. On mflux 0.19 and later, original Qwen-Image weights loaded through the command line, a `model_path` or a mirror resolve against the 2512 config and are detected as 2512, because that config keeps no trace of the original. To run them on the original calibration, pass `coefficients=` with the original's tuple and `rel_l1_thresh=0.30`; the [variant page](docs/variants/qwen-image-2512.md) has the details. It also covers the memory needed: each calibration prompt peaked at 25.93 GiB, above the roughly 25 GiB working set of a 32 GB Mac.

**[`qwen-image`](docs/variants/qwen-image.md)** — Qwen-Image base (Alibaba, Apache-2.0), a ~20B dual-stream MMDiT and the first variant that proxies `flux.transformer` (the FLUX.1 pattern) *and* runs true two-pass CFG. It's FLUX-shaped, so the gate taps the FLUX-canonical modulated block-0 input, calibrated in-repo at R² 0.849 (well above Z-Image's 0.400 and the FLUX.2 family's 0.11–0.47). Per-variant default `rel_l1_thresh=0.30`, set from the threshold sweep. At the red-apple 768×768 recipe it skips 33 of 48 active steps for **2.68× combined** (642.6 s → 239.4 s), SSIM 0.967 vs vanilla — the largest gain of any variant, all of it step-skipping (Qwen has no `mx.compile` path). Lowering the threshold to 0.20 gives back the earlier SSIM-0.978 operating point; the variant page has the whole sweep and footnote ¹ has the measurement history. **From mflux 0.19 on, the `qwen-image` alias loads `Qwen/Qwen-Image-2512`**, which has its own variant (above). Every number in this entry is the original checkpoint on mflux 0.18.0. On mflux 0.19 and later you get the original only from Python, by passing `ModelConfig.from_name("Qwen/Qwen-Image")` as the `model_config`; mflux's command line and any `model_path`, such as a local directory, resolve to Qwen-Image-2512. mflux 0.20's separate Qwen-Image-2.1 (Qwen Research License, non-commercial) is a different architecture and not a supported variant; `apply_teacache` raises `IncompatibleModelError` for it. On a 32 GB Mac stock 4-bit Qwen is grainy — the [comparison page](https://github.com/IonDen/mlx-teacache/blob/main/docs/comparison/qwen-image.md) now runs the stock 4-bit build directly, and the [variant page](docs/variants/qwen-image.md) documents a mixed-precision build that clears the artifact; mlx-teacache stays quantization-agnostic.

**[`flux2-klein-base-9b`](docs/variants/flux2-klein-base-9b.md)** — non-distilled FLUX.2 Klein 9B (FLUX Non-Commercial — see [License obligations](#license-obligations) and accept on the [Hugging Face model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) before downloading). Reuses base-4b's coefficients and `rel_l1_thresh=0.17` default (shared FLUX.2-Klein architecture family and calibration recipe). At the canonical 50-step + g=4.0 CFG recipe it skips 13/48 active steps for **1.37× combined wall-clock** (517.3 s → 377.5 s median; 1.35× gating + 1.02× compile-avoidance), at SSIM 0.993 against vanilla and a peak-memory drop from ~22 GB to ~9.8 GB. Full history in footnote ².

> **Correction.** v0.5.0 advertised 2.68× on this variant. Same-process MLX state leakage inflated it — vanilla ran cold while the wrapper inherited a warm allocator. Under cold subprocess-per-rep isolation the honest number is 1.37× (footnote ²).

See `_artifacts/v0.12.0_bench_klein_base_9b.json` for the full report (the v0.10.0 and v0.6.0 reports sit alongside it); regenerate side-by-side images with `scripts/bench_comparison.py`.

**[`flux2-klein-base-4b`](docs/variants/flux2-klein-base-4b.md)** — non-distilled FLUX.2 Klein 4B (Apache-2.0), the primary FLUX.2 use case. At the canonical upstream CFG recipe (`guidance=4.0, num_inference_steps=50`) with its per-variant default `rel_l1_thresh=0.17`, the gate skips 8/48 active steps for a **1.20× combined speedup** on M1 Max, almost all of it step-skipping, with compile-avoidance at noise level (1.01×). CFG runs through a per-branch gated path (each branch keeps its own cached residual). Measurement history, including v0.4.1's superseded decomposition, is in footnote ³.

**[`flux2-klein-9b`](docs/variants/flux2-klein-9b.md)** — coefficients are calibrated at `num_inference_steps=8`, origin-constrained polyfit. At the default threshold, the gate produces 0 step-skips on Klein 9B's 8-step schedule (the empirical adjacent-step body-output rel-L1 starts at 0.25 — above the 0.20 threshold). The only wall-clock effect left is `mx.compile`-path avoidance, and only on Macs where mflux compiles the prediction step; the gate contributes nothing, and output quality is preserved (SSIM ≥ 0.85 PR-gate). See [Benchmarks](#benchmarks) → "How the speedup happens".

**[`z-image-base`](docs/variants/z-image-base.md)** — Z-Image base (Tongyi-MAI, Apache-2.0), a single-stream DiT and the first non-FLUX model with a TeaCache mini-kernel. Its adaLN modulation is timestep-only, so there is no cheap caption-independent modulation input to gate on; the gate signal is the first-main-layer residual (calibrated in-repo as "Signal B", R² 0.400 — the threshold sweep, not the fit R², sets the quality bar here; the caption-independent noise-refiner tap was tried and rejected at R² 0.069). Per-variant default `rel_l1_thresh=0.12`, set at the SSIM knee. At the 512×512 red-apple recipe it skips 15/48 active steps for **1.31× combined wall-clock** (227.4 s → 174.2 s), entirely from gating — the no-gate wrapper runs at vanilla speed (0.99×), so compile-avoidance is not a tailwind here. Peak memory drops 17.2 GB → 11.5 GB (the eager wrapper bypassing mflux's compiled `_predict`, not the gate). SSIM 0.991 vs vanilla. Footnote ⁵ has the measurement history; [docs/comparison/z-image-base.md](https://github.com/IonDen/mlx-teacache/blob/main/docs/comparison/z-image-base.md) has a separate single-run comparison at 864×1152.

See `_artifacts/_bench_z_image_v0_7_0.json` for the full report.

**[`flux1-krea-dev`](docs/variants/flux1-krea-dev.md)** — FLUX.1 Krea [dev] (Black Forest Labs with Krea; FLUX.1-dev Non-Commercial, gated), a FLUX.1-dev-architecture finetune that reuses the FLUX.1 proxy strategy unchanged. It does not reuse FLUX.1-dev's coefficients: scored on Krea's own calibration pairs, dev's tuple gives R² −496, because Krea changes roughly three times more per step, so Krea ships its own fit (R² 0.68, calibrated in-repo at the model card's 28-step, guidance 4.5 recipe) and a per-variant default `rel_l1_thresh=0.30`, the sharp knee of its threshold sweep (SSIM 0.990 at 0.30, 0.890 at 0.35). The package fallback 0.20 would skip nothing on this model. The v0.11.0 three-way bench at the red-apple 512×512 recipe (subprocess-per-rep, three cold reps, q4, 28 steps, g=4.5): **1.62× combined** (125.3 s → 77.2 s), 10 of 26 active steps skipped in every rep and never two in a row; 1.57× of it is gating, and the 1.03× the no-gate wrapper shows is noise, since FLUX.1 has no `mx.compile` path to avoid. Reproduce with `uv run python scripts/bench_speedup.py --variant krea-dev --three-way --reps 3`.

## When to use mlx-teacache

TeaCache helps when neighboring steps look alike, so the network's output barely moves between them. In the models measured so far, that held for non-distilled models at 20 or more steps.

In practice, that means:

- Use mlx-teacache for **`qwen-image`** at 50 steps with CFG, the **non-distilled FLUX.2 Klein** family (`flux2-klein-base-9b`, `flux2-klein-base-4b`) at 20-50 steps with or without CFG, **`z-image-base`** at 50 steps with CFG, **`flux1-dev`** at 20-50 steps, and **`flux1-krea-dev`** at its 28-step recipe. These are the variants featured in [COMPARISON.md](https://github.com/IonDen/mlx-teacache/blob/main/COMPARISON.md), and the wrapper measurably skips steps and produces visually equivalent output. **`qwen-image-2512`** has its own calibration and threshold sweep but no speed benchmark yet.
- Do not reach for it on the **distilled** variants — `flux1-schnell` (4 steps), `flux2-klein-4b` and `flux2-klein-9b` at their distilled defaults (4-8 steps). The residual between adjacent steps is too large for the gate to engage at any reasonable threshold, so it skips zero steps and adds about 1-2% gating overhead. Run those through vanilla mflux. The full story of that zero-skip result is in [the distilled-schedules research note](https://ineshin.space/papers/why-teacache-does-not-engage-on-short-distilled-schedules/).

On FLUX.2 variants there can be a separate, incidental benefit whether or not the gate engages: on Macs where mflux compiles the prediction step (not base or Pro M1/M2), the wrapper sidesteps the compiled `_predict` path, which on Max and Ultra chips has been slower than the uncompiled one. That is a wall-clock effect from compile avoidance, not from step-skipping, and we keep the two attributions separate in the docs.

Distilled Klein or Base plus the wrapper? `scripts/bench_klein_base_vs_distilled.py` runs distilled, Base, and Base + TeaCache on one prompt and seed and prints a table. On an M1 Max at q4 the distilled 4-step model is about 22× faster than Base at 50 steps on 4B (26 s against 589 s; 768×1024, 3 reps) and about 20× on 9B (26 s against 522 s; 512×512, 2 reps), and the wrapper trims 17 percent off Base on 4B and 29 percent on 9B rather than closing that gap (`_artifacts/klein_base_vs_distilled_4b_q4.json`, `_artifacts/klein_base_vs_distilled_9b_q4_512x512.json`, both measured with mlx-teacache 0.11.0). So the distilled model stays the everyday choice, and Base is for when you need guidance, negative prompts, or a fine-tune. Run it with `uv run python scripts/bench_klein_base_vs_distilled.py --size 4b --quantize 4 --height 1024 --width 768 --reps 3` (for 9B, `--size 9b --quantize 4 --height 512 --width 512 --reps 2`); on a 64 GB or larger Mac, run it at `--quantize 8` and post your table.

See [COMPARISON.md](https://github.com/IonDen/mlx-teacache/blob/main/COMPARISON.md) for one scene per model with TeaCache off and on, step-by-step previews, and single-run timings on an M1 Max — the [Benchmarks](#benchmarks) section below has the multi-run numbers.

## Combining with mlx-taef

```python
from mlx_taef.integrations.mflux import LivePreviewCallback
from mlx_teacache import apply_teacache

preview = LivePreviewCallback(variant="taef2", every=5, save_to="preview.png",
                              latent_height=32, latent_width=32)
flux.callbacks.register(preview)

with apply_teacache(flux):  # the variant's default rel_l1_thresh
    flux.generate_image(prompt="...", seed=42, num_inference_steps=25)
```

## Checking support before loading

```python
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.z_image.variants.z_image import ZImage

from mlx_teacache import check_step_window, match_variant

info = match_variant(ModelConfig.from_name("z-image"), ZImage)
print(info.variant_id, info.default_thresh)  # z-image-base 0.12

check_step_window(50)  # returns; check_step_window(2) raises InvalidStepWindowError
```

`match_variant` returns `None` for a model or pipeline that `apply_teacache` would refuse. Otherwise it returns a `VariantInfo` with the variant id, a display name, the default threshold, the mflux model names the variant is detected by, and `calibrated`. `default_thresh` is `None` for the distilled Klein 4B and 9B, where the gate skips nothing. `calibrated` is `False` only for a config that names neither the checkpoint the coefficients were fitted on nor that checkpoint as its base model (only the two Qwen-Image variants check this); `apply_teacache` still patches such a model and warns. mflux records the base model in every config it resolves, so a finetune or a mirror loaded through mflux reports `calibrated=True`: the field cannot tell it from the checkpoint itself. Qwen-Image-2512, the `qwen-image` alias on mflux 0.19 and later, has its own variant since 0.14.0 and reports `calibrated=True`.

`check_step_window` raises when a run has too few denoising steps for the skip window. By default one step at each end always computes, so 3 or fewer active steps skip nothing and 2 or fewer are refused. A 3-step run therefore passes `check_step_window`, but `apply_teacache` warns that it cannot skip anything. For image-to-image, pass the number of steps mflux actually runs. A count of 0 is refused as well, which is stricter than `apply_teacache`, where a zero-step run is a valid no-op. Neither call reads weights, and both apply the same window rule as `apply_teacache`.

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

Passing `coefficients` also changes the default threshold. With your own polynomial, the FLUX.2, Z-Image and both Qwen-Image variants use `rel_l1_thresh=0.20` unless you pass one, because their per-variant defaults are tuned for the built-in polynomial. The FLUX.1 variants keep their per-variant default (0.30 for `flux1-krea-dev`). Your polynomial is evaluated on the raw input change. The calibrated-range clamp described under [How it works](#how-it-works) applies only to a variant's own built-in tuple.

## How it works

TeaCache observes that in diffusion denoising, consecutive transformer outputs change very little between most pairs of adjacent steps. The expensive transformer body (all the joint and single attention blocks) produces a residual that's added to the input, and that residual stays roughly stable for stretches of the denoising trajectory.

TeaCache trains a tiny polynomial that predicts how much the output will change given how much the input has changed, measured as the relative L1 distance between each step's modulated block-0 input and the immediately previous step's — matching how the polynomials are calibrated, and the comparison anchor advances on every step, including skipped ones. TeaCache accumulates the predicted change across consecutive skips; once that running total would cross `rel_l1_thresh`, TeaCache recomputes the transformer body instead of reusing the cached residual, and resets the running total. For the polynomials calibrated in this repo (FLUX.2 Klein base, Z-Image, Qwen-Image, Qwen-Image-2512), the input change is first held to the range the calibration actually saw, so a change smaller than any calibrated step is priced like the smallest one rather than as almost no change, and a change larger than any calibrated step always recomputes. Only the cheap prelude (embeddings) and tail (norm + projection) still run on a skipped step.

mlx-teacache implements this for mflux on Apple Silicon. For FLUX.1 we replace `flux.transformer` with a per-instance proxy; for FLUX.2 we replace `flux._predict` with an instance-level closure, which keeps gating live on chips where mflux would otherwise wrap `_predict` in `mx.compile`. The FLUX.1 polynomial coefficients are vendored from upstream. The FLUX.2 coefficients are derived in-repo by `scripts/calibrate_flux2.py` (distilled klein-4b and klein-9b, base-4b each have their own calibrated tuple; klein-base-9b cross-imports base-4b's tuple because the architectures share the calibration recipe). See [`docs/calibration.md`](docs/calibration.md) for procedure and provenance, and the per-variant docs under `docs/variants/` for each tuple's R² and origin date. The original method is described in the TeaCache paper at https://liewfeng.github.io/TeaCache/.

For the full contributor's map — the variant registry, the mflux-free import contract, the per-family patch strategies, the gate internals, and the teardown contract — see [`docs/architecture.md`](docs/architecture.md).

## Benchmarks

The timings come from `scripts/bench_speedup.py` on an M1 Max with 32 GB; the SSIM values in the footnotes come from the threshold sweeps and image comparisons each footnote names. *Vanilla* is plain mflux, *Wrapper* is the same run with mlx-teacache, and *Skipped* counts the steps where the transformer body did not run, out of the nominal step count. The first and last steps always run; where a footnote or note says "active steps", it counts out of the steps the gate can skip (48 in a 50-step run). Each condition ran three times, every run in a fresh Python process so it starts from a cold MLX allocator, and the table gives the median.

Shared recipe: bf16, quantize=4, 512×512, `seed=42`, the red-apple prompt, and each variant's default threshold (`0.30` on qwen-image and flux1-krea-dev, `0.17` on the base-4b and base-9b rows, `0.12` on z-image-base, `0.20` on flux1-dev). Two rows use their own pinned recipe: `qwen-image` runs at 768×768 and `z-image-base` at q8. The machine ran macOS 26.x, except macOS 27.0 for the base-4b and base-9b CFG rows. `qwen-image` was measured on 2026-09-06 (footnote ¹), the base-9b and base-4b CFG rows on 2026-09-28 with the v0.12.0 gate (footnotes ² ³), `z-image-base` and `flux1-dev` on 2026-08-15 with the v0.10.0 gate (footnotes ⁵ ⁶), and `flux1-krea-dev` on 2026-09-05 (footnote ⁷). Every measured row ran on mflux 0.18.0; the footnotes give the versions each run recorded. The development lock now resolves mflux 0.23.0, but new mflux releases are not re-benchmarked, so each row stays tied to the version it was measured on.

| Variant | Steps | Vanilla | Wrapper | Speedup | Skipped | Mechanism |
|---|---|---|---|---|---|---|
| `qwen-image` (CFG)¹ | 50 | 642.6s | 239.4s | **2.68×** | **33 / 50** | step-skipping (2.57× against its own no-gate run; the no-gate wrapper's 1.04× is inside the rep spread, see ¹) |
| `flux2-klein-base-9b` (CFG)² | 50 | 517.3s | 377.5s | **1.37×** | **13 / 50** | step-skipping + small compile-avoidance |
| `flux2-klein-base-4b` (CFG)³ | 50 | 237.5s | 197.8s | **1.20×** | **8 / 50** | step-skipping (compile-avoidance ≈ noise) |
| `flux2-klein-4b`, `flux2-klein-9b`⁴ | 8 | — | — | — | **0 / 8** | no skips; any wall-clock difference is `mx.compile` avoidance, only where mflux compiles `_predict` |
| `z-image-base` (CFG)⁵ | 50 | 227.4s | 174.2s | **1.31×** | **15 / 50** | step-skipping (q8; compile-avoidance not a tailwind) |
| `flux1-dev` | 25 | 113.1s | 71.9s | **1.57×** | **6 / 25** | TeaCache step-skipping⁶ |
| `flux1-krea-dev`⁷ | 28 | 125.3s | 77.2s | **1.62×** | **10 / 28** | step-skipping (1.57× gating; the no-gate wrapper's 1.03× is noise, FLUX.1 has no compile path) |
| `flux1-schnell` | — | — | — | — | — | shares dev's coefficients; gate behaves like dev at long schedules, like Klein at the 4-step distilled default (no benefit) |

¹ `qwen-image` under CFG at 50 steps, `guidance=4.0`, **768×768** (its pinned recipe — the rest of the table is 512×512), q4, per-variant default `rel_l1_thresh=0.30`. Measured 2026-09-06 under the subprocess-per-rep harness (mflux 0.18.0, chunks interleaved rep-outer, medians of three cold reps): **2.68× combined** against vanilla (642.6 s → gated 239.4 s), the largest gain in this table, and **2.57×** against the wrapper's own no-gate run (615.6 s → 239.4 s). The no-gate wrapper's 1.04× over vanilla is inside the spread of the vanilla reps themselves (610.9 / 642.6 / 715.9 s), so what 0.10.0 reported as an unexplained 1.10× reads as run-to-run noise on this harness; mflux does not `mx.compile` Qwen's `_predict`, so there is no compiled path to avoid. 33 of the 48 active steps skipped in every rep (the table's `33 / 50` counts nominal steps), longest run of consecutive skips 4, the same per-step pattern in all three reps and the same one the threshold sweep and the committed calibration replay produce. SSIM 0.967 against vanilla at this recipe, from the sweep on the variant page; 0.9.x skipped 24 here at SSIM 0.978, and `rel_l1_thresh=0.20` returns to that point. The bench and the sweep now free Qwen's text encoders once the prompt is encoded (mflux's `MemorySaver`; the report records `memory_saver`), which brings the loop peak from 26.2 GB to 24.1 GB in every condition, under this machine's 25.0 GB recommended working set, with the cache pool held at 1.1 GB; the worker still aborts with an artifact if active plus cached MLX memory exceeds physical memory minus 4 GiB. The 0.10.0 numbers (850.3 s vanilla, 3.02×) were taken with the loop peak above that working set; this run stays under it, and the vanilla time fell more than the gated one, which is where the ratio moved. The 0.10.0 report stays committed as `_artifacts/v0.10.0_bench_qwen_image.json`. This row is uniform 4-bit on mflux 0.18.0. From mflux 0.19 the 4-bit build keeps the image-stream modulation layers (`img_mod_linear`) at 8 bits (about 1.8 GB more memory), so a re-run on a newer mflux is not the same recipe. Full report: `_artifacts/v0.11.0_bench_qwen_image.json`. Reproduce with `uv run python scripts/bench_speedup.py --variant qwen --three-way --reps 3 --report out.json`.

² `flux2-klein-base-9b` under CFG at the canonical 50-step + g=4.0 recipe. Measured 2026-09-28 under the subprocess-per-rep harness (macOS 27.0, mflux 0.18.0, MLX 0.31.2, v0.12.0 gate, chunks interleaved rep-outer): **1.37× combined** = 1.35× gating × 1.02× compile-avoidance (vanilla 517.3 s, no-gate 508.3 s, gated 377.5 s medians). 13/50 skips in every rep, never two consecutive, the same steps v0.10.0 skipped, so clamping small changes to the calibrated range changes nothing at this recipe. This bench ran before the rule that always computes a change above the range was added, and that rule was not traced on 9B; on 4B at the same recipe no priced change came near the top of the range. SSIM 0.993 vs vanilla, computed the same way as in footnote ³. v0.10.0 measured 1.37× (520.6 s → 379.1 s, `_artifacts/v0.10.0_bench_klein_base_9b.json`) and v0.6.0 1.36× (`_artifacts/v0.6.0_bench_klein_base_9b.json`); all three replace v0.5.0's advertised 2.68×, which was inflated by same-process MLX state leakage in the v0.5.x harness (see [Per-variant notes](#per-variant-notes) → klein-base-9b correction blockquote). Full report: `_artifacts/v0.12.0_bench_klein_base_9b.json`. Wrapper peak memory ~9.8 GB vs vanilla's ~22 GB. Reproduce with `uv run python scripts/bench_speedup.py --variant klein-base-9b --three-way --reps 3 --report out.json`.

³ `flux2-klein-base-4b` under CFG at the canonical upstream BFL recipe (`guidance=4.0`, 50 steps), per-variant default `rel_l1_thresh=0.17`. Measured 2026-09-28 under the subprocess-per-rep harness (macOS 27.0, mflux 0.18.0, MLX 0.31.2, v0.12.0 gate, chunks interleaved rep-outer): **1.20× combined** = 1.18× from step-skipping (wrapped-no-gate vs wrapped-gated) × 1.01× from `mx.compile`-path avoidance (vanilla vs wrapped-no-gate); vanilla 237.5 s, no-gate 234.2 s, gated 197.8 s medians. 8 of the 48 active steps skipped in every rep, never two in a row; SSIM 0.985 against vanilla (scikit-image `structural_similarity` on the bench's rep-0 vanilla and gated images, `data_range=255`; recorded with the image hashes in `_artifacts/v0.12.0_klein_base_ssim.json`). Until v0.12.0 the gate also skipped step 6. Its input change was smaller than any step in the calibration, and the fit, which passes through zero, priced it as almost no change. v0.10.0 measured 1.22× with that extra skip (9/50, longest run 2; `_artifacts/v0.10.0_bench_klein_base_4b.json`, macOS 26) and v0.6.0 1.23× (`_artifacts/v0.6.0_bench_klein_base_4b.json`). Between the v0.10.0 and v0.12.0 runs the machine moved from macOS 26 to 27, and vanilla itself slowed by 1.5% (233.9 s to 237.5 s), so the gated time's 2.8% rise (5.5 s) is part host drift and part the one extra computed step. Compile-avoidance is effectively noise on this recipe; the v0.4.1-era 1.09× attribution was inflated by same-process MLX state leakage (vanilla ran cold, the wrapper inherited warm allocator state). Reproduce with `uv run python scripts/bench_speedup.py --variant klein-base-4b --three-way --reps 3 --report out.json`. Full report: `_artifacts/v0.12.0_bench_klein_base_4b.json`.

⁴ The distilled Klein variants (`flux2-klein-4b`, `flux2-klein-9b`) run an 8-step schedule where the gate skips nothing at the package default. They have no committed multi-run timing under the current harness, so the table shows none. Compile avoidance applies only on Macs where mflux compiles `_predict`; on base and Pro M1/M2 mflux already runs the eager path and there is nothing to avoid. Where it does apply, it measured 1.01× to 1.02× on the 50-step Klein base recipes (footnotes ² and ³), and a short 8-step schedule may show more, which is why the docs do not quote a figure for it.

⁵ `z-image-base` under CFG at 50 steps, `guidance=4.0`, 512×512, **q8** (its pinned recipe — the rest of the table is q4), per-variant default `rel_l1_thresh=0.12`. Measured 2026-08-15 under the subprocess-per-rep harness (mflux 0.18.0, v0.10.0 gate): **1.31× combined** (vanilla 227.4 s → wrapper 174.2 s median), entirely from step-skipping at 15 of the 48 active steps (the table's `15 / 50` is over nominal steps; 48 active = 50 minus the skip-first/skip-last windows), the same 15 in every rep and never two consecutive. `mx.compile`-path avoidance is not a tailwind here: the no-gate wrapper ran at 228.9 s median against vanilla's 227.4 s (0.99×), so the whole 1.31× is gating. Peak memory 17.2 GB → 11.5 GB, from the eager wrapper bypassing mflux's compiled `_predict` (the no-gate wrapper shows the same ~11.5 GB), not from gating. SSIM 0.991 vs vanilla, from the threshold sweep on the variant page. v0.7.0 reported 1.17× at this recipe (`_artifacts/_bench_z_image_v0_7_0.json`, wrapper 209.4 s) with a thermally confounded three-way split; the skip count and pattern are identical, so the difference is host state, not the gate. Full report: `_artifacts/v0.10.0_bench_z_image.json`. Reproduce with `uv run python scripts/bench_speedup.py --variant z-image --three-way --reps 3 --report out.json`.

⁶ `flux1-dev` at 25 steps, `guidance=3.5`, 512×512, default `rel_l1_thresh=0.20`. Measured 2026-08-15 under the subprocess-per-rep harness (mflux 0.18.0, v0.10.0 gate): **1.57× combined** as the median of three cold reps (vanilla 113.1 s → gated 71.9 s), all of it from step-skipping. The no-gate wrapper timed 1.11× ahead of vanilla this session, but mflux does not compile the FLUX.1 predict step, so that gap is not compile avoidance; the vanilla reps' 15 s spread is the likely cause. One caveat on the split: the vanilla reps were spread wide this session (104.7 / 113.1 / 119.8 s) while the gated wrapper held steady (69.8 / 71.9 / 73.1 s), the same 71 s v0.6.3 measured. Comparing fastest to fastest gives 1.50×, so 1.5× is the conservative headline. The earlier 1.46× (v0.6.3, vanilla 103.8 s) came from a session with a quicker vanilla baseline, not from a slower wrapper. 6/25 skips in every rep, never two consecutive. Full report: `_artifacts/v0.10.0_bench_flux1_dev.json` (v0.6.3's is kept alongside as `_artifacts/v0.6.3_bench_flux1_dev.json`). Reproduce with `uv run python scripts/bench_speedup.py --variant flux1-dev --three-way --reps 3 --report out.json`.

⁷ `flux1-krea-dev` at 28 steps, `guidance=4.5`, 512×512, q4, per-variant default `rel_l1_thresh=0.30`. Measured 2026-09-05 under the subprocess-per-rep harness (mflux 0.18.0): **1.62× combined** as the median of three cold reps (vanilla 125.3 s → gated 77.2 s), 1.57× from gating; the no-gate wrapper timed 1.03× ahead of vanilla, and since mflux does not `mx.compile` the FLUX.1 predict step that residual is run-to-run noise, not compile avoidance. 10 of the 26 active steps skipped in every rep (the table's `10 / 28` counts nominal steps), never two in a row. SSIM 0.990 against vanilla at this recipe, from the threshold sweep documented on the variant page; 0.35 already falls to 0.890, which is why the default sits exactly at the knee. Peak memory 11.1 GB. Full report: `_artifacts/v0.11.0_bench_krea_dev.json`. Reproduce with `uv run python scripts/bench_speedup.py --variant krea-dev --three-way --reps 3 --report out.json`.

Reproduce any row:

```bash
uv run python scripts/bench_speedup.py --variant qwen           # 50-step Qwen-Image under CFG (g=4.0, 768x768)
uv run python scripts/bench_speedup.py --variant klein-base-9b  # 50-step base-9B under CFG (g=4.0, v0.5.0+ default)
uv run python scripts/bench_speedup.py --variant klein-base-4b  # 50-step base-4B under CFG (g=4.0, v0.4.1+ default)
uv run python scripts/bench_speedup.py --variant klein-base-4b --guidance 1.0 --num-inference-steps 25  # v0.4.0 row
uv run python scripts/bench_speedup.py --variant klein-4b       # 8-step Klein 4B
uv run python scripts/bench_speedup.py --variant klein-9b       # 8-step Klein 9B
uv run python scripts/bench_speedup.py --variant z-image        # 50-step Z-Image base under CFG (g=4.0, q8)
uv run python scripts/bench_speedup.py --variant flux1-dev      # 25-step dev
uv run python scripts/bench_speedup.py --variant krea-dev       # 28-step FLUX.1 Krea [dev] (g=4.5)
```

For the three-way decomposition (vanilla / wrapped-no-gate / wrapped-gated), add `--three-way --reps 3 --report out.json`. This is how the v0.10.0 rows above were produced.

### How the speedup happens

The wall-clock improvement above comes from two distinct mechanisms; they fire independently depending on variant and schedule.

**1. TeaCache step-skipping.** This is the headline feature. The polynomial gate predicts how much the transformer body output will change from the immediately previous step, whether or not that step was itself skipped. The wrapper accumulates the predicted change across consecutive skips, and once that running total would cross `rel_l1_thresh`, it recomputes the transformer body instead of reusing the cached residual and resets the running total. On FLUX.1-dev at 25 steps, 6 of 25 steps are skippable and this is where the 1.57× speedup on FLUX.1-dev comes from. mflux does not compile the FLUX.1 predict step, so none of that speedup is compile avoidance. On non-distilled FLUX.2 Klein at 50 steps + CFG, the same mechanism produces 8 to 13 skips per generation and dominates the wall-clock win (the v0.12.0 three-way bench attributes 1.35× to gating on base-9b, footnote ²).

**2. `mx.compile` avoidance on FLUX.2 and Z-Image.** mflux wraps `Flux2Klein._predict` (and Z-Image's, behind the same chip check) in `mx.compile` on every chip *except* base + Pro M1/M2 — i.e., compilation is active on M1/M2 Max + Ultra and on every M3, M4, M5 chip. mlx-teacache replaces the compiled `_predict` with an eager Python closure so the gate can run live per step. The magnitude of this effect varies by schedule and chip. On the 50-step Klein base CFG recipes the subprocess-per-rep bench measures it at 1.01× to 1.02×: kernel-dispatch round-trips drop slightly, but with 50 steps per generation that gain is small relative to the per-step compute. Short distilled schedules may benefit more because each step is a larger share of the wall-clock, but this repository has no committed multi-run measurement of that, so no figure is quoted. On chips where mflux is already eager (base + Pro M1/M2), this mechanism does not fire: the wrapper just adds per-step gate overhead, and Klein with mlx-teacache on those chips is approximately neutral or slightly slower than vanilla.

On FLUX.2 Klein 4B and 9B at the distilled 4-8 step defaults, mechanism (1) does not engage: the empirical adjacent-step rel-L1 between consecutive transformer outputs is ≥ 0.25, so every step's predicted change exceeds the default 0.20 threshold and the gate signals "compute" every time. This is expected — distilled schedules collapse the entire denoising trajectory into a handful of consequential steps, so adjacent steps are not similar enough to skip. The gate skips nothing here, so whatever speed difference you see is mechanism (2), which only exists on Macs where mflux compiles `_predict`.

For algorithmic step-skipping on FLUX.2, use the non-distilled `flux2-klein-base-4b` (Apache-2.0) or `flux2-klein-base-9b` (FLUX Non-Commercial) variants. Both ship with a per-variant default `rel_l1_thresh=0.17`. At the canonical upstream 50-step + g=4.0 CFG recipe, base-4b skips 8/50 steps for a measured 1.20× combined speedup (subprocess-per-rep), and base-9b skips 13/50 steps for 1.37×. Pushing the threshold higher on distilled Klein is not recommended: the gate's prediction quality at thresholds > 0.25 is uncalibrated on a 4-8 step trajectory and image quality is not characterized there.

### SSIM suite

Quality gates use a 5-prompt SSIM suite defined at `tests/test_image_quality_flux1.py:45` and reused at `tests/test_image_quality_flux2.py:28`:

- "a red apple on a wooden table"
- "mountain landscape at sunset"
- "portrait of a woman"
- "abstract pattern with circles"
- "text saying HELLO"

The PR-gate prompt is the red-apple one; SSIM ≥ 0.90 on FLUX.1-dev and ≥ 0.85 on Klein 4B / 9B at the default threshold. Full suite floor is 0.80 to absorb high-frequency-detail variance (text, synthetic patterns). Run `uv run pytest tests/test_image_quality_flux1.py tests/test_image_quality_flux2.py -m parity` with real model weights to reproduce.

## Performance by chip

mflux wraps `_predict` in `mx.compile` on every Apple Silicon chip *except* base + Pro M1/M2 (the behaviour is unchanged from mflux 0.17.5 through 0.23). The `is_m1_or_m2()` predicate returns true (eager path) when the chip brand contains "Apple M1" or "Apple M2" *and* does not contain "Max" or "Ultra" — so M1 Pro and M2 Pro are eager too, while M1/M2 Max + Ultra and every M3/M4/M5 chip get the compiled path. mlx-teacache replaces `_predict` with an eager closure so per-step gating stays live, trading the compile gain for the skip gain on compiled chips. That compile-avoidance effect applies to FLUX.2 Klein and Z-Image (mflux compiles Z-Image's `_predict` behind the same chip check); FLUX.1, FLUX.1 Krea, and Qwen-Image have no `mx.compile` path, so their step-skipping speedup is the same on every chip. See `docs/m3-plus-tradeoff.md` for a benchmark recipe.

| Chip | Vanilla `_predict` in mflux | Expected speedup |
|---|---|---|
| Apple M1 / M2 (base) | eager | the model's skip fraction, chip-independent (2.68× Qwen-Image, 1.62× Krea, 1.57× FLUX.1-dev); FLUX.2 loses its small compile-avoidance share here |
| M1 Pro / M2 Pro | eager | same as base — the skip fraction, chip-independent |
| M1 Max / Ultra, M2 Max / Ultra | compiled | **measured on M1 Max:** Qwen-Image 2.68×, klein-base-9b 1.37×, klein-base-4b 1.20×, z-image-base 1.31×, FLUX.1 Krea 1.62×, FLUX.1-dev 1.57× (see [Benchmarks](#benchmarks)) |
| M3 / M3 Pro / M3 Max / Ultra | compiled | FLUX.1 / Qwen keep their skip fraction; FLUX.2 likely 1.1–1.3× — untested |
| M4 / M4 Pro / M4 Max | compiled | FLUX.1 / Qwen keep their skip fraction; FLUX.2 likely 1.1–1.3× — untested |
| M5+ (Neural Accelerators / TensorOps) | compiled + accelerator | FLUX.2 may approach 1.0× — the eager wrapper can lose some or all of the M5 TensorOps advantage; confirm with a profiler. FLUX.1 / Qwen stay skip-limited |

## Limitations

img2img reuses the txt2img calibration. A dedicated img2img calibration may follow in a future release if SSIM gates flag drift on specific schedules.

FLUX.2 with CFG (`guidance > 1.0`) runs through the gated path as of v0.4.1. The wrapper keeps two cached residuals (positive and negative branch) and shares one gate decision per step across both. The canonical base-4b recipe (`guidance_scale=4.0, num_inference_steps=50`) measures **1.20× combined** on M1 Max under the subprocess-per-rep harness (v0.12.0; v0.10.0 measured 1.22× and v0.6.0 1.23× before the gate's range check, v0.4.1 advertised 1.26× combined under same-process measurement; combined within day-to-day noise, but the decomposition shifted — see Benchmarks footnote ³).

**Distilled schedules are out of scope for algorithmic step-skipping by design.** This includes FLUX.2 Klein 4B / 9B at their 4-8 step defaults and FLUX.1 schnell at its 4-step default. The polynomial gate's premise — that consecutive transformer outputs are similar enough that the residual can be reused — does not hold on distilled trajectories where each step does a much larger share of the denoising work. On the v0.3.0 bench (M1 Max, quantize=4) the gate signals "compute" on every Klein step at the package default `rel_l1_thresh=0.20` (0 skips across 3 reps on both Klein 4B and 9B); empirical adjacent-step body-output rel-L1 on Klein is ≥ 0.25. Where mflux compiles `_predict` (not base or Pro M1/M2), Klein can still run a little faster from `mx.compile`-path avoidance, but the headline TeaCache step-skipping feature only fires on non-distilled schedules.

`flux2-klein-base-4b` runs TeaCache at both `guidance=1.0` (single-branch path) and `guidance > 1.0` (per-branch path, v0.4.1+). The upstream BFL base-4b model card recommends `guidance_scale=4.0, num_inference_steps=50`; the subprocess-per-rep harness measures **1.20× combined wall-clock** vs vanilla on M1 Max at that recipe (v0.12.0: 1.18× gating + 1.01× compile-avoidance; 8/50 skips, SSIM 0.985; v0.10.0 measured 1.22× and v0.6.0 1.23×). v0.4.1's same-process bench reported 1.26× combined and decomposed it 1.16× gating / 1.09× compile-avoidance; the combined was honest within day-to-day noise but the decomposition over-attributed to compile-avoidance.

`flux2-klein-base-9b` reuses base-4b's polynomial coefficients verbatim (same architecture family, same calibration recipe). The v0.12.0 subprocess-per-rep bench at the canonical 50-step / guidance=4.0 recipe measures **1.37× combined wall-clock** (1.35× gating + 1.02× compile-avoidance; v0.10.0 and v0.6.0 measured 1.37× and 1.36×), 13/50 steps skipped at `rel_l1_thresh=0.17`, SSIM 0.993 vs vanilla. v0.5.0 advertised 2.68× on this variant — that number was inflated by same-process MLX state leakage and is corrected here. Same FLUX Non-Commercial license + BFL safety-filter obligations as `flux2-klein-9b` — see [License obligations](#license-obligations).

The wrapper runs eager, which gives up mflux's `mx.compile` of `_predict` in exchange for live per-step gating. Vanilla mflux compiles `_predict` on every chip except base + Pro M1/M2 (the `is_m1_or_m2()` predicate is true for base and Pro M1/M2, which run eager; Max, Ultra and M3 and newer compile). The 1.57× measurement is from M1 Max / FLUX.1-dev / 25 steps; speedup on M3 and newer is plausible but untested locally. On M5, the GPU Neural Accelerators (Metal 4 TensorOps) are only reachable through the compiled path, so the eager wrapper can lose some or all of that advantage. Output stays correct either way. See `docs/m3-plus-tradeoff.md` for the per-chip recipe; PRs with measurements welcome.

FLUX.2 parity is numerical, not bit-exact. Replacing a function that mflux wraps in `mx.compile` produces about 1 ULP per element of divergence from Metal kernel-dispatch noise, which compounds across steps but keeps cosine similarity ≥ 0.97 on Klein 4B, Klein 9B, and base-4b under CFG at threshold 0. The user-facing guarantee is end-to-end image quality (SSIM ≥ 0.85 on all supported FLUX.2 variants at the package default threshold).

Fingerprints are recorded for mflux 0.17.5, 0.18.0, 0.18.1, 0.19.1, 0.19.2, 0.20.0, 0.21.0, 0.22.0 and 0.23.0, and the `[mflux]` range spans them; 0.21.0 was also checked on real weights for Z-Image. For 0.22.0 the checks are the fingerprints, a weight-free Qwen-Image test and a real-weights check of FLUX.1-dev, which is bit-identical to plain mflux at threshold 0. 0.23.0 changed none of the functions TeaCache copies; on it, FLUX.1-dev and Qwen-Image-2512 were run on real weights to check that the transformer is freed under mflux's memory saver, and Qwen-Image-2512's own calibration and threshold sweep also ran on 0.23.0. The range also allows the next two minor versions, so mflux 0.24 and 0.25 install without waiting for a release here. On an mflux newer than 0.23.0 (0.23.1 included), `apply_teacache` emits a `TeaCacheUntestedMfluxWarning` once per process and still patches the model; if a generation fails or looks wrong there, please open an issue with both version numbers.

On FLUX.1 and Qwen-Image the wrapper replaces `flux.transformer` with a proxy that the parent model's parameter tree does not reach. Calling `flux.parameters()`, `nn.quantize(flux)` or `flux.update(...)` at the parent level can therefore miss the transformer while the wrapper is active. Use `flux.transformer.parameters()` directly, or call `handle.restore()` first. Calls on the transformer itself work: `flux.transformer.load_weights(...)`, `set_dtype`, `eval()` and `nn.quantize(flux.transformer)`. When mflux's memory saver drops the transformer after the denoising loop (`model.transformer = None`), TeaCache keeps no copy of it, and a later `handle.restore()` leaves `flux.transformer` as `None` while undoing the rest of the patch, the same state plain mflux leaves.

## License obligations

The FLUX.1 variants (`flux1-dev`, `flux1-schnell`) and `flux2-klein-4b` come with their own upstream weight licenses; the wrapper this library applies does not change those terms.

`flux2-klein-9b` and `flux2-klein-base-9b` are both distributed under the FLUX.2 Klein license (non-commercial use + BFL safety-filter obligations). These terms flow with the weights, not with mlx-teacache. If you call `apply_teacache` on either variant — `Flux2Klein(model_config=ModelConfig.flux2_klein_9b())` or `Flux2Klein(model_config=ModelConfig.flux2_klein_base_9b())` — you are responsible for ensuring your use complies with the upstream license, including the safety-filter requirements the BFL model cards describe. See the official model cards at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B and https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B for the full terms.

## Research notes

**[Qwen-Image mixed precision on a 32 GB Mac](https://ineshin.space/papers/qwen-image-mixed-precision-on-a-32-gb-mac/)**
documents a uniform-q4 portrait artifact and the mixed q8/q4/bf16 recipe that removed it in one
controlled pair for about 1.9 GiB of additional peak MLX allocation. The recipe changes model
construction, not TeaCache.

**[Why the TeaCache gate did not engage on short distilled FLUX schedules](https://ineshin.space/papers/why-teacache-does-not-engage-on-short-distilled-schedules/)**
documents the zero-skip negative result on the 8-step distilled Klein schedules and the
measurement practice it forced: skip counts published next to every wall-clock number, so a
speedup cannot be mis-attributed to caching.

**[Why byte-exact parity is a poor MLX integration oracle](https://ineshin.space/papers/why-byte-exact-parity-is-a-poor-mlx-integration-oracle/)**
uses a pre-release FLUX.1 fixture failure to separate artifact drift, no-op parity, numerical
equivalence, and intentionally approximate output. It explains why paired same-process tests and
measured per-variant gates are more informative than committed real-model bytes.

## Contributing

Open an issue at https://github.com/IonDen/mlx-teacache/issues. If you're adding a variant or changing the gate, [`docs/architecture.md`](docs/architecture.md) is the contributor's map and [`docs/calibration.md`](docs/calibration.md) covers coefficient calibration.

## License + acknowledgements

Apache-2.0. See `LICENSE` and `NOTICE`.

- [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) — upstream method and FLUX.1 coefficients.
- [mflux-community/mflux](https://github.com/mflux-community/mflux) — the MLX diffusion runner this library integrates with (moved from `filipstrand/mflux` in August 2026; the old URL redirects).
- [Apple ML Explore](https://github.com/ml-explore/mlx) — MLX.

## Sister projects

Other MLX libraries for Apple Silicon:

- [mlx-taef](https://github.com/IonDen/mlx-taef) — tiny autoencoders for fast diffusion-latent previews and low-memory decode (FLUX / SD).
- [mlx-model-doctor](https://github.com/IonDen/mlx-model-doctor) — validate an MLX / Hugging Face model repo before you load it (config, tokenizer, safetensors, memory).
- [mlx-quant-fidelity](https://github.com/IonDen/mlx-quant-fidelity) — measure how much quality a quantization costs (KL divergence, top-token flips, perplexity delta).

---

By Denis Ineshin · [ineshin.space](https://ineshin.space)
