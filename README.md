# mlx-teacache

<p align="center">
  <img src="https://raw.githubusercontent.com/IonDen/mlx-teacache/main/docs/assets/mlx-teacache-logo.png" alt="mlx-teacache" width="100%">
</p>

[![PyPI version](https://img.shields.io/pypi/v/mlx-teacache.svg)](https://pypi.org/project/mlx-teacache/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://pypi.org/project/mlx-teacache/)
[![License: Apache 2.0](https://img.shields.io/pypi/l/mlx-teacache.svg)](https://github.com/IonDen/mlx-teacache/blob/main/LICENSE)
[![CI](https://github.com/IonDen/mlx-teacache/actions/workflows/ci.yml/badge.svg)](https://github.com/IonDen/mlx-teacache/actions/workflows/ci.yml)

**Faster image generation on Apple Silicon: TeaCache skips the denoising steps that barely change the picture. For Qwen-Image, FLUX.2 Klein base, Z-Image and FLUX.1 on mflux.**

mlx-teacache makes image generation on an Apple Silicon Mac faster. It plugs into [mflux](https://github.com/mflux-community/mflux), which runs open image models on MLX, Apple's machine-learning framework. You wrap the model in one call, `apply_teacache(flux)`, and keep your prompt, seed and settings. At each model's default setting the image stays very close to what plain mflux makes; it just arrives sooner. Nothing is retrained and nothing extra is downloaded.

Measured on an M1 Max with 32 GB, against plain mflux, at each model's default threshold. The FLUX.1 dev row was measured on mflux 0.24.0 and MLX 0.32.2; the other rows on mflux 0.18.0. Qwen-Image ran at 768×768 with 4-bit weights, Z-Image base at 512×512 with 8-bit weights, the others at 512×512 with 4-bit weights. No other row's record names its MLX version; the lock files at the commits involved resolve MLX 0.31.2. Each figure carries a marker for its note under [Benchmarks](#benchmarks), which gives the record and says what that record does and does not name (for example uncommitted changes, or no date). A record is the results file a bench or sweep script wrote under `_artifacts/`. The Qwen-Image note is the first:

| Model | Speedup | Steps skipped |
|---|---|---|
| Qwen-Image (original checkpoint) | 2.68×¹ | 33 of 50 |
| FLUX.2 Klein base 9B | 1.37×² | 13 of 50 |
| FLUX.2 Klein base 4B | 1.20×³ | 8 of 50 |
| Z-Image base | 1.31×⁵ | 15 of 50 |
| FLUX.1 Krea [dev] | 1.62×⁷ (above its 1.56× limit) | 10 of 28 |
| FLUX.1 dev | 1.30×⁶ | 6 of 25 |

Skipped steps set a limit on each speedup: the number of steps divided by the number still computed. FLUX.1 dev's 1.30×⁶ comes entirely from the 6 skipped steps, and the wrapper alone gains nothing on mflux 0.24: the no-gate run, which computes every step, took the same time as plain mflux (89.46 s both). It is under its 1.32× limit (footnote ⁶ has the arithmetic). An older 1.57× from mflux 0.18.0 was above that limit and is kept only as history in the same footnote. Krea's 1.62×⁷ is above its 1.56× limit (footnote ⁷).

Each speedup is the median of three runs, each in a fresh process; [Benchmarks](#benchmarks) has the full recipe and a command to reproduce every row. Image similarity to plain mflux (SSIM, where 1.0 means identical) was 0.985 and 0.993 for the two Klein base runs (recorded in `_artifacts/v0.12.0_klein_base_ssim.json`). The threshold sweeps for Qwen-Image, Qwen-Image-2512, Z-Image and Krea also measured image similarity. Each has a record under `_artifacts/sweeps/`, and the per-variant notes below give the figures. FLUX.1 dev passes the test suite's SSIM floor of 0.80 across five prompts. On mflux 0.19 and later the `qwen-image` name loads Qwen-Image-2512, which has its own calibration but no speed measurement yet.

It does not help *distilled* models, which are trained to finish in 4 to 8 steps: FLUX.1 schnell and FLUX.2 Klein 4B and 9B (the ones without "base" in the name). Each of their few steps changes the picture a lot, so there is nothing safe to skip. Run those with plain mflux; `apply_teacache` warns on them. On Max, Ultra and M3-or-newer chips the wrapper for FLUX.2 Klein and Z-Image also sidesteps mflux's compiled prediction step, a separate small effect on speed; [How the speedup happens](#how-the-speedup-happens) explains it.

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
pip install "mlx-teacache[mflux]==0.18.1"  # pin for reproducibility
```

Requires Python ≥ 3.11 and Apple Silicon. The `[mflux]` extra pulls in `mflux>=0.24`. Version notes:

- mlx-teacache 0.18.1 needs Python 3.11 or newer and mflux 0.24 or newer; there is no upper bound on mflux. [docs/stability.md](docs/stability.md) describes what 1.0 will promise. A newer mflux installs and is patched, but has not been tested; `apply_teacache` warns until it is. mflux 0.24.0 itself needs Python 3.11 and Pillow 13, which fixes a critical Pillow security advisory ([GHSA-5x8w-8jj5-q52h](https://github.com/advisories/GHSA-5x8w-8jj5-q52h); Pillow 12.3 and older are affected), so an install with the `[mflux]` extra gets the fixed Pillow through mflux. The base install pulls in neither mflux nor Pillow.
- Python 3.10 and mflux 0.17.5 through 0.23.x are no longer supported. If you need either, stay on the 0.14 line: `pip install "mlx-teacache[mflux]<0.15"`. Its last release, 0.14.1, still runs on mflux 0.24 but prints the untested-version warning there.
- mflux 0.24 needs MLX 0.32 (below 0.33), torch 2.13 or newer, opencv-python, Pillow 13 or newer and Python 3.11 or newer.
- For mlx-taef's live preview, use mlx-taef 0.9.0 or later. Its `[mflux]` extra supports mflux 0.24, so `mlx-teacache[mflux]` and `mlx-taef[mflux]>=0.9.0` install together.
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

On an M1 Max this takes about three to four minutes (Benchmarks footnote ³ has the Klein base 4B timings). Every supported model is wrapped the same way: build it with mflux, then call `apply_teacache`. Each page under [`docs/variants/`](docs/variants) gives the mflux constructor and the recipe. Qwen-Image is the heavy one: at 4-bit and 768×768 it uses about all the memory a 32 GB Mac can give the GPU (see the Qwen-Image-2512 note under Supported models).

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
| 0.10 | none | Not recorded; with no step skipped, the cache gives no gain | No step skipped; no similarity recorded | Cache never engages |
| 0.15 | none | Not recorded; with no step skipped, the cache gives no gain | No step skipped; no similarity recorded | Cache never engages |
| **0.20 (default)**, mflux 0.24.0 | **6 / 25** | **1.30×** (`_artifacts/v0.18.0_bench_flux1_dev.json`) | **≥ 0.80 (test-suite floor, 5 prompts; not measured in this run)** | **Visually-lossless sweet spot** |
| 0.20 (default), mflux 0.18.0 | 6 / 25 | 1.57× (`_artifacts/v0.10.0_bench_flux1_dev.json`), above its 1.32× limit (footnote ⁶) | ≥ 0.80 (test-suite floor, 5 prompts; not measured in this run) | Same skipped steps as the row above |
| 0.25 | not recorded | Not recorded | Not recorded | Visible style changes on text/synthetic prompts |

The two committed benchmarks, `_artifacts/v0.18.0_bench_flux1_dev.json` (mflux 0.24.0, 2026-11-16) and `_artifacts/v0.10.0_bench_flux1_dev.json` (mflux 0.18.0, 2026-08-15), each give the median of three cold reps for the 0.20 default. No record of the single runs behind the 0.10, 0.15 and 0.25 rows was committed, so the table gives no speedup or similarity for them.

0.20 was picked after side-by-side visual comparison. At 0.25, text prompts that vanilla renders as neon tubes can come out as dot-matrix. At 0.20, the output is indistinguishable from vanilla and the cache still skips around 25% of steps (6 of 25, Benchmarks footnote ⁶). SSIM is conservative on high-frequency-detail prompts like text and synthetic patterns, which is why the suite floor (0.80) is lower than the PR-gate floor (0.90) on the red-apple prompt.

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

Each variant has its own page under [`docs/variants/`](docs/variants) — mflux constructor, recipe, license obligations, coefficient provenance, quirks. The highlights for the variants where behavior diverges from the default story. Where a figure comes from a threshold sweep, the entry names its record under `_artifacts/sweeps/`. Each record's date is the file time on the machine when the record was made (the file name carries it; git keeps no file times), and none names an MLX or mflux version, the hardware, a commit or a flag for uncommitted changes; [`docs/calibration.md`](docs/calibration.md#threshold-sweep-records) lists what each record leaves out.

**[`qwen-image-2512`](docs/variants/qwen-image-2512.md)** — Qwen-Image-2512 (Alibaba, Apache-2.0), the newer checkpoint that mflux 0.19 and later load under the `qwen-image` alias. It runs the same proxy forward and gate signal as `qwen-image` (next entry), but the original checkpoint's coefficients do not fit it (R² −0.35 on 2512's calibration data), so it has its own fit: R² 0.818, held-out 0.785, ten prompts at the 768×768, 50-step, guidance 4.0, 4-bit recipe on mflux 0.23.0. Per-variant default `rel_l1_thresh=0.25`, set from its threshold sweep on the red-apple prompt at that recipe: 29 of 48 active steps skipped at SSIM 0.969 against plain mflux, longest run of skips 4. The original checkpoint's 0.30 falls to SSIM 0.947 on 2512. Both figures come from `_artifacts/sweeps/qwen-image-2512_plain-q4_2026-10-20.json`. The sweep ran on a working commit that was not kept, the record names no mflux version (`uv.lock` at b883378 pins 0.23.0), and the images behind the scores are not published. There is no speed benchmark for this variant yet. On mflux 0.19 and later, original Qwen-Image weights loaded through the command line, a `model_path` or a mirror resolve against the 2512 config and are detected as 2512, because that config keeps no trace of the original. To run them on the original calibration, pass `coefficients=` with the original's tuple and `rel_l1_thresh=0.30`; the [variant page](docs/variants/qwen-image-2512.md) has the details. It also covers the memory needed: each calibration prompt peaked above the working set of a 32 GB M1 Max. The calibration record, `scripts/_calibration_qwen_2512.json`, is committed but does not hold that peak.

**[`qwen-image`](docs/variants/qwen-image.md)** — Qwen-Image base (Alibaba, Apache-2.0), a ~20B dual-stream MMDiT and the first variant that proxies `flux.transformer` (the FLUX.1 pattern) *and* runs true two-pass CFG. It's FLUX-shaped, so the gate taps the FLUX-canonical modulated block-0 input, calibrated in-repo at R² 0.849 (well above Z-Image's 0.400 and the FLUX.2 family's 0.11–0.47). Per-variant default `rel_l1_thresh=0.30`, set from the threshold sweep. At the red-apple 768×768 recipe it skips 33 of 48 active steps for **2.68× combined** (642.6 s → 239.4 s, Benchmarks footnote ¹), SSIM 0.967 vs vanilla — the largest gain of any variant, almost all of it step-skipping (Qwen has no `mx.compile` path). The SSIM and the skip count come from a threshold sweep whose record is `_artifacts/sweeps/qwen-image_plain-q4_2026-09-06.json`. `uv.lock` at 3b86c8b pins mflux 0.18.0 and MLX 0.31.2. mflux 0.22.0 loads the bias of Qwen-Image's last normalization layer again, and its release notes say the images change visibly, so these scores hold for mflux 0.18.0. The plain-mflux reference image comes from an earlier attempt that was stopped when the host ran out of memory, and the option that frees the text encoders was added to the script during the run, so the reference and the 0.15 run predate it. The images behind the scores are not published. Lowering the threshold to 0.20 gives back a higher-fidelity operating point (26 skips at SSIM 0.980 in the same record); the variant page has the whole sweep, and its [benchmark section](docs/variants/qwen-image.md#benchmark) has the measurement history. **From mflux 0.19 on, the `qwen-image` alias loads `Qwen/Qwen-Image-2512`**, which has its own variant (above). Every number in this entry is the original checkpoint on mflux 0.18.0. On mflux 0.19 and later you get the original only from Python, by passing `ModelConfig.from_name("Qwen/Qwen-Image")` as the `model_config`; mflux's command line and any `model_path`, such as a local directory, resolve to Qwen-Image-2512. mflux 0.20's separate Qwen-Image-2.1 (Qwen Research License, non-commercial) is a different architecture and not a supported variant; `apply_teacache` raises `IncompatibleModelError` for it. On a 32 GB Mac stock 4-bit Qwen is grainy — the [comparison page](https://github.com/IonDen/mlx-teacache/blob/main/docs/comparison/qwen-image.md) now runs the stock 4-bit build directly, and the [variant page](docs/variants/qwen-image.md) documents a mixed-precision build that clears the artifact; mlx-teacache stays quantization-agnostic.

**[`flux2-klein-base-9b`](docs/variants/flux2-klein-base-9b.md)** — non-distilled FLUX.2 Klein 9B (FLUX Non-Commercial — see [License obligations](#license-obligations) and accept on the [Hugging Face model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) before downloading). Reuses base-4b's coefficients and `rel_l1_thresh=0.17` default (shared FLUX.2-Klein architecture family and calibration recipe). At the canonical 50-step + g=4.0 CFG recipe it skips 13/48 active steps for **1.37× combined wall-clock** (517.3 s → 377.5 s median; 1.35× gating + 1.02× compile-avoidance), at SSIM 0.993 against vanilla and a peak-memory drop from ~22 GiB to ~9.8 GiB (footnote ²). Footnote ² gives the full history and says what each record does not name.

> **Correction.** v0.5.0 advertised 2.68× on this variant (Klein base-9B; that figure is withdrawn and unrelated to Qwen-Image's current 2.68×). It came from one run of each condition, plain mflux and the wrapper, each in its own process, at 768×1024 (`_artifacts/validation_klein_base_9b.json`, dated 2026-05-19 in the record; the record names no mflux or MLX version, no commit and no dirty flag). Plain mflux peaked at 25.21 GiB there, above the 24 GiB cap the run set and above the 24.96 GiB working set the scripts assume for a 32 GB M1 Max (a default in the scripts, not a value recorded on a device). The wrapper peaked at 13.17 GiB. That fits a slowdown from memory pressure in the plain run. The run did not test whether memory pressure caused it, and nobody repeated it. The 1.37× above is a different recipe: 512×512, a different prompt, three reps. No other tracked 768×1024 timing of this model exists. The 2.68× stays withdrawn, and 1.37× stands at its own recipe (footnote ²).
>
> (Corrected 2026-11-13: an earlier version of this note said the two runs shared one Python process, so the wrapper inherited a warm allocator. The record shows each ran in its own process. That explanation was wrong.)

See `_artifacts/v0.12.0_bench_klein_base_9b.json` for the full report (the v0.10.0 and v0.6.0 reports sit alongside it); regenerate side-by-side images with `scripts/bench_comparison.py`.

**[`flux2-klein-base-4b`](docs/variants/flux2-klein-base-4b.md)** — non-distilled FLUX.2 Klein 4B (Apache-2.0), the primary FLUX.2 use case. At the canonical upstream CFG recipe (`guidance=4.0, num_inference_steps=50`) with its per-variant default `rel_l1_thresh=0.17`, the gate skips 8/48 active steps for a **1.20× combined speedup**³ on M1 Max, almost all of it step-skipping, with compile-avoidance at noise level (1.01×). CFG runs through a per-branch gated path (each branch keeps its own cached residual). Measurement history, including v0.4.1's superseded decomposition, is in footnote ³, which also says what each record does not name.

**[`flux2-klein-9b`](docs/variants/flux2-klein-9b.md)** — coefficients are calibrated at `num_inference_steps=8`, origin-constrained polyfit. At the default threshold, the gate produces 0 step-skips on Klein 9B's 8-step schedule (the empirical adjacent-step body-output rel-L1 starts at 0.25 — above the 0.20 threshold). The only wall-clock effect left is `mx.compile`-path avoidance, and only on Macs where mflux compiles the prediction step; the gate contributes nothing, and output quality is preserved (SSIM ≥ 0.85 PR-gate). See [Benchmarks](#benchmarks) → "How the speedup happens".

**[`z-image-base`](docs/variants/z-image-base.md)** — Z-Image base (Tongyi-MAI, Apache-2.0), a single-stream DiT and the first non-FLUX model with a TeaCache mini-kernel. Its adaLN modulation is timestep-only, so there is no cheap caption-independent modulation input to gate on; the gate signal is the first-main-layer residual (calibrated in-repo as "Signal B", R² 0.400 — the threshold sweep, not the fit R², sets the quality bar here; the caption-independent noise-refiner tap was tried and rejected at R² 0.069). Per-variant default `rel_l1_thresh=0.12`, set at the SSIM knee. At the 512×512 red-apple recipe it skips 15/48 active steps for **1.31× combined wall-clock**⁵ (227.4 s → 174.2 s), entirely from gating — the no-gate wrapper runs at vanilla speed (0.99×), so compile-avoidance is not a tailwind here. Peak memory drops 17.2 GiB → 11.5 GiB (the eager wrapper bypassing mflux's compiled `_predict`, not the gate). mflux 0.22.0 moved Z-Image's transformer to bfloat16, which changes its speed and images, and mflux 0.24.0 changed the default image decode, so the speedup and the memory peaks hold for mflux 0.18.0; the bench row was measured there and not since. SSIM 0.991 vs vanilla, from the threshold sweep recorded in `_artifacts/sweeps/z-image-base_2026-05-31.json`, not from the bench. `uv.lock` at c6a4109 pins mflux 0.17.5. For the same reasons the SSIM scores hold for mflux 0.17.5. Commit c6a4109, made shortly before the run, has the script as it then stood; whether the working tree matched it is unknown. The sweep predates the 0.10.0 gate change, and the images behind the scores are not published. Footnote ⁵ has the measurement history of the bench record (`_artifacts/v0.10.0_bench_z_image.json`, which has no commit or dirty field, no date and no MLX version; its install-time version string names commit 90fca33); [docs/comparison/z-image-base.md](https://github.com/IonDen/mlx-teacache/blob/main/docs/comparison/z-image-base.md) has a separate single-run comparison at 864×1152.

See `_artifacts/v0.10.0_bench_z_image.json` for the full report.

**[`flux1-krea-dev`](docs/variants/flux1-krea-dev.md)** — FLUX.1 Krea [dev] (Black Forest Labs with Krea; FLUX.1-dev Non-Commercial, gated), a FLUX.1-dev-architecture finetune that reuses the FLUX.1 proxy strategy unchanged. It does not reuse FLUX.1-dev's coefficients: scored on Krea's own calibration pairs, dev's tuple gives R² −496, because Krea changes roughly three times more per step, so Krea ships its own fit (R² 0.68, calibrated in-repo at the model card's 28-step, guidance 4.5 recipe) and a per-variant default `rel_l1_thresh=0.30`, the sharp knee of its threshold sweep (SSIM 0.990 at 0.30, 0.890 at 0.35; record: `_artifacts/sweeps/flux1-krea-dev_2026-09-05.json`). `uv.lock` at 3b86c8b pins mflux 0.18.0. mflux 0.22.0 loads the bias of FLUX.1's last normalization layer again, which changes its images slightly, so these scores hold for mflux 0.18.0. The script was committed on a working branch before the run and reached main in 3b86c8b; the images behind the scores are not published. The package fallback 0.20 would skip nothing on this model; that follows from the fit, and no run at 0.20 was recorded. The v0.11.0 three-way bench at the red-apple 512×512 recipe (subprocess-per-rep, three cold reps, q4, 28 steps, g=4.5): **1.62× combined**⁷ (125.3 s → 77.2 s), 10 of 26 active steps skipped in every rep and never two in a row. The gated run is about 1.57× faster than the no-gate run, and the no-gate wrapper's 1.03× over plain mflux is within the rep-to-rep range; FLUX.1 has no `mx.compile` path to avoid. Both the 1.62× and the 1.57× are above its 1.56× limit (footnote ⁷). Reproduce with `uv run python scripts/bench_speedup.py --variant krea-dev --three-way --reps 3`.

## When to use mlx-teacache

TeaCache helps when neighboring steps look alike, so the network's output barely moves between them. In the models measured so far, that held for non-distilled models at 20 or more steps.

In practice, that means:

- Use mlx-teacache for **`qwen-image`** at 50 steps with CFG, the **non-distilled FLUX.2 Klein** family (`flux2-klein-base-9b`, `flux2-klein-base-4b`) at 20-50 steps with or without CFG, **`z-image-base`** at 50 steps with CFG, **`flux1-dev`** at 20-50 steps, and **`flux1-krea-dev`** at its 28-step recipe. These are the variants featured in [COMPARISON.md](https://github.com/IonDen/mlx-teacache/blob/main/COMPARISON.md), and the wrapper measurably skips steps and produces visually equivalent output. **`qwen-image-2512`** has its own calibration and threshold sweep but no speed benchmark yet.
- Do not reach for it on the **distilled** variants — `flux1-schnell` (4 steps), `flux2-klein-4b` and `flux2-klein-9b` at their distilled defaults (4-8 steps). The residual between adjacent steps is too large for the gate to engage at any reasonable threshold, so it skips zero steps and adds a small gating overhead (not measured on its own). Run those through vanilla mflux. The full story of that zero-skip result is in [the distilled-schedules research note](https://ineshin.space/papers/why-teacache-does-not-engage-on-short-distilled-schedules/).

On FLUX.2 variants there can be a separate, incidental benefit whether or not the gate engages: on Macs where mflux compiles the prediction step (not base or Pro M1/M2), the wrapper sidesteps the compiled `_predict` path, which on Max and Ultra chips has been slower than the uncompiled one. That is a wall-clock effect from compile avoidance, not from step-skipping, and we keep the two attributions separate in the docs.

Distilled Klein or Base plus the wrapper? `scripts/bench_klein_base_vs_distilled.py` runs distilled, Base, and Base + TeaCache on one prompt and seed and prints a table. On an M1 Max at q4 the distilled 4-step model is about 22× faster than Base at 50 steps on 4B (26 s against 589 s; 768×1024, 3 reps) and about 20× on 9B (26 s against 522 s; 512×512, 2 reps), and the wrapper trims 17 percent off Base on 4B and 29 percent on 9B rather than closing that gap (`_artifacts/klein_base_vs_distilled_4b_q4.json`, `_artifacts/klein_base_vs_distilled_9b_q4_512x512.json`, both measured with mlx-teacache 0.11.0 on mflux 0.18.0 and MLX 0.31.2; neither record names a commit or a dirty flag, and both carry the same date stamp). So the distilled model stays the everyday choice, and Base is for when you need guidance, negative prompts, or a fine-tune. Run it with `uv run python scripts/bench_klein_base_vs_distilled.py --size 4b --quantize 4 --height 1024 --width 768 --reps 3` (for 9B, `--size 9b --quantize 4 --height 512 --width 512 --reps 2`); on a 64 GB or larger Mac, run it at `--quantize 8` and post your table.

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
print(handle.decision_rule_revision, handle.mflux_version)
```

The last line prints the revision of the gate's decision rule and the mflux version the patch was applied on. Skip counts can differ between minor releases or mflux versions, and these two values show which of them changed.

## Custom coefficients

```python
custom = [...]  # length 5, all finite
apply_teacache(flux, coefficients=custom, rel_l1_thresh=0.20)  # use your own threshold
```

Pass a `rel_l1_thresh` with your coefficients. A threshold is tuned for one polynomial, so the model's default does not carry over to yours. If you leave it out, `apply_teacache` picks 0.30 on FLUX.1 Krea [dev] and 0.20 on every other model, and warns with a `FutureWarning`. From 1.0 the same call raises an error. To find a threshold for a new polynomial, run `scripts/sweep_threshold_qwen.py`, `scripts/sweep_threshold_z_image.py`, `scripts/sweep_threshold_klein_base_4b.py`, or `scripts/calibrate_flux1.py --sweep` for the FLUX.1 models. Your polynomial is evaluated on the raw input change. The calibrated-range clamp described under [How it works](#how-it-works) applies only to a variant's own built-in tuple.

## How it works

TeaCache observes that in diffusion denoising, consecutive transformer outputs change very little between most pairs of adjacent steps. The expensive transformer body (all the joint and single attention blocks) produces a residual that's added to the input, and that residual stays roughly stable for stretches of the denoising trajectory.

TeaCache trains a tiny polynomial that predicts how much the output will change given how much the input has changed, measured as the relative L1 distance between each step's modulated block-0 input and the immediately previous step's — matching how the polynomials are calibrated, and the comparison anchor advances on every step, including skipped ones. TeaCache accumulates the predicted change across consecutive skips; once that running total would cross `rel_l1_thresh`, TeaCache recomputes the transformer body instead of reusing the cached residual, and resets the running total. For the polynomials calibrated in this repo (FLUX.2 Klein base, Z-Image, Qwen-Image, Qwen-Image-2512), the input change is first held to the range the calibration actually saw, so a change smaller than any calibrated step is priced like the smallest one rather than as almost no change, and a change larger than any calibrated step always recomputes. Only the cheap prelude (embeddings) and tail (norm + projection) still run on a skipped step.

mlx-teacache implements this for mflux on Apple Silicon. For FLUX.1 we replace `flux.transformer` with a per-instance proxy; for FLUX.2 we replace `flux._predict` with an instance-level closure, which keeps gating live on chips where mflux would otherwise wrap `_predict` in `mx.compile`. The FLUX.1 polynomial coefficients are vendored from upstream. The FLUX.2 coefficients are derived in-repo by `scripts/calibrate_flux2.py` (distilled klein-4b and klein-9b, base-4b each have their own calibrated tuple; klein-base-9b cross-imports base-4b's tuple because the architectures share the calibration recipe). See [`docs/calibration.md`](docs/calibration.md) for procedure and provenance, and the per-variant docs under `docs/variants/` for each tuple's R² and origin date. The original method is described in the TeaCache paper at https://liewfeng.github.io/TeaCache/.

For the full contributor's map — the variant registry, the mflux-free import contract, the per-family patch strategies, the gate internals, and the teardown contract — see [`docs/architecture.md`](docs/architecture.md).

## Benchmarks

The timings come from `scripts/bench_speedup.py` on an M1 Max with 32 GB; each SSIM value in the footnotes is one with a record the footnote names. *Vanilla* is plain mflux, *Wrapper* is the same run with mlx-teacache, and *Skipped* counts the steps where the transformer body did not run, out of the nominal step count. The first and last steps always run; where a footnote or note says "active steps", it counts out of the steps the gate can skip (48 in a 50-step run). Each condition ran three times, every run in a fresh Python process so it starts from a cold MLX allocator, and the table gives the median.

Shared recipe: bf16, quantize=4, 512×512, `seed=42`, the red-apple prompt, and each variant's default threshold (`0.30` on qwen-image and flux1-krea-dev, `0.17` on the base-4b and base-9b rows, `0.12` on z-image-base, `0.20` on flux1-dev); no record stores its threshold, so these are the defaults in the code at the commits involved. Two rows use their own pinned recipe: `qwen-image` runs at 768×768 and `z-image-base` at q8. The records store the Darwin kernel version, not a macOS name: Darwin 25.6.0 (macOS 26), except Darwin 27.0.0, which maps to macOS 27.0, for the base-4b and base-9b CFG rows. `qwen-image` was measured on 2026-09-06 (footnote ¹), the base-9b and base-4b CFG rows on 2026-09-28 with the gate as of the v0.12.0 work (footnotes ² ³), `z-image-base` on 2026-08-15 with the gate as of the v0.10.0 work (footnote ⁵), `flux1-krea-dev` on 2026-09-05 (footnote ⁷), and `flux1-dev` on 2026-11-16 (footnote ⁶). The `flux1-dev` record has a date field and names its commit; no other headline row's record has a report-level date, and each footnote says where its date comes from. Every measured row ran on mflux 0.18.0, which each record names, except `flux1-dev`, which ran on mflux 0.24.0 and MLX 0.32.2 (both named in its record; Darwin 27.2.0). The development lock now resolves mflux 0.24.0. Only the `flux1-dev` row has been re-measured on it; every other row stays tied to the version it was measured on.

No other headline row's record names its MLX version. Where a footnote gives one, it is the version the lock file resolves at that commit; at the commits of the older records the lock lists MLX 0.31.1 for Linux and 0.31.2 for macOS, and the footnotes quote the macOS one.

Each record also holds the mlx-teacache version string. In the older records this is the install-time version string, fixed when the package is installed, so it can name an earlier commit than the run. The `flux1-dev` record's string comes from git at run time. Where a record has a git commit field, that commit is the HEAD when the report was written, after the last chunk.

| Variant | Steps | Vanilla | Wrapper | Speedup | Skipped | Mechanism |
|---|---|---|---|---|---|---|
| `qwen-image` (CFG)¹ | 50 | 642.6s | 239.4s | **2.68×** | **33 / 50** | step-skipping (2.57× against its own no-gate run; the no-gate wrapper's 1.04× is inside the rep spread; provenance in ¹) |
| `flux2-klein-base-9b` (CFG)² | 50 | 517.3s | 377.5s | **1.37×** | **13 / 50** | step-skipping + small compile-avoidance |
| `flux2-klein-base-4b` (CFG)³ | 50 | 237.5s | 197.8s | **1.20×** | **8 / 50** | step-skipping (compile-avoidance ≈ noise) |
| `flux2-klein-4b`, `flux2-klein-9b`⁴ | 8 | — | — | — | **0 / 8** | no skips; any wall-clock difference is `mx.compile` avoidance, only where mflux compiles `_predict` |
| `z-image-base` (CFG)⁵ | 50 | 227.4s | 174.2s | **1.31×** | **15 / 50** | step-skipping (q8; compile-avoidance not a tailwind) |
| `flux1-dev`⁶ | 25 | 89.5s | 68.7s | **1.30×** | **6 / 25** | step-skipping (mflux 0.24.0; the no-gate wrapper equals plain mflux; 6 skipped steps give at most about 1.32×) |
| `flux1-krea-dev`⁷ | 28 | 125.3s | 77.2s | **1.62×** | **10 / 28** | gate on against gate off: about 1.57×, above its 1.56× limit (footnote ⁷); the no-gate wrapper's 1.03× over plain mflux is within the rep-to-rep range, and FLUX.1 has no compile path |
| `flux1-schnell` | — | — | — | — | — | shares dev's coefficients; gate behaves like dev at long schedules, like Klein at the 4-step distilled default (no benefit) |

¹ **2.68×** for `qwen-image` under CFG at 50 steps, `guidance=4.0`, **768×768** (its pinned recipe; the rest of the table is 512×512), q4, default `rel_l1_thresh=0.30`: 642.6 s plain mflux against 239.4 s with TeaCache (2.57× against the wrapper's own no-gate run), 33 of the 48 active steps skipped in every rep. The record is [`_artifacts/v0.11.0_bench_qwen_image.json`](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/v0.11.0_bench_qwen_image.json). It has no date field (2026-09-06 comes from its chunk timestamps), no MLX version, and neither the checkpoint nor the threshold: the bench built the model from mflux's `qwen-image` alias, which on mflux 0.18.0 loads the original Qwen/Qwen-Image. The report names commit 1536f28 and `git_dirty: true`; the harness writes both when it writes the report, after the last chunk, and the report lists no files. `scripts/` and `src/` are identical at 46612df (the last commit before the first chunk), at 1536f28 and at 97f4385 (the commit that added the record, on the pull-request branch, public in [PR #35](https://github.com/IonDen/mlx-teacache/pull/35) and squashed into main as 3b86c8b), so no committed change separates them, but an uncommitted edit during the run cannot be ruled out. The figure holds for mflux 0.18.0: from 0.19.0 the alias loads Qwen-Image-2512, and the 4-bit build keeps `img_mod_linear` at 8 bits (a source change, mflux commit [acad1f9](https://github.com/mflux-community/mflux/commit/acad1f90bf38ae17332013da5d94a672478acf34), [mflux PR #517](https://github.com/mflux-community/mflux/pull/517); not in the 0.19.0 release notes). mflux 0.22.0 also restored a bias in the model's last layer, which changes Qwen-Image's images visibly, so the figure holds for mflux 0.18.0. The peak, the 0.10.0 report and the comparison between the two runs are on the [Qwen-Image page](docs/variants/qwen-image.md#benchmark). Reproduce with `uv run python scripts/bench_speedup.py --variant qwen --three-way --reps 3 --report out.json`.

² `flux2-klein-base-9b` under CFG at the canonical 50-step + g=4.0 recipe. Measured 2026-09-28 under the subprocess-per-rep harness (Darwin 27.0.0, which maps to macOS 27.0; the record stores only the kernel version; mflux 0.18.0, the gate as of the v0.12.0 work, chunks interleaved rep-outer): **1.37× combined** = 1.35× gating × 1.02× compile-avoidance (vanilla 517.3 s, no-gate 508.3 s, gated 377.5 s medians). Each factor is rounded on its own from the unrounded medians, so the rounded factors need not multiply to the rounded total. The 13 skipped steps of 50 alone allow 1.35× (50/37); the rest of the 1.37× is the 1.02× compile avoidance. The record names commit ec4d110 with a clean tree and has no MLX version field; MLX 0.31.2 per the lock file at ec4d110. The date comes from the record's chunk timestamps, and `scripts/` and `src/` are identical at ec4d110 and at 1290c01, the commit that added the record, on the pull-request branch (public in [PR #42](https://github.com/IonDen/mlx-teacache/pull/42), squashed into main as 0a3482c; that merge also added to `gate.py` the rule that a change above the calibrated range always computes). mflux 0.24.0 changed the default image decode for FLUX.2 Klein to use far less memory, and the bench timer includes the decode, so the timings and the memory peaks here hold for mflux 0.18.0. 13/50 skips in every rep, never two consecutive, the same steps v0.10.0 skipped, so clamping small changes to the calibrated range changes nothing at this recipe. This bench ran before the rule that always computes a change above the range was added, and that rule was not traced on 9B; on 4B at the same recipe no priced change came near the top of the range. SSIM 0.993 vs vanilla (recorded in `_artifacts/v0.12.0_klein_base_ssim.json`), computed the same way as in footnote ³. v0.10.0 measured 1.37× (520.6 s → 379.1 s, `_artifacts/v0.10.0_bench_klein_base_9b.json`) and v0.6.0 1.36× (`_artifacts/v0.6.0_bench_klein_base_9b.json`); like the current figure, both are above that 1.35× limit. In each of the three records the gated run against the no-gate run is under the limit, and the rest is the wrapper's lead over plain mflux. All three use the 512×512 recipe and replace v0.5.0's advertised 2.68×, which is withdrawn: it was one unrepeated run at 768×1024, a different recipe (see [Per-variant notes](#per-variant-notes) → klein-base-9b correction blockquote). Full report: `_artifacts/v0.12.0_bench_klein_base_9b.json`. Wrapper peak memory ~9.8 GiB vs vanilla's ~22 GiB. The v0.10.0 record has no commit or dirty field, no date and no MLX version; its install-time version string names commit 90fca33. The v0.6.0 record (mflux 0.17.5) has no commit, dirty flag, date or MLX version, and its version string names no commit. Reproduce with `uv run python scripts/bench_speedup.py --variant klein-base-9b --three-way --reps 3 --report out.json`.

³ `flux2-klein-base-4b` under CFG at the canonical upstream BFL recipe (`guidance=4.0`, 50 steps), per-variant default `rel_l1_thresh=0.17`. Measured 2026-09-28 under the subprocess-per-rep harness (Darwin 27.0.0, which maps to macOS 27.0, as in footnote ²; mflux 0.18.0, the gate as of the v0.12.0 work, chunks interleaved rep-outer): **1.20× combined** = 1.18× from step-skipping (wrapped-no-gate vs wrapped-gated) × 1.01× from `mx.compile`-path avoidance (vanilla vs wrapped-no-gate); vanilla 237.5 s, no-gate 234.2 s, gated 197.8 s medians. Each factor is rounded on its own from the unrounded medians, so the rounded factors need not multiply to the rounded total. The 8 skipped steps of 50 alone allow 1.19× (50/42); the rest of the 1.20× is the 1.01× compile avoidance. The record names commit ec4d110 with a clean tree and has no MLX version field; MLX 0.31.2 per the lock file at ec4d110. The date comes from the record's chunk timestamps, and `scripts/` and `src/` are identical at ec4d110 and at 1290c01, the commit that added the record, on the [PR #42](https://github.com/IonDen/mlx-teacache/pull/42) branch (see footnote ²). The decode change in footnote ² applies here too: the timings and the memory peaks hold for mflux 0.18.0. As in footnote ², this bench ran before the rule that always computes a change above the calibrated range was added. 8 of the 48 active steps skipped in every rep, never two in a row; SSIM 0.985 against vanilla (scikit-image `structural_similarity` on the bench's rep-0 vanilla and gated images, `data_range=255`; recorded with the image hashes in `_artifacts/v0.12.0_klein_base_ssim.json`). Until v0.12.0 the gate also skipped step 6. Its input change was smaller than any step in the calibration, and the fit, which passes through zero, priced it as almost no change. v0.10.0 measured 1.22× with that extra skip (9/50, longest run 2; `_artifacts/v0.10.0_bench_klein_base_4b.json`, macOS 26) and v0.6.0 1.23× (`_artifacts/v0.6.0_bench_klein_base_4b.json`), which is above the 1.22× that 9 skipped steps of 50 allow (50/41). Between the v0.10.0 and v0.12.0 runs the machine moved from macOS 26 to 27, and vanilla itself slowed by 1.5% (233.9 s to 237.5 s), so the gated time's 2.8% rise (5.5 s) is part host drift and part the one extra computed step. Compile-avoidance is effectively noise on this recipe; the v0.4.1-era 1.09× attribution stays withdrawn. The v0.4.1 bench script ran an untimed plain warmup first, then plain mflux, the no-gate wrapper and the gated wrapper in one Python process. Its record, `_artifacts/_bench_report_klein_base_4b_v0_4_1.json`, has medians of 254.0 s plain, 233.5 s no-gate and 201.2 s gated. When v0.6.0 gave each rep its own process (`_artifacts/v0.6.0_bench_klein_base_4b.json`), the medians were 236.2 s, 233.4 s and 191.8 s. The no-gate wrapper took the same time in both. So the change in the compile-avoidance ratio came from the plain run. The gated run was also slower in v0.4.1 (201.2 s against 191.8 s), and the records do not say why either run was slower. The v0.10.0 record has no commit or dirty field, no date and no MLX version; its install-time version string names commit 90fca33. The v0.6.0 record (mflux 0.17.5) has no commit, dirty flag, date or MLX version, and its version string names no commit. The v0.4.1 record names no hardware, versions, date or commit at all. (Corrected 2026-11-13: an earlier version of this note said the wrapper inherited a warm allocator. That explanation was wrong.) Reproduce with `uv run python scripts/bench_speedup.py --variant klein-base-4b --three-way --reps 3 --report out.json`. Full report: `_artifacts/v0.12.0_bench_klein_base_4b.json`.

⁴ The distilled Klein variants (`flux2-klein-4b`, `flux2-klein-9b`) run an 8-step schedule where the gate skips nothing at the package default. They have no committed multi-run timing under the current harness, so the table shows none. Compile avoidance applies only on Macs where mflux compiles `_predict`; on base and Pro M1/M2 mflux already runs the eager path and there is nothing to avoid. Where it does apply, it measured 1.01× to 1.02× on the 50-step Klein base recipes (footnotes ² and ³), and a short 8-step schedule may show more, which is why the docs do not quote a figure for it.

⁵ `z-image-base` under CFG at 50 steps, `guidance=4.0`, 512×512, **q8** (its pinned recipe — the rest of the table is q4), per-variant default `rel_l1_thresh=0.12`. Measured 2026-08-15 under the subprocess-per-rep harness (mflux 0.18.0, the gate as of the v0.10.0 work). The record has no commit or dirty field, no timestamp and no MLX version; its install-time version string names commit 90fca33. The date comes from the README at the v0.10.0 release; MLX 0.31.2 per the lock file at c33e8d5. `src/` is identical at 90fca33 (public in [PR #33](https://github.com/IonDen/mlx-teacache/pull/33)) and at c33e8d5 (public in [PR #33](https://github.com/IonDen/mlx-teacache/pull/33)), which added the record; the gate in the v0.10.0 release differs slightly from the one measured. mflux 0.22.0 moved Z-Image's transformer to bfloat16, which changes its speed and images, and mflux 0.24.0 changed the default image decode, so the speedup and the memory peaks hold for mflux 0.18.0; this row was measured there and not since. Result: **1.31× combined** (vanilla 227.4 s → wrapper 174.2 s median), entirely from step-skipping at 15 of the 48 active steps (the table's `15 / 50` is over nominal steps; 48 active = 50 minus the skip-first/skip-last windows), the same 15 in every rep and never two consecutive. `mx.compile`-path avoidance is not a tailwind here: the no-gate wrapper ran at 228.9 s median against vanilla's 227.4 s (0.99×), so the whole 1.31× is gating. Peak memory 17.2 GiB → 11.5 GiB, from the eager wrapper bypassing mflux's compiled `_predict` (the no-gate wrapper shows the same ~11.5 GiB), not from gating. v0.7.0 reported 1.17× at this recipe (`_artifacts/_bench_z_image_v0_7_0.json`, wrapper 209.4 s; that record has mflux 0.17.5, no commit or dirty field, no date and no MLX version; its install-time version string names commit c9db998 and marks uncommitted changes at install time). The skip counts are identical (the record stores counts, not which steps were skipped); the cause of the gap is not recorded. Full report: `_artifacts/v0.10.0_bench_z_image.json`. Reproduce with `uv run python scripts/bench_speedup.py --variant z-image --three-way --reps 3 --report out.json`.

⁶ `flux1-dev` at 25 steps, `guidance=3.5`, 512×512, q4, default `rel_l1_thresh=0.20`. **Current figure, 1.30×.** Measured 2026-11-16 on mflux 0.24.0 and MLX 0.32.2 (Darwin 27.2.0) under the subprocess-per-rep harness: each rep is a fresh process with no warmup generation, so each timing includes first-call setup. Reproduce with `uv run python scripts/bench_speedup.py --variant flux1-dev --three-way --reps 3 --report out.json`. The record is [`_artifacts/v0.18.0_bench_flux1_dev.json`](https://github.com/IonDen/mlx-teacache/blob/main/_artifacts/v0.18.0_bench_flux1_dev.json). It names commit e7cb468, a pre-release build of 0.18.0 on the development branch; that commit is not in the published history. Between it and the 0.18.0 release three files differ: the warning texts in `src/mlx_teacache/api.py`, a docstring in `src/mlx_teacache/support.py`, and the benchmark script's git and resume handling (a timeout, an explicit untracked-files flag, and reading back only the start keys on resume). The generation path the bench timed is the same, and `pyproject.toml` and `uv.lock` are identical. The record's version string, `v0.16.0-20-ge7cb468`, names that same commit, counted from the v0.16.0 tag.

Plain mflux took 89.46 s (reps 89.48 / 89.46 / 89.45). The no-gate run, which computes every step, took 89.46 s (89.42 / 96.76 / 89.46), the same time as plain mflux, so on mflux 0.24 the wrapper alone gains nothing. The gated wrapper took 68.74 s (68.74 / 68.75 / 68.67), a median speedup of 1.30×. 6 of 25 steps were skipped in every rep, never two in a row, and peak memory was 11.08 GiB. All of the 1.30× comes from the 6 skipped steps. 19 of 25 steps were computed: 17 the gate decided, plus the first and last step, which always run. So the skipped steps can give at most about 1.32× (25/19), and 1.30× is under that limit.

Plain mflux's median differs between the three FLUX.1-dev records, and this note does not say why: about 104 s on mflux 0.17.5 (the v0.6.3 record, steady reps), 113.05 s on mflux 0.18.0 (the v0.10.0 record, reps spread from 104.7 to 119.8 s) and 89.46 s on mflux 0.24.0 (the v0.18.0 record, steady reps).

History, the 1.57× run. Measured 2026-08-15 under the subprocess-per-rep harness (mflux 0.18.0, the gate as of the v0.10.0 work). The record has no commit or dirty field, no timestamp and no MLX version; its install-time version string names commit 90fca33. The date comes from the README at the v0.10.0 release; MLX 0.31.2 per the lock file at c33e8d5. `src/` is identical at 90fca33 (public in [PR #33](https://github.com/IonDen/mlx-teacache/pull/33)) and at c33e8d5 (public in [PR #33](https://github.com/IonDen/mlx-teacache/pull/33)), which added the record; the gate in the v0.10.0 release differs slightly from the one measured. Measured on mflux 0.18.0; mflux 0.22.0 loads the bias of FLUX.1's last normalization layer again, which changes its images slightly, so the figure holds for mflux 0.18.0. Result: **1.57× combined** as the median of three cold reps (vanilla 113.05 s → gated 71.85 s). That is above the 1.32× limit worked out above, so it cannot all come from the skipped steps. The plain-mflux reps varied between runs (104.7 / 113.1 / 119.8 s) while the gated wrapper held steady (69.8 / 71.9 / 73.1 s), the same 71 s v0.6.3 measured. Even fastest to fastest, 1.50× is above the limit. The no-gate wrapper took 101.64 s, 1.11× ahead of plain mflux, and mflux does not compile the FLUX.1 predict step, so that gap is not compile avoidance. The 0.24.0 run above, with steady plain times, measured 1.30×; that is the current figure, and 1.57× stays here as history. The earlier 1.46× (v0.6.3, mflux 0.17.5: plain 103.78 s → gated 70.98 s) is also above the 1.32× limit and was not published as a headline; its record has no commit or dirty field, no date and no MLX version; its install-time version string names commit c9db998 and marks uncommitted changes at install time. 6/25 skips in every rep, never two consecutive. Full report: `_artifacts/v0.10.0_bench_flux1_dev.json` (v0.6.3's is kept alongside as `_artifacts/v0.6.3_bench_flux1_dev.json`). Reproduce with `uv run python scripts/bench_speedup.py --variant flux1-dev --three-way --reps 3 --report out.json`.

⁷ `flux1-krea-dev` at 28 steps, `guidance=4.5`, 512×512, q4, per-variant default `rel_l1_thresh=0.30`. Measured 2026-09-05 under the subprocess-per-rep harness (mflux 0.18.0). The record is `_artifacts/v0.11.0_bench_krea_dev.json`. It has no date field (the date comes from its chunk timestamps), names commit 040056a, notes uncommitted changes in the working tree without listing files, and has no MLX version field; MLX 0.31.2 per the lock file at 040056a. The run started 43 seconds before that commit was made, and the commit before the run still had FLUX.1-dev's coefficients and a 0.20 default, so this variant's own coefficients and 0.30 default were not committed when the run started. `scripts/` and `src/` are identical at 040056a and at e5bd1da, the commit that added the record, on the pull-request branch (public in [PR #35](https://github.com/IonDen/mlx-teacache/pull/35)). mflux 0.22.0 restored a bias in the model's last layer, which changes FLUX.1's images slightly, so the figure holds for mflux 0.18.0. Result: **1.62× combined** as the median of three cold reps (vanilla 125.3 s → gated 77.2 s). The gated run was about 1.57× faster than the no-gate run; the no-gate wrapper's 1.03× over plain mflux is within the rep-to-rep range, and mflux does not `mx.compile` the FLUX.1 predict step, so there is no compile path to avoid. 10 of the 26 active steps skipped in every rep (the table's `10 / 28` counts nominal steps), never two in a row. Ten skipped steps out of 28 leave 18 computed, so they can give at most about 1.56× (28/18); the 1.62× and the 1.57× are both above that, and the record does not say why. The three TeaCache reps were 77.2 / 79.6 / 66.5 s: one was much faster than the other two, against 125.3 / 129.3 / 123.9 s for plain mflux. This row will be measured again. The default sits at the knee of a threshold sweep recorded in `_artifacts/sweeps/flux1-krea-dev_2026-09-05.json`; the variant page lists what that record leaves out. Peak memory 11.1 GiB. Full report: `_artifacts/v0.11.0_bench_krea_dev.json`. Reproduce with `uv run python scripts/bench_speedup.py --variant krea-dev --three-way --reps 3 --report out.json`.

Reproduce any row:

```bash
uv run python scripts/bench_speedup.py --variant qwen           # 50-step Qwen-Image under CFG (g=4.0, 768x768)
uv run python scripts/bench_speedup.py --variant klein-base-9b  # 50-step base-9B under CFG (g=4.0, v0.5.0+ default)
uv run python scripts/bench_speedup.py --variant klein-base-4b  # 50-step base-4B under CFG (g=4.0, v0.4.1+ default)
uv run python scripts/bench_speedup.py --variant klein-base-4b --guidance 1.0 --num-inference-steps 25  # v0.4.0 row
uv run python scripts/bench_speedup.py --variant klein-4b       # 8-step Klein 4B
uv run python scripts/bench_speedup.py --variant klein-9b       # 8-step Klein 9B
uv run python scripts/bench_speedup.py --variant z-image        # 50-step Z-Image base under CFG (g=4.0, q8)
uv run python scripts/bench_speedup.py --variant flux1-dev --three-way --reps 3      # 25-step dev, the three-way median
uv run python scripts/bench_speedup.py --variant krea-dev       # 28-step FLUX.1 Krea [dev] (g=4.5)
```

For the three-way decomposition (vanilla / wrapped-no-gate / wrapped-gated), add `--three-way --reps 3 --report out.json`. This is how the v0.10.0 rows above were produced.

### How the speedup happens

The wall-clock improvement above comes from two distinct mechanisms; they fire independently depending on variant and schedule.

**1. TeaCache step-skipping.** This is the headline feature. The polynomial gate predicts how much the transformer body output will change from the immediately previous step, whether or not that step was itself skipped. The wrapper accumulates the predicted change across consecutive skips, and once that running total would cross `rel_l1_thresh`, it recomputes the transformer body instead of reusing the cached residual and resets the running total. On FLUX.1-dev at 25 steps, 6 of 25 steps are skippable and this is where the 1.30× speedup on FLUX.1-dev (mflux 0.24.0, Benchmarks footnote ⁶) comes from. The 1.30× is under its 1.32× limit (footnote ⁶). mflux does not compile the FLUX.1 predict step, so none of that speedup is compile avoidance, and the no-gate wrapper runs at plain mflux's speed. On non-distilled FLUX.2 Klein at 50 steps + CFG, the same mechanism produces 8 to 13 skips per generation and dominates the wall-clock win (the v0.12.0 three-way bench attributes 1.35× to gating on base-9b, whose 13 skips are in footnote ²; base-4b's 8 skips are in footnote ³).

**2. `mx.compile` avoidance on FLUX.2 and Z-Image.** mflux wraps `Flux2Klein._predict` (and Z-Image's, behind the same chip check) in `mx.compile` on every chip *except* base + Pro M1/M2 — i.e., compilation is active on M1/M2 Max + Ultra and on every M3, M4, M5 chip. mlx-teacache replaces the compiled `_predict` with an eager Python closure so the gate can run live per step. The magnitude of this effect varies by schedule and chip. On the 50-step Klein base CFG recipes the subprocess-per-rep bench measures it at 1.01× to 1.02× (footnotes ³ and ²): kernel-dispatch round-trips drop slightly, but with 50 steps per generation that gain is small relative to the per-step compute. Short distilled schedules may benefit more because each step is a larger share of the wall-clock, but this repository has no committed multi-run measurement of that, so no figure is quoted. On chips where mflux is already eager (base + Pro M1/M2), this mechanism does not fire: the wrapper just adds per-step gate overhead, and Klein with mlx-teacache on those chips is approximately neutral or slightly slower than vanilla.

On FLUX.2 Klein 4B and 9B at the distilled 4-8 step defaults, mechanism (1) does not engage: the empirical adjacent-step rel-L1 between consecutive transformer outputs is ≥ 0.25, so every step's predicted change exceeds the default 0.20 threshold and the gate signals "compute" every time. This is expected — distilled schedules collapse the entire denoising trajectory into a handful of consequential steps, so adjacent steps are not similar enough to skip. The gate skips nothing here, so whatever speed difference you see is mechanism (2), which only exists on Macs where mflux compiles `_predict`.

For algorithmic step-skipping on FLUX.2, use the non-distilled `flux2-klein-base-4b` (Apache-2.0) or `flux2-klein-base-9b` (FLUX Non-Commercial) variants. Both ship with a per-variant default `rel_l1_thresh=0.17`. At the canonical upstream 50-step + g=4.0 CFG recipe, base-4b skips 8/50 steps for a measured 1.20× combined speedup (subprocess-per-rep, footnote ³), and base-9b skips 13/50 steps for 1.37× (footnote ²). Pushing the threshold higher on distilled Klein is not recommended: the gate's prediction quality at thresholds > 0.25 is uncalibrated on a 4-8 step trajectory and image quality is not characterized there.

### SSIM suite

Quality gates use a 5-prompt SSIM suite defined at `tests/test_image_quality_flux1.py:45` and reused at `tests/test_image_quality_flux2.py:28`:

- "a red apple on a wooden table"
- "mountain landscape at sunset"
- "portrait of a woman"
- "abstract pattern with circles"
- "text saying HELLO"

The PR-gate prompt is the red-apple one; SSIM ≥ 0.90 on FLUX.1-dev and ≥ 0.85 on Klein 4B / 9B at the default threshold. Full suite floor is 0.80 to absorb high-frequency-detail variance (text, synthetic patterns). Run `uv run pytest tests/test_image_quality_flux1.py tests/test_image_quality_flux2.py -m parity` with real model weights to reproduce.

## Performance by chip

mflux wraps `_predict` in `mx.compile` on every Apple Silicon chip *except* base + Pro M1/M2 (the behaviour is unchanged from mflux 0.17.5 through 0.24). The `is_m1_or_m2()` predicate returns true (eager path) when the chip brand contains "Apple M1" or "Apple M2" *and* does not contain "Max" or "Ultra" — so M1 Pro and M2 Pro are eager too, while M1/M2 Max + Ultra and every M3/M4/M5 chip get the compiled path. mlx-teacache replaces `_predict` with an eager closure so per-step gating stays live, trading the compile gain for the skip gain on compiled chips. That compile-avoidance effect applies to FLUX.2 Klein and Z-Image (mflux compiles Z-Image's `_predict` behind the same chip check); FLUX.1, FLUX.1 Krea, and Qwen-Image have no `mx.compile` path, so their step-skipping speedup is the same on every chip. See `docs/m3-plus-tradeoff.md` for a benchmark recipe.

| Chip | Vanilla `_predict` in mflux | Expected speedup |
|---|---|---|
| Apple M1 / M2 (base) | eager | the model's skip fraction, chip-independent (2.68×¹ Qwen-Image, 1.62×⁷ Krea (above its 1.56× limit), 1.30×⁶ FLUX.1-dev, all measured on M1 Max and untested on this hardware); FLUX.2 loses its small compile-avoidance share here |
| M1 Pro / M2 Pro | eager | same as base — the skip fraction, chip-independent |
| M1 Max / Ultra, M2 Max / Ultra | compiled | **measured on M1 Max:** Qwen-Image 2.68×¹, klein-base-9b 1.37×², klein-base-4b 1.20×³, z-image-base 1.31×⁵, FLUX.1 Krea 1.62×⁷ (above its 1.56× limit), FLUX.1-dev 1.30×⁶ (see [Benchmarks](#benchmarks)) |
| M3 / M3 Pro / M3 Max / Ultra | compiled | FLUX.1 / Qwen keep their skip fraction; FLUX.2 likely 1.1–1.3× — untested |
| M4 / M4 Pro / M4 Max | compiled | FLUX.1 / Qwen keep their skip fraction; FLUX.2 likely 1.1–1.3× — untested |
| M5+ (Neural Accelerators / TensorOps) | compiled + accelerator | FLUX.2 may approach 1.0× — the eager wrapper can lose some or all of the M5 TensorOps advantage; confirm with a profiler. FLUX.1 / Qwen stay skip-limited |

## Limitations

img2img reuses the txt2img calibration. A dedicated img2img calibration may follow in a future release if SSIM gates flag drift on specific schedules.

FLUX.2 with CFG (`guidance > 1.0`) runs through the gated path as of v0.4.1. The wrapper keeps two cached residuals (positive and negative branch) and shares one gate decision per step across both. The canonical base-4b recipe (`guidance_scale=4.0, num_inference_steps=50`) measures **1.20× combined**³ on M1 Max under the subprocess-per-rep harness (v0.12.0, where 8 skipped steps alone allow 1.19×; v0.10.0 measured 1.22× and v0.6.0 1.23× (above its 1.22× limit, 50/41) before the gate's range check, v0.4.1 advertised 1.26× combined under same-process measurement, above its 1.22× limit (50/41); combined within day-to-day noise, but the decomposition shifted — see Benchmarks footnote ³).

**Distilled schedules are out of scope for algorithmic step-skipping by design.** This includes FLUX.2 Klein 4B / 9B at their 4-8 step defaults and FLUX.1 schnell at its 4-step default. The polynomial gate's premise — that consecutive transformer outputs are similar enough that the residual can be reused — does not hold on distilled trajectories where each step does a much larger share of the denoising work. On the v0.3.0 bench (M1 Max, quantize=4) the gate signals "compute" on every Klein step at the package default `rel_l1_thresh=0.20` (0 skips across 3 reps on both Klein 4B and 9B); empirical adjacent-step body-output rel-L1 on Klein is ≥ 0.25. Where mflux compiles `_predict` (not base or Pro M1/M2), Klein can still run a little faster from `mx.compile`-path avoidance, but the headline TeaCache step-skipping feature only fires on non-distilled schedules.

`flux2-klein-base-4b` runs TeaCache at both `guidance=1.0` (single-branch path) and `guidance > 1.0` (per-branch path, v0.4.1+). The upstream BFL base-4b model card recommends `guidance_scale=4.0, num_inference_steps=50`; the subprocess-per-rep harness measures **1.20× combined wall-clock** vs vanilla on M1 Max at that recipe (v0.12.0, footnote ³: 1.18× gating + 1.01× compile-avoidance; 8/50 skips, SSIM 0.985; v0.10.0 measured 1.22× and v0.6.0 1.23×, which is above its 1.22× limit, 50/41). v0.4.1's same-process bench reported 1.26× combined (also above that 1.22× limit) and decomposed it 1.16× gating / 1.09× compile-avoidance; the combined was honest within day-to-day noise but the decomposition over-attributed to compile-avoidance.

`flux2-klein-base-9b` reuses base-4b's polynomial coefficients verbatim (same architecture family, same calibration recipe). The v0.12.0 subprocess-per-rep bench at the canonical 50-step / guidance=4.0 recipe measures **1.37× combined wall-clock**² (1.35× gating + 1.02× compile-avoidance; 13 skipped steps alone allow 1.35×, and v0.10.0 and v0.6.0 measured 1.37× and 1.36×, both above that limit, 50/37), 13/50 steps skipped at `rel_l1_thresh=0.17`, SSIM 0.993 vs vanilla. v0.5.0 advertised 2.68× on this variant. That figure is withdrawn: it was one unrepeated run at 768×1024, a different recipe from this one (see the correction note under Per-variant notes). Same FLUX Non-Commercial license + BFL safety-filter obligations as `flux2-klein-9b` — see [License obligations](#license-obligations).

The wrapper runs eager, which gives up mflux's `mx.compile` of `_predict` in exchange for live per-step gating. Vanilla mflux compiles `_predict` on every chip except base + Pro M1/M2 (the `is_m1_or_m2()` predicate is true for base and Pro M1/M2, which run eager; Max, Ultra and M3 and newer compile). The 1.30×⁶ measurement is from M1 Max / FLUX.1-dev / 25 steps / mflux 0.24.0; speedup on M3 and newer is plausible but untested locally. On M5, the GPU Neural Accelerators (Metal 4 TensorOps) are only reachable through the compiled path, so the eager wrapper can lose some or all of that advantage. Output stays correct either way. See `docs/m3-plus-tradeoff.md` for the per-chip recipe; PRs with measurements welcome.

FLUX.2 parity is numerical, not bit-exact. Replacing a function that mflux wraps in `mx.compile` produces about 1 ULP per element of divergence from Metal kernel-dispatch noise, which compounds across steps but keeps cosine similarity ≥ 0.97 on Klein 4B, Klein 9B, and base-4b under CFG at threshold 0. The user-facing guarantee is end-to-end image quality (SSIM ≥ 0.85 on all supported FLUX.2 variants at the package default threshold).

Fingerprints are recorded for mflux 0.17.5, 0.18.0, 0.18.1, 0.19.1, 0.19.2, 0.20.0, 0.21.0, 0.22.0, 0.23.0 and 0.24.0; 0.21.0 was also checked on real weights for Z-Image. For 0.22.0 the checks are the fingerprints, a weight-free Qwen-Image test and a real-weights check of FLUX.1-dev, which is bit-identical to plain mflux at threshold 0. 0.23.0 changed none of the functions TeaCache copies; on it, FLUX.1-dev and Qwen-Image-2512 were run on real weights to check that the transformer is freed under mflux's memory saver, and Qwen-Image-2512's own calibration and threshold sweep also ran on 0.23.0. On 0.24.0 the offline test suite (the parity, slow, benchmark and network tests deselected) passed with no model weights loaded, and the functions TeaCache copies or wraps match 0.23.0 in the denoising steps and transformer forwards. In FLUX.2 Klein's and Z-Image's `generate_image`, only code outside the denoising loop changed. Klein gained a check that refuses `pid_decode` together with `small_decoder`, plus extra image metadata; Z-Image gained only extra image metadata. On 0.24.0, real-weights checks passed for FLUX.1-dev, FLUX.2 Klein base 4B and Z-Image (threshold-0 parity and the default-threshold checks). The three Qwen-Image-2512 real-weights tests also passed on 0.24.0: parity with plain mflux under CFG at threshold 0, image similarity at the default threshold, and a run where the skip path engages and returns a finite result. The original Qwen-Image checkpoint was not re-run. The `[mflux]` extra has no upper bound, so a newer mflux installs without waiting for a release here. On an mflux newer than 0.24.0 (0.24.1 included), `apply_teacache` emits a `TeaCacheUntestedMfluxWarning` once per process and still patches the model; if a generation fails or looks wrong there, please open an issue with both version numbers.

On FLUX.1 and Qwen-Image the wrapper replaces `flux.transformer` with a proxy that the parent model's parameter tree does not reach. Calling `flux.parameters()`, `nn.quantize(flux)` or `flux.update(...)` at the parent level can therefore miss the transformer while the wrapper is active. Use `flux.transformer.parameters()` directly, or call `handle.restore()` first. Calls on the transformer itself work: `flux.transformer.load_weights(...)`, `set_dtype`, `eval()` and `nn.quantize(flux.transformer)`. When mflux's memory saver drops the transformer after the denoising loop (`model.transformer = None`), TeaCache keeps no copy of it, and a later `handle.restore()` leaves `flux.transformer` as `None` while undoing the rest of the patch, the same state plain mflux leaves.

## License obligations

The FLUX.1 variants (`flux1-dev`, `flux1-schnell`) and `flux2-klein-4b` come with their own upstream weight licenses; the wrapper this library applies does not change those terms.

`flux2-klein-9b` and `flux2-klein-base-9b` are both distributed under the FLUX.2 Klein license (non-commercial use + BFL safety-filter obligations). These terms flow with the weights, not with mlx-teacache. If you call `apply_teacache` on either variant — `Flux2Klein(model_config=ModelConfig.flux2_klein_9b())` or `Flux2Klein(model_config=ModelConfig.flux2_klein_base_9b())` — you are responsible for ensuring your use complies with the upstream license, including the safety-filter requirements the BFL model cards describe. See the official model cards at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B and https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B for the full terms.

## Research notes

**[Qwen-Image mixed precision on a 32 GB Mac](https://ineshin.space/papers/qwen-image-mixed-precision-on-a-32-gb-mac/)**
documents a uniform-q4 portrait artifact and the mixed q8/q4/bf16 recipe that removed it in one
controlled 512×512 image pair. Both builds were also run at 768×768. A stock q4 run without the text encoders freed (red-apple prompt,
mflux 0.18.0, [`_artifacts/v0.10.0_bench_qwen_image.json`](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/v0.10.0_bench_qwen_image.json))
peaked above the recommended working set of a 32 GB Mac. The mixed-precision build (portrait prompt,
mflux 0.17.5, [`_artifacts/comparison_report.json`](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/comparison_report.json)) peaked higher still. The two runs differ in prompt, mflux version and harness, so the gap is not a measured
effect of the recipe. Neither record has an MLX version, a commit field or a dirty flag, and the first has no date; its install-time version string names commit 90fca33. The comparison report was assembled across library versions and its run date is not
recorded; its row's provenance stamp gives 2026-06-18 and mlx-teacache 0.9.0, and names no commit. The recipe changes model construction, not TeaCache.

**[Why the TeaCache gate did not engage on short distilled FLUX schedules](https://ineshin.space/papers/why-teacache-does-not-engage-on-short-distilled-schedules/)**
documents the zero-skip negative result on the 8-step distilled Klein schedules and the
measurement practice it forced: skip counts published next to every wall-clock number, so a
speedup cannot be mis-attributed to caching.

**[Why byte-exact parity is a poor MLX integration oracle](https://ineshin.space/papers/why-byte-exact-parity-is-a-poor-mlx-integration-oracle/)**
uses a pre-release FLUX.1 fixture failure to separate artifact drift, no-op parity, numerical
equivalence, and intentionally approximate output. It explains why paired same-process tests and
measured per-variant gates are more informative than committed real-model bytes.

**[Benchmarking above the recommended working set](https://ineshin.space/papers/benchmarking-above-the-recommended-working-set/)**
documents how the 0.10.0 Qwen-Image figures (3.02× against plain mflux, plus an unexplained 1.10×
with the gate off) came from a run whose MLX active memory peaked above the GPU's
recommended working set (24.96 GiB is the default the repository's comparison recipes use for a 32 GB M1 Max, not a recorded device value). That report has no date, commit field, dirty flag or MLX version; its install-time version string names commit 90fca33. The [0.11.0 run](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/v0.11.0_bench_qwen_image.json)
measured an active peak of 24.08–24.13 GiB after model load (the load phase reads about zero, so this is also the whole-run peak) and 2.68× combined. Its report has no date field or MLX version, and its working tree was not clean (Benchmarks footnote ¹).
That peak excludes cached memory; adding the cache reading taken after decode gives an estimate of roughly 25.1–25.2 GiB, computed rather than observed, and not an upper bound, because MLX's cache pool can overshoot its limit.
The 3.02× is above the 2.94× that 33 skipped steps of 50 allow (50/17). The record does not say why, and it does not explain the same report's 1.10× no-gate gap either. The later median ratio, 2.68×, is below 3.02×. The two runs differ in six things, and no run isolates any one of them as the cause of the lower ratio or the lower peak. Before trusting a
benchmark's timings, check that its peak, cached memory counted, stays under the recommended
working set.

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
