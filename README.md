# mlx-teacache

<p align="center">
  <img src="docs/assets/mlx-teacache-logo.png" alt="mlx-teacache" width="100%">
</p>

[![PyPI version](https://img.shields.io/pypi/v/mlx-teacache.svg)](https://pypi.org/project/mlx-teacache/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlx-teacache.svg)](https://pypi.org/project/mlx-teacache/)
[![License: Apache 2.0](https://img.shields.io/pypi/l/mlx-teacache.svg)](https://github.com/IonDen/mlx-teacache/blob/main/LICENSE)
[![CI](https://github.com/IonDen/mlx-teacache/actions/workflows/ci.yml/badge.svg)](https://github.com/IonDen/mlx-teacache/actions/workflows/ci.yml)

**TeaCache step-skipping for FLUX diffusion on Apple Silicon, in pure MLX.**

`mlx-teacache` is the first MLX port of TeaCache, a training-free inference optimization that predicts which denoising steps add little to the final image and reuses the previous step's output instead of running the full transformer. On FLUX.1-dev at 25 steps the polynomial gate skips 6 of 25 steps and produces a measured 1.46× wall-clock speedup with visually-equivalent output (SSIM ≥ 0.80 across a 5-prompt suite, ≥ 0.90 on the PR-gate prompt).

On FLUX.2 Klein at the distilled 4-8 step defaults the polynomial gate does not trigger any skips. Every adjacent-step body output change already exceeds the default threshold, so the gate signals "compute" every time. The wrapper still runs faster than vanilla mflux on distilled Klein on M1 Max — measured ~1.5-2.0× in v0.4-era same-process benches, with high thermal variance — but the win comes from sidestepping mflux's `mx.compile` of `_predict` rather than from caching. v0.6.0's subprocess-per-rep harness measured the compile-avoidance contribution at 1.01-1.02× on non-distilled klein-base at 50 steps + CFG, so the wide range applies specifically to the distilled 8-step schedules; longer schedules show a much smaller compile-avoidance effect. See [Benchmarks](#benchmarks) → "How the speedup happens" for the full mechanism breakdown.

## Which library do I need?

**You want FLUX generation to be faster on Apple Silicon?** You're in the right place. `mlx-teacache` skips redundant denoising steps on FLUX.1 and non-distilled FLUX.2 Klein — measured 1.46× on FLUX.1-dev at 25 steps. Drops into mflux via one line.

**You want live previews while generating, or low-memory latent decode?** You want [`mlx-taef`](https://github.com/IonDen/mlx-taef) — tiny TAESD-family decoders in MLX.

**You want both?** They compose cleanly. mflux 4-step Klein + TeaCache + TAEF2 previews = 1.30× wall-clock and 26% less peak memory vs vanilla.

## What it does

Diffusion models run the same big transformer 20-50 times in a loop. Between consecutive steps the output changes very little, and TeaCache uses a tiny polynomial fit to predict which steps can reuse the previous step's output. On M1 Max with FLUX.1-dev at 25 steps the default threshold (`rel_l1_thresh=0.20`) skips 6 of 25 steps and produces a 1.46× speedup.

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
pip install "mlx-teacache==0.9.0[mflux]"  # pin for reproducibility
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
| **0.20 (default)** | **6 / 25** | **1.46×** | **≥ 0.80 (5-prompt suite)** | **Visually-lossless sweet spot** |
| 0.25 | 11 / 25 | 1.96× | 0.57-0.93 | Visible style changes on text/synthetic prompts |

The committed benchmark `_artifacts/v0.6.3_bench_flux1_dev.json` validates the 0.20 default row; the other rows show single-run measurements, not a committed multi-rep sweep.

0.20 was picked after side-by-side visual comparison. At 0.25, text prompts that vanilla renders as neon tubes can come out as dot-matrix. At 0.20, the output is indistinguishable from vanilla and the cache still skips around 25% of steps. SSIM is conservative on high-frequency-detail prompts like text and synthetic patterns, which is why the suite floor (0.80) is lower than the PR-gate floor (0.90) on the red-apple prompt.

## Supported models

The table below is generated from the variant registry — see `docs/_generate_supported_models.py`. Per-variant detail (mflux constructor, coefficient provenance, quirks) lives in `docs/variants/<id>.md`.

<!-- SUPPORTED_MODELS_START -->
| Variant id | Display name | Distilled? | Default recipe | License |
|---|---|---|---|---|
| `flux1-dev` | FLUX.1 dev | no | 25 steps, g=3.5 | [FLUX.1-dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| `flux1-schnell` | FLUX.1 schnell | yes | 4 steps, g=1.0 | [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |
| `flux2-klein-4b` | FLUX.2 Klein 4B | yes | 8 steps, g=1.0 | [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| `flux2-klein-9b` | FLUX.2 Klein 9B | yes | 8 steps, g=1.0 | [FLUX Non-Commercial](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |
| `flux2-klein-base-4b` | FLUX.2 Klein base 4B | no | 50 steps, g=4.0 | [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) |
| `flux2-klein-base-9b` | FLUX.2 Klein base 9B | no | 50 steps, g=4.0 | [FLUX Non-Commercial](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) |
| `qwen-image` | Qwen-Image | no | 20 steps, g=4.0 | [Apache-2.0](https://huggingface.co/Qwen/Qwen-Image) |
| `z-image-base` | Z-Image base | no | 50 steps, g=4.0 | [Apache-2.0](https://huggingface.co/Tongyi-MAI/Z-Image) |
<!-- SUPPORTED_MODELS_END -->

### Per-variant notes

Each variant has its own page under [`docs/variants/`](docs/variants) — mflux constructor, recipe, license obligations, coefficient provenance, quirks. The highlights for the variants where behavior diverges from the default story:

**[`flux2-klein-9b`](docs/variants/flux2-klein-9b.md)** — coefficients are calibrated at `num_inference_steps=8`, origin-constrained polyfit. At the default threshold, the gate produces 0 step-skips on Klein 9B's 8-step schedule (the empirical adjacent-step body-output rel-L1 starts at 0.25 — above the 0.20 threshold). The library still helps via `mx.compile`-path avoidance (measured ~1.5-2.0× wall-clock improvement), and output quality is preserved (SSIM ≥ 0.85 PR-gate). See [Benchmarks](#benchmarks) → "How the speedup happens".

**[`flux2-klein-base-4b`](docs/variants/flux2-klein-base-4b.md)** — non-distilled FLUX.2 Klein 4B (Apache-2.0). TeaCache engages at `guidance=1.0` with a per-variant default `rel_l1_thresh=0.17`. At 25 steps the gate skips 3/25 steps and the wrapper measures 1.41× wall-clock vs vanilla (v0.4.0 same-process measurement, not yet re-bench'd under subprocess-per-rep); SSIM > 0.99 vs vanilla. CFG (`guidance > 1.0`) runs through a per-branch gated path as of v0.4.1: the canonical upstream recipe (`guidance_scale=4.0, num_inference_steps=50`) skips 9/50 steps for a **1.23× combined speedup** on M1 Max under v0.6.0's subprocess-per-rep harness (1.22× gating + 1.01× `mx.compile`-path avoidance). v0.4.1 advertised 1.26× combined with a 1.16× / 1.09× decomposition; the combined was honest within day-to-day noise but the decomposition over-attributed to compile-avoidance because the same-process harness left the wrapper inheriting warm allocator state.

**[`flux2-klein-base-9b`](docs/variants/flux2-klein-base-9b.md)** — non-distilled FLUX.2 Klein 9B (FLUX Non-Commercial — see [License obligations](#license-obligations) and accept on the [Hugging Face model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) before downloading). Reuses base-4b's polynomial coefficients verbatim and the same `rel_l1_thresh=0.17` default — justified by the shared FLUX.2-Klein architecture family and identical non-distilled 25-step / g=1.0 calibration recipe. v0.6.0's clean three-way bench at the canonical 50-step + g=4.0 recipe (subprocess-per-rep, M1 Max 32 GB, bf16, q4): **1.36× combined wall-clock** (517.6 s vanilla → 380.6 s wrapper median; 13/48 active steps skipped). Decomposed: 1.34× from gating (the v0.4.1 effect) + 1.02× from `mx.compile`-path avoidance (the v0.4 effect, small on M1 Max for this recipe). Wrapper peak memory ~10 GB vs vanilla's ~22 GB. SSIM 0.986 vs vanilla (carried over from v0.5.0 validation; visually equivalent).

> **Correction.** v0.5.0 advertised a 2.68× headline on this variant. That number was inflated by same-process MLX state leakage in the v0.5.x bench harness: vanilla ran genuinely cold while the wrapper inherited the warm MLX allocator state. v0.6.0's subprocess-per-rep harness makes every condition cold, and the honest number is 1.36×.

See `_artifacts/v0.6.0_bench_klein_base_9b.json` for the full report and `tests/_artifacts/bench_images/klein-base-9b/` for side-by-side images.

**[`z-image-base`](docs/variants/z-image-base.md)** — Z-Image base (Tongyi-MAI, Apache-2.0), a single-stream DiT and the first non-FLUX model with a TeaCache mini-kernel (added in v0.7.0). Its adaLN modulation is timestep-only, so there is no cheap caption-independent modulation input to gate on; the gate signal is the first-main-layer residual (calibrated in-repo as "Signal B", R² 0.400 — in line with the shipped FLUX.2 fits, and the threshold sweep rather than the fit R² sets the quality bar here; the caption-independent noise-refiner tap was tried and rejected at R² 0.069). Per-variant default `rel_l1_thresh=0.12`, set at the SSIM knee. At the 512×512 red-apple recipe (subprocess-per-rep, q8, 50 steps, g=4.0): **1.17× combined wall-clock** (245.3 s → 209.4 s; 15/48 active steps skipped), the win entirely from gating — `mx.compile`-path avoidance is not a tailwind on Z-Image. Peak memory drops 17.2 GB → 11.9 GB (from the eager wrapper bypassing mflux's compiled `_predict`, not from gating). SSIM 0.991 vs vanilla at this recipe. The [COMPARISON.md](COMPARISON.md) portrait row is a separate 640×896 generation at 1.33×.

See `scripts/_bench_z_image_v0_7_0.json` for the full report.

**[`qwen-image`](docs/variants/qwen-image.md)** — Qwen-Image base (Alibaba, Apache-2.0), a ~20B dual-stream MMDiT and the first variant that proxies `flux.transformer` (the FLUX.1 pattern) *and* runs true two-pass CFG. It's FLUX-shaped, so the gate taps the FLUX-canonical modulated block-0 input — calibrated in-repo at R² 0.946, well above Z-Image's 0.400 and the FLUX.2 family's 0.11–0.47. Per-variant default `rel_l1_thresh=0.25`, set at the SSIM knee. At the shared 512×512 portrait recipe (subprocess-per-condition, 3 reps, q4, 20 steps, g=4.0): **1.41× warm-median** (117.3 s → 83.4 s; 6 of 18 active steps skipped), SSIM 0.99 — the largest speedup of the [COMPARISON.md](COMPARISON.md) set. The win is entirely step-skipping: Qwen-Image's `_predict` is not `mx.compile`-wrapped in mflux, so there is no compile-path effect to separate out (unlike the FLUX.2 variants). The recipe is 512×512 because the 20B model peaks ~27.6 GB at q4 even there — the peak is weights-dominated, so 768×768 only adds ~0.7 GB and crosses the 32 GB ceiling. Reproduce with `uv run python scripts/bench_comparison.py --only qwen-image`.

## When to use mlx-teacache

The wrapper helps when the underlying schedule actually has cacheable redundancy. That is the case for non-distilled FLUX schedules with enough denoising steps for adjacent transformer outputs to look similar, which is what TeaCache's gate exploits.

In practice, that means:

- Use mlx-teacache for **`flux1-dev`** at 20-50 steps, the **non-distilled FLUX.2 Klein** family (`flux2-klein-base-4b`, `flux2-klein-base-9b`) at 20-50 steps with or without CFG, **`z-image-base`** at 50 steps with CFG, and **`qwen-image`** at 20 steps with CFG. These are the variants featured in [COMPARISON.md](COMPARISON.md), and the wrapper measurably skips steps and produces visually equivalent output.
- Do not reach for it on the **distilled** variants — `flux1-schnell` (4 steps), `flux2-klein-4b` and `flux2-klein-9b` at their distilled defaults (4-8 steps). The residual between adjacent steps is too large for the gate to engage at any reasonable threshold, so it skips zero steps and adds about 1-2% gating overhead. Run those through vanilla mflux.

There is a separate, incidental benefit on FLUX.2 variants regardless of whether the gate engages: the wrapper sidesteps mflux's compiled `_predict` path, which on Max and Ultra chips happens to be slower than the uncompiled path on the current MLX release. That is a wall-clock effect from compile avoidance, not from step-skipping, and we keep the two attributions separate in the docs.

See [COMPARISON.md](COMPARISON.md) for side-by-side images and warm-median wall-clock numbers on an M1 Max.

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

mlx-teacache implements this for mflux on Apple Silicon. For FLUX.1 we replace `flux.transformer` with a per-instance proxy; for FLUX.2 we replace `flux._predict` with an instance-level closure, which keeps gating live on chips where mflux would otherwise wrap `_predict` in `mx.compile`. The FLUX.1 polynomial coefficients are vendored from upstream. The FLUX.2 coefficients are derived in-repo by `scripts/calibrate_flux2.py` (distilled klein-4b and klein-9b, base-4b each have their own calibrated tuple; klein-base-9b cross-imports base-4b's tuple because the architectures share the calibration recipe). See [`docs/calibration.md`](docs/calibration.md) for procedure and provenance, and the per-variant docs under `docs/variants/` for each tuple's R² and origin date. The original method is described in the TeaCache paper at https://liewfeng.github.io/TeaCache/.

## Benchmarks

All numbers are reproducible via `scripts/bench_speedup.py`. M1 Max 32GB, macOS 26.x, mflux 0.17.5, bf16, quantize=4 (q8 on the `z-image-base` row — its pinned recipe), 512×512, `seed=42`, red-apple prompt; default `rel_l1_thresh=0.20` (per-variant default `0.17` on base-4b rows, `0.12` on z-image-base). Measurement dates: pre-v0.4.1 rows measured 2026-05-16; base-4b 25-step row measured 2026-05-17 (v0.4.0 harness); base-4b CFG row re-measured 2026-05-26 under v0.6.0's subprocess-per-rep harness (column footnote ⁴). Subprocess-per-rep means every (variant, condition, rep) gets a fresh Python interpreter so each timing starts from a cold MLX allocator; older same-process rows are flagged with their measurement era.

| Variant | Steps | Vanilla | Wrapper | Speedup | Skipped | Mechanism |
|---|---|---|---|---|---|---|
| `flux1-dev` | 25 | 103.8s | 71.0s | **1.46×** | **6 / 25** | TeaCache step-skipping¹ |
| `flux1-schnell` | — | — | — | — | — | shares dev's coefficients; gate behaves like dev at long schedules, like Klein at the 4-step distilled default (no benefit) |
| `flux2-klein-4b`† | 8 | 28.1s | 22.3s | 1.26× | **0 / 8** | `mx.compile` avoidance only |
| `flux2-klein-9b`† | 8 | 119.0s | 61.8s | 1.93× | **0 / 8** | `mx.compile` avoidance only |
| `flux2-klein-base-4b`³ | 25 | 77.5s | 55.1s | **1.41×** | **3 / 25** | step-skipping + `mx.compile` avoidance |
| `flux2-klein-base-4b` (CFG)⁴ | 50 | 236.2s | 191.8s | **1.23×** | **9 / 50** | step-skipping (compile-avoidance ≈ noise) |
| `flux2-klein-base-9b` (CFG)⁵ | 50 | 517.6s | 380.6s | **1.36×** | **13 / 50** | step-skipping + small compile-avoidance |
| `z-image-base` (CFG)⁶ | 50 | 245.3s | 209.4s | **1.17×** | **15 / 50** | step-skipping (q8; compile-avoidance not a tailwind) |

¹ `flux1-dev` at 25 steps, `guidance=3.5`, 512×512, default `rel_l1_thresh=0.20`. Measured 2026-05-31 under the subprocess-per-rep harness: **1.46× combined**, and the three-way split puts the whole win on gating: 1.47× from step-skipping, 1.00× from `mx.compile`-path avoidance (the eager wrapper and vanilla run neck-and-neck on this recipe). 6/25 skips across all three reps. Full report: `_artifacts/v0.6.3_bench_flux1_dev.json`. Reproduce with `uv run python scripts/bench_speedup.py --variant flux1-dev --three-way --reps 3 --report out.json`.

† Distilled klein rows are v0.4-era same-process measurements with high thermal variance. The klein-9b 1.93× median combined a thermally-throttled vanilla rep (227s) with a recovered wrapper rep (46s); the steady-state range across reps is roughly 1.5-2.0× depending on system load. The 0/8 skip count is stable across all reps. These rows are pending a re-bench under the v0.6.0 subprocess-per-rep harness; v0.6.0's measurement on klein-base CFG showed the compile-avoidance contribution at 1.01-1.02× on 50-step schedules, so the distilled 1.5-2.0× figure is specific to the 8-step distilled path and may be smaller under cold-isolation conditions.

³ `flux2-klein-base-4b` at `guidance=1.0`, per-variant default `rel_l1_thresh=0.17`. The 1.41× was measured under the v0.4.0 same-process harness; not yet re-bench'd under subprocess-per-rep. The combined number historically credited step-skipping (3/25 skips save ~12% directly) and `mx.compile`-path avoidance. Under v0.6.0's subprocess-per-rep harness on the related 50-step CFG recipe, compile-avoidance came out at 1.01× — so the 1.41× decomposition may shift toward "almost entirely step-skipping" when re-measured. CFG (`guidance > 1.0`) is gated end-to-end as of v0.4.1; see footnote ⁴.

⁴ `flux2-klein-base-4b` under CFG at the canonical upstream BFL recipe (`guidance=4.0`, 50 steps), per-variant default `rel_l1_thresh=0.17`. Re-measured under v0.6.0's subprocess-per-rep harness on 2026-05-26: **1.23× combined** = 1.22× from step-skipping (wrapped-no-gate vs wrapped-gated) × 1.01× from `mx.compile`-path avoidance (vanilla vs wrapped-no-gate). Compile-avoidance is effectively noise on this recipe — the v0.4.1-era 1.09× attribution was inflated by same-process MLX state leakage (vanilla ran cold, wrapper inherited warm allocator state). Skip count is byte-identical to v0.4.1 at 9/50 across three reps. Reproduce with `uv run python scripts/bench_speedup.py --variant klein-base-4b --three-way --reps 3 --report out.json`. Full report: `_artifacts/v0.6.0_bench_klein_base_4b.json`.

⁵ `flux2-klein-base-9b` under CFG at the canonical 50-step + g=4.0 recipe. Measured under v0.6.0's subprocess-per-rep harness: **1.36× combined** = 1.34× gating × 1.02× compile-avoidance. 13/50 skips stable across reps; SSIM 0.986 vs vanilla. Replaces v0.5.0's advertised 2.68× headline, which was inflated by same-process MLX state leakage in the v0.5.x harness (see [Per-variant notes](#per-variant-notes) → klein-base-9b correction blockquote, and `_artifacts/v0.6.0_bench_klein_base_9b.json` for the full report). Wrapper peak memory ~10 GB vs vanilla's ~22 GB. Reproduce with `uv run python scripts/bench_speedup.py --variant klein-base-9b --three-way --reps 3 --report out.json`.

⁶ `z-image-base` under CFG at 50 steps, `guidance=4.0`, 512×512, **q8** (its pinned recipe — the rest of the table is q4), per-variant default `rel_l1_thresh=0.12`. Measured under the subprocess-per-rep harness: **1.17× combined** (vanilla 245.3 s → wrapper 209.4 s median), entirely from step-skipping at 15 of the 48 active steps (the table's `15 / 50` is over nominal steps; 48 active = 50 minus the skip-first/skip-last windows), stable across all three reps. `mx.compile`-path avoidance is not a tailwind here — the no-gate wrapper measured no faster than vanilla, though the three-way decomposition is thermally confounded (conditions run in blocks, the no-gate block ran hotter), so only the net 1.17× is reported. Peak memory 17.2 GB → 11.9 GB, from the eager wrapper bypassing mflux's compiled `_predict` (the no-gate wrapper shows the same ~11.9 GB), not from gating. SSIM 0.991 vs vanilla. [COMPARISON.md](COMPARISON.md) reports a higher 1.33× for this variant at the 640×896 portrait recipe (14 skips) — the larger resolution amortizes the per-step gating overhead better. Full report: `scripts/_bench_z_image_v0_7_0.json`. Reproduce with `uv run python scripts/bench_speedup.py --variant z-image --three-way --reps 3 --report out.json`.

Reproduce any row:

```bash
uv run python scripts/bench_speedup.py --variant flux1-dev      # 25-step dev
uv run python scripts/bench_speedup.py --variant klein-4b       # 8-step Klein 4B
uv run python scripts/bench_speedup.py --variant klein-9b       # 8-step Klein 9B
uv run python scripts/bench_speedup.py --variant klein-base-4b  # 50-step base-4B under CFG (g=4.0, v0.4.1+ default)
uv run python scripts/bench_speedup.py --variant klein-base-4b --guidance 1.0 --num-inference-steps 25  # v0.4.0 row
uv run python scripts/bench_speedup.py --variant klein-base-9b  # 50-step base-9B under CFG (g=4.0, v0.5.0+ default)
uv run python scripts/bench_speedup.py --variant z-image        # 50-step Z-Image base under CFG (g=4.0, q8)
```

For the three-way decomposition (vanilla / wrapped-no-gate / wrapped-gated), add `--three-way --reps 3 --report out.json`. This is how the v0.6.0 base-4b and base-9b numbers were produced.

### How the speedup happens

The wall-clock improvement above comes from two distinct mechanisms; they fire independently depending on variant and schedule.

**1. TeaCache step-skipping.** This is the headline feature. The polynomial gate predicts how much the transformer body output will change since the last actual compute step. When the accumulated predicted change stays below `rel_l1_thresh`, the wrapper reuses the cached residual instead of running the body again. On FLUX.1-dev at 25 steps, 6 of 25 steps are skippable and this is where the 1.46× speedup on FLUX.1-dev comes from. On non-distilled FLUX.2 Klein at 50 steps + CFG, the same mechanism produces 9-13 skips per generation and dominates the wall-clock win (the v0.6.0 three-way bench attributes 1.22-1.34× to gating on base-4b/9b).

**2. `mx.compile` avoidance on FLUX.2.** mflux wraps `Flux2Klein._predict` in `mx.compile` on every chip *except* base + Pro M1/M2 — i.e., compilation is active on M1/M2 Max + Ultra and on every M3, M4, M5 chip. mlx-teacache replaces the compiled `_predict` with an eager Python closure so the gate can run live per step. The magnitude of this effect varies by schedule and chip. v0.4-era same-process benches on M1 Max at quantize=4 showed the eager closure running 1.5-2.0× faster than the compiled path on distilled klein's 8-step schedule even with zero gate-engagements. v0.6.0's subprocess-per-rep harness re-measured the same mechanism on the 50-step klein-base CFG recipe and found it contributing only 1.01-1.02× — kernel-dispatch round-trips drop slightly, but with 50 steps per generation that gain is small relative to the per-step compute. The distilled schedules likely benefit more because each step is a larger fraction of the wall-clock, so per-step dispatch overhead matters more. The distilled klein rows are still measured under same-process conditions and pending a re-bench. On chips where mflux is already eager (base + Pro M1/M2), this mechanism does not fire: the wrapper just adds per-step gate overhead, and Klein with mlx-teacache on those chips is approximately neutral or slightly slower than vanilla.

On FLUX.2 Klein 4B and 9B at the distilled 4-8 step defaults, mechanism (1) does not engage: the empirical adjacent-step rel-L1 between consecutive transformer outputs is ≥ 0.25, so every step's predicted change exceeds the default 0.20 threshold and the gate signals "compute" every time. This is expected — distilled schedules collapse the entire denoising trajectory into a handful of consequential steps, so adjacent steps are not similar enough to skip. Klein's wall-clock improvement on these variants is real and reproducible, but it comes entirely from mechanism (2).

For algorithmic step-skipping on FLUX.2, use the non-distilled `flux2-klein-base-4b` (Apache-2.0) or `flux2-klein-base-9b` (FLUX Non-Commercial) variants. Both ship with a per-variant default `rel_l1_thresh=0.17`. At the canonical upstream 50-step + g=4.0 CFG recipe, base-4b skips 9/50 steps for a measured 1.23× combined speedup (v0.6.0 subprocess-per-rep), and base-9b skips 13/50 steps for 1.36×. Pushing the threshold higher on distilled Klein is not recommended: the gate's prediction quality at thresholds > 0.25 is uncalibrated on a 4-8 step trajectory and image quality is not characterized there.

### SSIM suite

Quality gates use a 5-prompt SSIM suite defined at `tests/test_image_quality_flux1.py:45` and reused at `tests/test_image_quality_flux2.py:28`:

- "a red apple on a wooden table"
- "mountain landscape at sunset"
- "portrait of a woman"
- "abstract pattern with circles"
- "text saying HELLO"

The PR-gate prompt is the red-apple one; SSIM ≥ 0.90 on FLUX.1-dev and ≥ 0.85 on Klein 4B / 9B at the default threshold. Full suite floor is 0.80 to absorb high-frequency-detail variance (text, synthetic patterns). Run `uv run pytest tests/test_image_quality_flux1.py tests/test_image_quality_flux2.py -m parity` with real model weights to reproduce.

## Performance by chip

mflux 0.17.5 wraps `_predict` in `mx.compile` on every Apple Silicon chip *except* base + Pro M1/M2. The `is_m1_or_m2()` predicate returns true (eager path) when the chip brand contains "Apple M1" or "Apple M2" *and* does not contain "Max" or "Ultra" — so M1 Pro and M2 Pro are eager too, while M1/M2 Max + Ultra and every M3/M4/M5 chip get the compiled path. mlx-teacache replaces `_predict` with an eager closure so per-step gating stays live. On compiled chips that gives up the compile gain to get the skip gain. See `docs/m3-plus-tradeoff.md` for a benchmark recipe.

| Chip | Vanilla `_predict` in mflux 0.17.5 | Expected speedup |
|---|---|---|
| Apple M1 / M2 (base) | eager | ≈ pure skip fraction (~1.5–1.6×) |
| M1 Pro / M2 Pro | eager | ≈ pure skip fraction — same as base |
| M1 Max / Ultra, M2 Max / Ultra | compiled | **1.46× measured** on M1 Max FLUX.1-dev / 25 steps; 1.23× on klein-base-4b 50-step CFG; 1.36× on klein-base-9b 50-step CFG |
| M3 / M3 Pro / M3 Max / Ultra | compiled | Likely 1.1–1.3× — untested |
| M4 / M4 Pro / M4 Max | compiled | Likely 1.1–1.3× — untested |
| M5+ (Neural Accelerators / TensorOps) | compiled + accelerator | May approach 1.0×. The eager wrapper can lose some or all of the M5 TensorOps advantage. Confirm with a profiler before treating as fact. |

## Limitations

img2img reuses the txt2img calibration. A dedicated img2img calibration may follow in a future release if SSIM gates flag drift on specific schedules.

FLUX.2 with CFG (`guidance > 1.0`) runs through the gated path as of v0.4.1. The wrapper keeps two cached residuals (positive and negative branch) and shares one gate decision per step across both. The canonical base-4b recipe (`guidance_scale=4.0, num_inference_steps=50`) measures **1.23× combined** on M1 Max under v0.6.0's subprocess-per-rep harness (v0.4.1 advertised 1.26× combined under same-process measurement; combined within day-to-day noise, but the decomposition shifted — see Benchmarks footnote ⁴).

**Distilled schedules are out of scope for algorithmic step-skipping by design.** This includes FLUX.2 Klein 4B / 9B at their 4-8 step defaults and FLUX.1 schnell at its 4-step default. The polynomial gate's premise — that consecutive transformer outputs are similar enough that the residual can be reused — does not hold on distilled trajectories where each step does a much larger share of the denoising work. On the v0.3.0 bench (M1 Max, quantize=4) the gate signals "compute" on every Klein step at the package default `rel_l1_thresh=0.20` (0 skips across 3 reps on both Klein 4B and 9B); empirical adjacent-step body-output rel-L1 on Klein is ≥ 0.25. Klein still gets a real wall-clock improvement (~1.2-1.9×) from `mx.compile`-path avoidance, but the headline TeaCache step-skipping feature only fires on non-distilled schedules.

`flux2-klein-base-4b` runs TeaCache at both `guidance=1.0` (single-branch path) and `guidance > 1.0` (per-branch path, v0.4.1+). The upstream BFL base-4b model card recommends `guidance_scale=4.0, num_inference_steps=50`; v0.6.0's subprocess-per-rep harness measures **1.23× combined wall-clock** vs vanilla on M1 Max at that recipe (1.22× gating + 1.01× compile-avoidance; 9/50 skips, SSIM PR-gate passed). v0.4.1's same-process bench reported 1.26× combined and decomposed it 1.16× gating / 1.09× compile-avoidance; the combined was honest within day-to-day noise but the decomposition over-attributed to compile-avoidance.

`flux2-klein-base-9b` reuses base-4b's polynomial coefficients verbatim (same architecture family, same calibration recipe). v0.6.0's subprocess-per-rep bench at the canonical 50-step / guidance=4.0 recipe measures **1.36× combined wall-clock** (1.34× gating + 1.02× compile-avoidance), 13/48 active steps skipped at `rel_l1_thresh=0.17`, SSIM 0.986 vs vanilla. v0.5.0 advertised 2.68× on this variant — that number was inflated by same-process MLX state leakage and is corrected here. Same FLUX Non-Commercial license + BFL safety-filter obligations as `flux2-klein-9b` — see [License obligations](#license-obligations).

The wrapper runs eager, which gives up mflux's `mx.compile` of `_predict` in exchange for live per-step gating. Vanilla mflux compiles `_predict` on every chip except base + Pro M1/M2 (the `is_m1_or_m2()` predicate only excludes Max + Ultra). The 1.46× measurement is from M1 Max / FLUX.1-dev / 25 steps; speedup on M3 and newer is plausible but untested locally. On M5, the GPU Neural Accelerators (Metal 4 TensorOps) are only reachable through the compiled path, so the eager wrapper can lose some or all of that advantage. Output stays correct either way. See `docs/m3-plus-tradeoff.md` for the per-chip recipe; PRs with measurements welcome.

FLUX.2 parity is numerical, not bit-exact. Replacing a function that mflux wraps in `mx.compile` produces about 1 ULP per element of divergence from Metal kernel-dispatch noise, which compounds across steps but keeps cosine similarity ≥ 0.97 on Klein 4B, Klein 9B, and base-4b under CFG at threshold 0. The user-facing guarantee is end-to-end image quality (SSIM ≥ 0.85 on all supported FLUX.2 variants at the package default threshold).

The mflux pin is strict at `>=0.17,<0.18`. Bumping it is a deliberate release.

Calling `flux.parameters()` at the parent level can miss transformer parameters while the wrapper is active. Use `flux.transformer.parameters()` directly, or call `handle.restore()` first.

## License obligations

The FLUX.1 variants (`flux1-dev`, `flux1-schnell`) and `flux2-klein-4b` come with their own upstream weight licenses; the wrapper this library applies does not change those terms.

`flux2-klein-9b` and `flux2-klein-base-9b` are both distributed under the FLUX.2 Klein license (non-commercial use + BFL safety-filter obligations). These terms flow with the weights, not with mlx-teacache. If you call `apply_teacache` on either variant — `Flux2Klein(model_config=ModelConfig.flux2_klein_9b())` or `Flux2Klein(model_config=ModelConfig.flux2_klein_base_9b())` — you are responsible for ensuring your use complies with the upstream license, including the safety-filter requirements the BFL model cards describe. See the official model cards at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B and https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B for the full terms.

## Contributing

Open an issue at https://github.com/IonDen/mlx-teacache/issues.

## License + acknowledgements

Apache-2.0. See `LICENSE` and `NOTICE`.

- [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache) — upstream method and FLUX.1 coefficients.
- [filipstrand/mflux](https://github.com/filipstrand/mflux) — MLX FLUX runner this library integrates with.
- [Apple ML Explore](https://github.com/ml-explore/mlx) — MLX.

---

By Denis Ineshin · [ineshin.space](https://ineshin.space)
