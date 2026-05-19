# mlx-teacache

[![PyPI version](https://img.shields.io/pypi/v/mlx-teacache.svg)](https://pypi.org/project/mlx-teacache/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlx-teacache.svg)](https://pypi.org/project/mlx-teacache/)
[![License: Apache 2.0](https://img.shields.io/pypi/l/mlx-teacache.svg)](https://github.com/IonDen/mlx-teacache/blob/main/LICENSE)
[![CI](https://github.com/IonDen/mlx-teacache/actions/workflows/ci.yml/badge.svg)](https://github.com/IonDen/mlx-teacache/actions/workflows/ci.yml)

**TeaCache step-skipping for FLUX diffusion on Apple Silicon, in pure MLX.**

`mlx-teacache` is the first MLX port of TeaCache, a training-free inference optimization that predicts which denoising steps add little to the final image and reuses the previous step's output instead of running the full transformer. On FLUX.1-dev at 25 steps the polynomial gate skips 6 of 25 steps and produces a measured 1.44× wall-clock speedup with visually-equivalent output (SSIM ≥ 0.80 across a 5-prompt suite, ≥ 0.90 on the PR-gate prompt).

On FLUX.2 Klein at the distilled 4-8 step defaults the polynomial gate does not trigger any skips. Every adjacent-step body output change already exceeds the default threshold, so the gate signals "compute" every time. The wrapper still runs ~1.3-1.9× faster than vanilla mflux on Klein, but the win comes from sidestepping mflux's `mx.compile` of `_predict` rather than from caching. See [Benchmarks](#benchmarks) → "How the speedup happens" and the [postmortem](docs/superpowers/notes/2026-05-16-flux2-teacache-non-engagement-postmortem.md) for the full story.

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
pip install "mlx-teacache==0.4.1[mflux]"  # pin for reproducibility
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
| `flux2-klein-base-4b`² | `Flux2Klein(model_config=ModelConfig.flux2_klein_base_4b())` | in-repo (25-step calibration, origin-constrained; see [`docs/calibration.md`](docs/calibration.md)) |
| `flux2-klein-base-9b`³ | `Flux2Klein(model_config=ModelConfig.flux2_klein_base_9b())` | reused from base-4b; validated empirically (see [`docs/calibration.md`](docs/calibration.md)) — see [License obligations](#license-obligations) |

¹ `flux2-klein-9b` coefficients are calibrated at `num_inference_steps=8`, origin-constrained polyfit. At the default threshold, the gate produces 0 step-skips on Klein 9B's 8-step schedule (the empirical adjacent-step body-output rel-L1 starts at 0.25 — above the 0.20 threshold). The library still helps via `mx.compile`-path avoidance (measured ~1.5-2.0× wall-clock improvement), and output quality is preserved (SSIM ≥ 0.85 PR-gate). See the [Benchmarks](#benchmarks) "How the speedup happens" subsection.

² `flux2-klein-base-4b` is the non-distilled FLUX.2 Klein 4B variant (Apache-2.0). TeaCache engages at `guidance=1.0` with a per-variant default `rel_l1_thresh=0.17`. At 25 steps the gate skips 3/25 steps and the wrapper measures 1.41× wall-clock vs vanilla (~12% from step-skipping plus the FLUX.2 `mx.compile`-path avoidance contribution); SSIM > 0.99 vs vanilla. CFG (`guidance > 1.0`) runs through a per-branch gated path as of v0.4.1: the canonical upstream recipe (`guidance_scale=4.0, num_inference_steps=50`) skips 9/50 steps for a 1.26× wall-clock speedup vs vanilla mflux on M1 Max.

³ `flux2-klein-base-9b` is the non-distilled FLUX.2 Klein 9B variant (FLUX Non-Commercial license — see [License obligations](#license-obligations) and accept on the [Hugging Face model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) before downloading). Ships in v0.5.0 reusing base-4b's polynomial coefficients verbatim and the same `rel_l1_thresh=0.17` default — justified by the shared FLUX.2-Klein architecture family and identical non-distilled 25-step / g=1.0 calibration recipe. Validated empirically at the canonical 50-step CFG recipe (subprocess-isolated cold rep on M1 Max 32GB, bf16, q4): wrapper measures **2.68× wall-clock vs vanilla mflux** (vanilla 2744s, wrapper 1025s; 12/48 active steps skipped at `rel_l1_thresh=0.17`); SSIM 0.986 vs vanilla. The 2.68× combines step-skipping with the FLUX.2 `mx.compile`-path avoidance contribution; clean attribution between the two mechanisms is deferred to v0.5.1. See `_artifacts/validation_klein_base_9b.json` for the full evidence and `_artifacts/validation_klein_base_9b_images/` for the side-by-side images.

## When to use mlx-teacache

The wrapper helps when the underlying schedule actually has cacheable redundancy. That is the case for non-distilled FLUX schedules with enough denoising steps for adjacent transformer outputs to look similar, which is what TeaCache's gate exploits.

In practice, that means:

- Use mlx-teacache for **`flux1-dev`** at 20-50 steps and the **non-distilled FLUX.2 Klein** family (`flux2-klein-base-4b`, `flux2-klein-base-9b`) at 20-50 steps, with or without CFG. These are the variants featured in [COMPARISON.md](COMPARISON.md), and the wrapper measurably skips steps and produces visually equivalent output.
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

mlx-teacache implements this for mflux on Apple Silicon. For FLUX.1 we replace `flux.transformer` with a per-instance proxy; for FLUX.2 we replace `flux._predict` with an instance-level closure, which keeps gating live on chips where mflux would otherwise wrap `_predict` in `mx.compile`. The FLUX.1 polynomial coefficients are vendored from upstream. The FLUX.2 Klein 4B and 9B coefficients are derived in-repo by `scripts/calibrate_flux2.py --variant klein-4b` / `--variant klein-9b`; see [`docs/calibration.md`](docs/calibration.md) for procedure and provenance. The original method is described in the TeaCache paper at https://liewfeng.github.io/TeaCache/.

## Benchmarks

All numbers are reproducible via `scripts/bench_speedup.py`. M1 Max 32GB, macOS 26.x, mflux 0.17.5, bf16, quantize=4, 512×512, `seed=42`, red-apple prompt; one vanilla warmup + 3 timed reps per condition, median reported, default `rel_l1_thresh=0.20` (per-variant default `0.17` on base-4b rows). Pre-v0.4.1 rows measured 2026-05-16; the base-4b CFG row measured 2026-05-17.

| Variant | Steps | Vanilla | Wrapper | Speedup | Skipped | Mechanism |
|---|---|---|---|---|---|---|
| `flux1-dev` | 25 | 103.7s | 71.8s | **1.44×** | **6 / 25** | TeaCache step-skipping |
| `flux1-schnell` | — | — | — | — | — | shares dev's coefficients; gate behaves like dev at long schedules, like Klein at the 4-step distilled default (no benefit) |
| `flux2-klein-4b` | 8 | 28.1s | 22.3s | 1.26× | **0 / 8** | `mx.compile` avoidance only |
| `flux2-klein-9b` | 8 | 119.0s | 61.8s | 1.93×† | **0 / 8** | `mx.compile` avoidance only |
| `flux2-klein-base-4b`³ | 25 | 77.5s | 55.1s | **1.41×** | **3 / 25** | step-skipping + `mx.compile` avoidance |
| `flux2-klein-base-4b` (CFG)⁴ | 50 | 254.0s | 201.2s | **1.26×** | **9 / 50** | step-skipping + `mx.compile` avoidance |

† Klein 9B wall-clock has high variance from thermal throttling on M1 Max at quantize=4. The 1.93× median combined a thermally-throttled vanilla rep (227s) with a recovered wrapper rep (46s); the steady-state range across reps is roughly 1.5-2.0× depending on system load. The 0/8 skip count is stable across all reps.

³ `flux2-klein-base-4b` at `guidance=1.0`, per-variant default `rel_l1_thresh=0.17`. The 1.41× combines both FLUX.2 speedup mechanisms — step-skipping (3/25 step skips save ~12% directly) and `mx.compile`-path avoidance (the same mechanism that gives distilled Klein its 1.2-1.9× wall-clock benefit on M1 Max at quantize=4). CFG (`guidance > 1.0`) is gated end-to-end as of v0.4.1; see footnote ⁴.

⁴ `flux2-klein-base-4b` under CFG at the canonical upstream BFL recipe (`guidance=4.0`, 50 steps), per-variant default `rel_l1_thresh=0.17`. The 1.26× combined ratio splits into 1.09× from `mx.compile`-path avoidance (vanilla mflux vs wrapped with the gate disabled, `rel_l1_thresh=0`) and 1.16× from step-skipping (wrapped-no-gate vs wrapped-gated). The gating contribution is lower than the v0.4.0 25-step row because CFG runs two transformer calls per step (positive + negative branch); a gated step still saves both calls in full, but each saved call is individually more expensive. Skip count is stable at 9/50 across three reps. Reproduce with `uv run python scripts/bench_speedup.py --variant klein-base-4b` (defaults to the CFG recipe in v0.4.1+).

Reproduce any row:

```bash
uv run python scripts/bench_speedup.py --variant flux1-dev      # 25-step dev
uv run python scripts/bench_speedup.py --variant klein-4b       # 8-step Klein 4B
uv run python scripts/bench_speedup.py --variant klein-9b       # 8-step Klein 9B
uv run python scripts/bench_speedup.py --variant klein-base-4b  # 50-step base-4B under CFG (g=4.0, v0.4.1 default)
uv run python scripts/bench_speedup.py --variant klein-base-4b --guidance 1.0 --num-inference-steps 25  # v0.4.0 row
```

### How the speedup happens

The wall-clock improvement above comes from two distinct mechanisms; they fire independently depending on variant and schedule.

**1. TeaCache step-skipping.** This is the headline feature. The polynomial gate predicts how much the transformer body output will change since the last actual compute step. When the accumulated predicted change stays below `rel_l1_thresh`, the wrapper reuses the cached residual instead of running the body again. On FLUX.1-dev at 25 steps, 6 of 25 steps are skippable and this is where the 1.44× speedup comes from.

**2. `mx.compile` avoidance on FLUX.2.** mflux wraps `Flux2Klein._predict` in `mx.compile` on every chip *except* base + Pro M1/M2 — i.e., compilation is active on M1/M2 Max + Ultra and on every M3, M4, M5 chip. mlx-teacache replaces the compiled `_predict` with an eager Python closure so the gate can run live per step. On M1 Max at quantize=4, the eager closure happens to be ~1.2-1.9× faster than the compiled path even when zero steps get skipped — kernel-dispatch round-trips drop, and there's no recompile pressure when input shapes change between generations. On chips where mflux is already eager (base + Pro M1/M2), this mechanism does not fire: the wrapper just adds per-step gate overhead, and Klein with mlx-teacache on those chips is approximately neutral or slightly slower than vanilla. The benefit on M1/M2 base + Pro is only the step-skipping benefit, which requires longer schedules than Klein's distilled 8.

On FLUX.2 Klein 4B and 9B at the distilled 4-8 step defaults, mechanism (1) does not engage: the empirical adjacent-step rel-L1 between consecutive transformer outputs is ≥ 0.25, so every step's predicted change exceeds the default 0.20 threshold and the gate signals "compute" every time. This is expected — distilled schedules collapse the entire denoising trajectory into a handful of consequential steps, so adjacent steps are not similar enough to skip. Klein's wall-clock improvement on these variants is real and reproducible, but it comes entirely from mechanism (2). See [`docs/superpowers/notes/2026-05-16-flux2-teacache-non-engagement-postmortem.md`](docs/superpowers/notes/2026-05-16-flux2-teacache-non-engagement-postmortem.md) for the investigation.

For algorithmic step-skipping on FLUX.2, use the non-distilled `flux2-klein-base-4b` variant (Apache-2.0, runs at 20-50 steps) — shipped in v0.4.0 with a per-variant default `rel_l1_thresh=0.17` that skips 3/25 steps. The wrapper measures 1.41× wall-clock on M1 Max at 25 steps; the speedup combines both FLUX.2 mechanisms (~12% from step-skipping itself, the rest from `mx.compile`-path avoidance). Pushing the threshold higher on distilled Klein is not recommended: the gate's prediction quality at thresholds > 0.25 is uncalibrated on a 4-8 step trajectory and image quality is not characterized there.

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
| M1 Max / Ultra, M2 Max / Ultra | compiled | **1.48× measured** on M1 Max FLUX.1-dev / 25 steps |
| M3 / M3 Pro / M3 Max / Ultra | compiled | Likely 1.1–1.3× — untested |
| M4 / M4 Pro / M4 Max | compiled | Likely 1.1–1.3× — untested |
| M5+ (Neural Accelerators / TensorOps) | compiled + accelerator | May approach 1.0×. The eager wrapper can lose some or all of the M5 TensorOps advantage. Confirm with a profiler before treating as fact. |

## Limitations

img2img reuses the txt2img calibration. A dedicated img2img calibration may follow in a future release if SSIM gates flag drift on specific schedules.

FLUX.2 with CFG (`guidance > 1.0`) runs through the gated path as of v0.4.1. The wrapper keeps two cached residuals (positive and negative branch) and shares one gate decision per step across both. The canonical base-4b recipe (`guidance_scale=4.0, num_inference_steps=50`) measures 1.26× on M1 Max.

**Distilled schedules are out of scope for algorithmic step-skipping by design.** This includes FLUX.2 Klein 4B / 9B at their 4-8 step defaults and FLUX.1 schnell at its 4-step default. The polynomial gate's premise — that consecutive transformer outputs are similar enough that the residual can be reused — does not hold on distilled trajectories where each step does a much larger share of the denoising work. On the v0.3.0 bench (M1 Max, quantize=4) the gate signals "compute" on every Klein step at the package default `rel_l1_thresh=0.20` (0 skips across 3 reps on both Klein 4B and 9B); empirical adjacent-step body-output rel-L1 on Klein is ≥ 0.25. Klein still gets a real wall-clock improvement (~1.2-1.9×) from `mx.compile`-path avoidance, but the headline TeaCache step-skipping feature only fires on non-distilled schedules. See the postmortem at `docs/superpowers/notes/2026-05-16-flux2-teacache-non-engagement-postmortem.md`.

`flux2-klein-base-4b` runs TeaCache at both `guidance=1.0` (single-branch path) and `guidance > 1.0` (per-branch path, v0.4.1+). The upstream BFL base-4b model card recommends `guidance_scale=4.0, num_inference_steps=50`; v0.4.1 measures 1.26× wall-clock vs vanilla on M1 Max at that recipe (9/50 skips, SSIM PR-gate passed).

`flux2-klein-base-9b` is not yet supported. Planned for v0.5.0 (FLUX Non-Commercial license + BFL safety filter).

The wrapper runs eager, which gives up mflux's `mx.compile` of `_predict` in exchange for live per-step gating. Vanilla mflux compiles `_predict` on every chip except base + Pro M1/M2 (the `is_m1_or_m2()` predicate only excludes Max + Ultra). The 1.48× measurement is from M1 Max / FLUX.1-dev / 25 steps; speedup on M3 and newer is plausible but untested locally. On M5, the GPU Neural Accelerators (Metal 4 TensorOps) are only reachable through the compiled path, so the eager wrapper can lose some or all of that advantage. Output stays correct either way. See `docs/m3-plus-tradeoff.md` for the per-chip recipe; PRs with measurements welcome.

FLUX.2 parity is numerical, not bit-exact. Replacing a function that mflux wraps in `mx.compile` produces about 1 ULP per element of divergence from Metal kernel-dispatch noise, which compounds across steps but keeps cosine similarity ≥ 0.97 on Klein 4B, Klein 9B, and base-4b under CFG at threshold 0. The user-facing guarantee is end-to-end image quality (SSIM ≥ 0.85 on all supported FLUX.2 variants at the package default threshold).

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
