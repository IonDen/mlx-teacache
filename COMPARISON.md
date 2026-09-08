# Side-by-side: vanilla mflux vs mlx-teacache

Visual showcase of what the wrapper does on real generations. Two harnesses contribute to this page:

- The `flux1-dev`, `flux2-klein-base-4b`, `z-image-base`, and `qwen-image` rows are from `scripts/bench_comparison.py` (subprocess-per-condition, three reps per subprocess; cold = rep 1, warm = median of reps 2 and 3). Full report: `_artifacts/comparison_report.json`. Committed images alongside this file under `_artifacts/comparison/<variant>/`. They share the portrait prompt and seed below; `z-image-base` renders at 640×896 q8 and `qwen-image` at 768×768 (a mixed-precision build, for quality on 32 GB — see its row), rather than the page's 768×1024.
- The `flux2-klein-base-9b` row is from `scripts/bench_speedup.py --three-way --reps 3` (subprocess-per-rep — every rep gets a fresh interpreter, fully cold). Full report: `_artifacts/v0.10.0_bench_klein_base_9b.json`. Regenerate the side-by-side images with `scripts/bench_comparison.py`.

The two harnesses use different prompts and resolutions (the bench_comparison rows use the 768×1024 portrait listed below, except z-image-base at 640×896 and qwen-image at 768×768; bench_speedup is the 512×512 red-apple recipe described in the README's Benchmarks section). Cross-reading the numbers across rows therefore requires care.

The `flux1-dev`, `flux2-klein-base-4b`, and `z-image-base` images were regenerated on 2026-08-16 under the v0.10.0 gate and came out byte-identical to the files the previous release committed, so the timings below are re-measured while the pictures are the same ones you saw before.

The `qwen-image` row is different, and should not be read as one of those. It is carried over from its v0.9.0 measurement and v0.10.0 would **not** reproduce it: the corrected gate anchoring makes Qwen skip more of its schedule, so both its timing and its image move. Measured on the red-apple recipe, Qwen goes from 24 skipped steps to 33 and from SSIM 0.978 to 0.967 against vanilla. The portrait below was not re-run — the recipe peaks above 30 GB on this 32 GB machine — so it stands as a record of what 0.9.x produced, not of what this release does.

Only non-distilled variants are listed here. Distilled schedules (`flux1-schnell`, `flux2-klein-4b`, `flux2-klein-9b`) skip zero steps and gain nothing from the wrapper. See the "When to use mlx-teacache" section in the README for the recommendation.

## Test machine

Apple M1 Max, 32 GB unified memory. Models loaded at `quantize=4` in bf16 (q8 for the `z-image-base` row — its pinned recipe). The `flux1-dev`, `flux2-klein-base-4b`, `flux2-klein-base-9b`, and `z-image-base` rows were measured under mlx-teacache 0.10.0 on mflux 0.18.0 (macOS Darwin 25.4.0, and 25.6.0 for the 9B row); the `qwen-image` row is the earlier mlx-teacache 0.9.0 measurement on mflux 0.17.5.

Shared inputs across every cell:

- **Prompt:** *"Portrait of a young woman with auburn hair and green eyes, soft golden-hour window light, photorealistic, shallow depth of field, 50mm prime lens, subtle freckles, neutral background, cinematic color grading."*
- **Seed:** 42
- **Resolution:** 768 × 1024 (portrait) — `z-image-base` is the exception at 640 × 896 (q8 memory; see its section)
- **Image format on disk:** webp, quality 88, method 6

## FLUX.1 family

### `flux1-dev` — 25 steps, guidance=3.5

|  | Vanilla mflux | mlx-teacache |
|---|---|---|
| Time (cold) | 239.1 s | 198.9 s |
| Time (warm median) | 235.5 s | 198.1 s |
| Warm speedup | — | **1.19×** |
| Steps skipped | 0 of 23 active | 4 of 23 active (never two in a row) |
| Threshold | — | rel_l1 = 0.20 |
| Image | ![vanilla](_artifacts/comparison/flux1-dev/vanilla.webp) | ![wrapper](_artifacts/comparison/flux1-dev/wrapper.webp) |

FLUX.1 dev runs the schedule that mlx-teacache was originally calibrated for. The wrapper drops four transformer evaluations out of the 23 active steps (the first two are always computed; the algorithm gates from step 2 onward). Output is visually indistinguishable from vanilla on this prompt.

## FLUX.2 family

### `flux2-klein-base-4b` — 50 steps, guidance=4.0

|  | Vanilla mflux | mlx-teacache |
|---|---|---|
| Time (cold) | 575.1 s | 485.4 s |
| Time (warm median) | 573.5 s | 485.7 s |
| Warm speedup | — | **1.18×** |
| Steps skipped | 0 of 48 active | 8 of 48 active (never two in a row) |
| Threshold | — | rel_l1 = 0.17 |
| Image | ![vanilla](_artifacts/comparison/klein-base-4b-cfg/vanilla.webp) | ![wrapper](_artifacts/comparison/klein-base-4b-cfg/wrapper.webp) |

This is the canonical FLUX.2 Klein base setting from Black Forest Labs: 50 steps with real classifier-free guidance at 4.0. The wrapper skips eight of the 48 active transformer evaluations under both the conditional and unconditional branches (per-branch caching, the v0.4.1 feature). The two portraits are perceptually equivalent at output size; close inspection shows slightly softer microtexture in the wrapped version, which is the expected trade for the skipped steps.

### `flux2-klein-base-9b` — 50 steps, guidance=4.0

|  | Vanilla mflux | mlx-teacache |
|---|---|---|
| Time (median of 3 reps)¹ | 520.6 s | 379.1 s |
| Speedup | — | **1.37×** |
| Steps skipped | 0 of 48 active | 13 of 48 active (never two in a row) |
| Threshold | — | rel_l1 = 0.17 |
| Peak memory | ~22 GB | ~9.5 GB |
| SSIM vs vanilla | — | **0.986** |
| Image | ![vanilla](_artifacts/validation_klein_base_9b_images/vanilla.webp) | ![wrapper](_artifacts/validation_klein_base_9b_images/wrapper.webp) |

¹ Subprocess-per-rep bench on M1 Max 32 GB, bf16, q4 — every (variant, condition, rep) runs in a fresh Python interpreter so each timing starts from a cold MLX allocator. Full report: `_artifacts/v0.10.0_bench_klein_base_9b.json` (2026-08-15, mflux 0.18.0; v0.6.0 measured 1.36× at the same recipe). v0.5.0 advertised 2.68× on this same recipe; that number was inflated by same-process MLX state leakage in the v0.5.x bench harness — vanilla ran cold while the wrapper inherited warm allocator state. Subprocess isolation exposes the honest 1.36–1.37×.

`flux2-klein-base-9b` is the non-distilled FLUX.2 Klein 9B variant (FLUX Non-Commercial license — see [README License obligations](README.md#license-obligations) and accept on the [Hugging Face model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) before downloading). Reuses base-4b's polynomial coefficients verbatim — same FLUX.2 Klein architecture family, same calibration recipe. The 1.37× combined speedup decomposes into 1.34× from gating (the v0.4.1 effect) and 1.02× from `mx.compile`-path avoidance (the v0.4 effect; small on M1 Max for this recipe — the peak-memory drop from 22 GB to 9.5 GB is the tell that the wrapper bypasses mflux's compiled `_predict`).

## Distilled Klein vs Base Klein: is Base + TeaCache worth it?

FLUX.2 Klein ships two ways. The distilled model draws an image in four steps and ignores guidance. The base model runs the full 50-step schedule with real classifier-free guidance, which is what negative prompts, guidance control, and most fine-tunes need. So a fair question, and the one raised in [mflux issue 113](https://github.com/mflux-community/mflux/issues/113): if you are going to run Base, does TeaCache make it a real alternative to the distilled model?

On speed, no. Base with TeaCache is still far slower than distilled, because the distilled schedule does a small fraction of the work. Base earns its place when you need guidance, negative prompts, or a fine-tune the distilled model cannot run, and there TeaCache takes 15 to 30 percent off the wall clock.

Three conditions, one prompt and seed: distilled at its four-step default, base at 50 steps with guidance 4.0, and base with the wrapper at its default threshold.

### 4B, 768×1024, q4

| Condition | Steps | Wall-clock (s) | vs base | Peak (GB) | SSIM vs distilled | SSIM vs base |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| distilled | 4 | 26 | 22.52× | 12.7 | 1.000 | 0.376 |
| base | 50 | 584 | 1.00× | 13.3 | 0.376 | 1.000 |
| base+TeaCache | 50 (−8 skipped) | 487 | 1.20× | 8.1 | 0.384 | 0.933 |

Images (distilled, base, base + TeaCache):

![distilled](_artifacts/klein_base_vs_distilled/4b_q4/distilled.webp) ![base](_artifacts/klein_base_vs_distilled/4b_q4/base.webp) ![base + TeaCache](_artifacts/klein_base_vs_distilled/4b_q4/base-teacache.webp)

### 9B, 512×512, q4

| Condition | Steps | Wall-clock (s) | vs base | Peak (GB) | SSIM vs distilled | SSIM vs base |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| distilled | 4 | 25 | 20.75× | 22.4 | 1.000 | 0.541 |
| base | 50 | 523 | 1.00× | 21.9 | 0.541 | 1.000 |
| base+TeaCache | 50 (−14 skipped) | 369 | 1.42× | 9.8 | 0.553 | 0.975 |

Images (distilled, base, base + TeaCache):

![distilled](_artifacts/klein_base_vs_distilled/9b_q4_512x512/distilled.webp) ![base](_artifacts/klein_base_vs_distilled/9b_q4_512x512/base.webp) ![base + TeaCache](_artifacts/klein_base_vs_distilled/9b_q4_512x512/base-teacache.webp)

The 9B row renders at 512×512 rather than the page's 768×1024. At q4 the 9B model already peaks near 22 GB on this 32 GB machine, and the larger image pushes it past what the machine runs cleanly. The three 9B conditions share the lower resolution, so they stay comparable to each other, while the 4B row keeps the page's portrait size.

Two things stand out. The distilled and base images are not the same picture (SSIM 0.38 at 4B, 0.54 at 9B): a different model on a different schedule, so moving between them changes the look, not only the speed. Base with TeaCache stays close to plain Base (0.93 and 0.98), which is the point, since it is the same base image for less compute. The peak-memory drop on the wrapped row, 13 GB to 8 GB at 4B and 22 GB to 10 GB at 9B, is the eager wrapper bypassing mflux's compiled predict path, the same effect the klein-base-9b row above shows.

The recommendation is the plain one. Run the distilled model for everyday work, and reach for Base when you need what guidance and fine-tunes give you, with TeaCache to soften the cost.

Measured on an M1 Max 32 GB, mflux 0.18.0, mlx 0.31.2, q4. The 4B row is three reps (cold, then the warm median); the 9B row is two reps, which agree within 2%. Full reports: `_artifacts/klein_base_vs_distilled_4b_q4.json` and `_artifacts/klein_base_vs_distilled_9b_q4_512x512.json`.

### Run it on your Mac

```bash
uv run python scripts/bench_klein_base_vs_distilled.py --size 4b --quantize 4 --reps 3
uv run python scripts/bench_klein_base_vs_distilled.py --size 9b --quantize 4 --reps 3
```

Each condition and repeat runs in its own process and writes its result as soon as it finishes, so an interrupted run picks up where it left off. The tables above are q4, which is what fits a 32 GB Mac. On a 64 GB or larger Mac, run it at `--quantize 8` or `--quantize none` and post your table: the higher-precision numbers are the ones this machine cannot measure.

## Z-Image

### `z-image-base` — 50 steps, guidance=4.0, q8, 640×896

|  | Vanilla mflux | mlx-teacache |
|---|---|---|
| Time (cold) | 510.1 s | 381.9 s |
| Time (warm median) | 510.8 s | 380.1 s |
| Warm speedup | — | **1.34×** |
| Steps skipped | 0 of 48 active | 14 of 48 active (never two in a row) |
| Threshold | — | rel_l1 = 0.12 |
| Peak memory | 18.7 GB | 13.9 GB |
| SSIM vs vanilla | — | **0.957** |
| Image | ![vanilla](_artifacts/comparison/z-image/vanilla.webp) | ![wrapper](_artifacts/comparison/z-image/wrapper.webp) |

This row uses the shared portrait prompt and seed, but at **640×896 q8** rather than the page's 768×1024. Z-Image's weights are 8-bit, and at full 768×1024 the peak crosses the 32 GB unified-memory ceiling on this machine; 640×896 keeps the peak under control (18.7 GB vanilla) while staying the same portrait subject as the FLUX rows.

Z-Image (Tongyi-MAI, Apache-2.0) is a single-stream DiT, not a FLUX model, so it gets its own TeaCache mini-kernel and a separately calibrated gate. Its adaLN modulation is timestep-only, so there is no cheap caption-independent prelude signal to gate on; the calibration taps the first-main-layer residual instead (the caption-independent noise-refiner tap was tried and rejected — its rel-L1 range was too compressed to track the body). At the quality-first threshold of 0.12 the wrapper skips 14 of the 48 active transformer evaluations, each skip avoiding both CFG branches' 30-layer bodies. SSIM is 0.957 — lower than the 0.991 a simpler scene gets at the same threshold, because a detailed face shows the skipped-step softening more readily. The two portraits are perceptually equivalent at output size, with slightly softer microtexture in the wrapped version on close inspection.

Two notes on the numbers. The warm 1.34× here sits next to the 1.31× the README cites for this variant from the 512×512 red-apple bench recipe; the two recipes now agree closely (the README figure used to read 1.17×, from a session whose wrapper time was host-constrained). Both are honest measured wins on their own recipe. The wall-clock speedup is all gating — `mx.compile`-path avoidance is neutral on Z-Image (the 512² three-way bench puts the no-gate wrapper at 0.99× of vanilla). The peak-memory drop from 18.7 GB to 13.9 GB is a separate benefit of the eager wrapper, not of gating: the 512² three-way bench shows the no-gate wrapper landing at the same ~11.5 GB as the gated one (vanilla 17.2 GB), so the drop comes from bypassing mflux's compiled `_predict`, the same effect seen on klein-base-9b.

## Qwen-Image

### `qwen-image` — 50 steps, guidance=4.0, q4, 768×768

|  | Vanilla mflux | mlx-teacache |
|---|---|---|
| Time (cold) | 1153.2 s | 707.0 s |
| Time (warm median) | 1207.2 s | 695.1 s |
| Warm speedup | — | **1.74×** |
| Steps skipped | 0 of 48 active | 25 of 48 active |
| Threshold | — | rel_l1 = 0.30 |
| Peak memory | 30.4 GB | 30.8 GB |
| Image | ![vanilla](_artifacts/comparison/qwen-image/vanilla.webp) | ![wrapper](_artifacts/comparison/qwen-image/wrapper.webp) |

This row uses the shared portrait prompt and seed at **768×768** rather than the page's 768×1024 — same portrait subject as the other rows, only the resolution (and incidentally the aspect) differs. Qwen-Image is a ~20B model; on a 32 GB Mac, stock 4-bit quantization is grainy (a Qwen + q4 limitation, not TeaCache), so these portraits were rendered with a mixed-precision build (8-bit edge transformer blocks + bf16 embeddings), which clears the artifact and peaks ~30.4 GB. mlx-teacache itself stays quantization-agnostic; the [variant page](docs/variants/qwen-image.md) has the construction snippet.

Qwen-Image (Alibaba, Apache-2.0) is a dual-stream MMDiT, FLUX-shaped, so it gets the FLUX-canonical gate signal — the modulated block-0 image input — calibrated at R² 0.849. At the quality-first threshold of 0.30 the wrapper skips 25 of the 48 active steps (~52%) here, each skip avoiding both CFG branches' 60-block bodies, and the two portraits are perceptually equivalent (SSIM 0.987).

Both numbers are v0.9.0's. Under the current gate the same threshold skips more — 33 of 48 on the red-apple recipe the README benchmarks use, at SSIM 0.967 against vanilla — for **2.68×** there, 2.57× of it from gating. See the README's Benchmarks footnote ⁷ for the full measurement history.

## What is excluded and why

- `flux1-schnell` and `flux2-klein-4b` / `flux2-klein-9b` distilled schedules: the residual change between adjacent steps is too large for the gate to engage at any reasonable threshold. They skip zero steps and the wrapper would only add a ~1-2% gating tax. Run them through vanilla mflux instead.
- `flux2-klein-base-4b` at `guidance=1.0`: the base model is not guidance-distilled, so running it without CFG produces low-quality output and the wrapper has no CFG branch to cache. Misuse of the model, not a meaningful comparison row.

## Reproducing these numbers

The `flux1-dev`, `klein-base-4b`, and `z-image-base` rows above:

```bash
uv run python scripts/bench_comparison.py
```

Writes images into `_artifacts/comparison/<variant>/{vanilla,wrapper}.webp` and the full JSON to `_artifacts/comparison_report.json`. Expect about 90 minutes total wall time on an M1 Max: roughly 12 minutes for flux1-dev (six 25-step reps), 30 minutes for klein-base-4b at the CFG schedule (six 50-step CFG reps), and 50 minutes for z-image-base (six 50-step CFG reps at 640×896 q8, the slowest per-step recipe).

The `klein-base-9b` row:

```bash
uv run python scripts/bench_speedup.py --variant klein-base-9b --three-way --reps 3 --report _artifacts/v0.10.0_bench_klein_base_9b.json
```

Subprocess-per-rep (every rep gets a fresh Python interpreter), three-way decomposition (vanilla / wrapped-no-gate / wrapped-gated). Expect roughly 70 minutes on an M1 Max (9 cold generations, each loading the 9B model once).

Override the hardware label in the bench_comparison.py JSON if you are running on a different chip:

```bash
uv run python scripts/bench_comparison.py --machine-label "Apple M3 Max" --ram-gb 64
```

To re-run a single variant after a partial bench_comparison.py run, pass `--only <slug>` (one of `flux1-dev`, `klein-base-4b-cfg`, `z-image`, `qwen-image`).

---

By Denis Ineshin · [ineshin.space](https://ineshin.space)
