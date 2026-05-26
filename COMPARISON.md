# Side-by-side: vanilla mflux vs mlx-teacache

Visual showcase of what the wrapper does on real generations. Two harnesses contribute to this page:

- The `flux1-dev` and `flux2-klein-base-4b` rows are from `scripts/bench_comparison.py` (subprocess-per-condition, three reps per subprocess; cold = rep 1, warm = median of reps 2 and 3). Full report: `_artifacts/comparison_report.json`. Committed images alongside this file under `_artifacts/comparison/<variant>/`.
- The `flux2-klein-base-9b` row is from `scripts/bench_speedup.py --three-way --reps 3` (subprocess-per-rep — every rep gets a fresh interpreter, fully cold). Full report: `_artifacts/v0.6.0_bench_klein_base_9b.json`. Images under `tests/_artifacts/bench_images/klein-base-9b/`.

The two harnesses use different prompts and resolutions (the bench_comparison row is the 768×1024 portrait listed below; bench_speedup is the 512×512 red-apple recipe described in the README's Benchmarks section). Cross-reading the numbers across rows therefore requires care.

Only non-distilled variants are listed here. Distilled schedules (`flux1-schnell`, `flux2-klein-4b`, `flux2-klein-9b`) skip zero steps and gain nothing from the wrapper. See the "When to use mlx-teacache" section in the README for the recommendation.

## Test machine

Apple M1 Max, 32 GB unified memory, macOS Darwin 25.4.0. Models loaded at `quantize=4` in bf16 via mflux 0.17.5. mlx-teacache 0.6.1.

Shared inputs across every cell:

- **Prompt:** *"Portrait of a young woman with auburn hair and green eyes, soft golden-hour window light, photorealistic, shallow depth of field, 50mm prime lens, subtle freckles, neutral background, cinematic color grading."*
- **Seed:** 42
- **Resolution:** 768 × 1024 (portrait)
- **Image format on disk:** webp, quality 88, method 6

## FLUX.1 family

### `flux1-dev` — 25 steps, guidance=3.5

|  | Vanilla mflux | mlx-teacache |
|---|---|---|
| Time (cold) | 238.0 s | 200.1 s |
| Time (warm median) | 233.2 s | 198.3 s |
| Warm speedup | — | **1.18×** |
| Steps skipped | 0 of 23 active | 4 of 23 active |
| Threshold | — | rel_l1 = 0.20 |
| Image | ![vanilla](_artifacts/comparison/flux1-dev/vanilla.webp) | ![wrapper](_artifacts/comparison/flux1-dev/wrapper.webp) |

FLUX.1 dev runs the schedule that mlx-teacache was originally calibrated for. The wrapper drops four transformer evaluations out of the 23 active steps (the first two are always computed; the algorithm gates from step 2 onward). Output is visually indistinguishable from vanilla on this prompt.

## FLUX.2 family

### `flux2-klein-base-4b` — 50 steps, guidance=4.0

|  | Vanilla mflux | mlx-teacache |
|---|---|---|
| Time (cold) | 576.2 s | 486.4 s |
| Time (warm median) | 601.6 s | 503.3 s |
| Warm speedup | — | **1.20×** |
| Steps skipped | 0 of 48 active | 8 of 48 active |
| Threshold | — | rel_l1 = 0.17 |
| Image | ![vanilla](_artifacts/comparison/klein-base-4b-cfg/vanilla.webp) | ![wrapper](_artifacts/comparison/klein-base-4b-cfg/wrapper.webp) |

This is the canonical FLUX.2 Klein base setting from Black Forest Labs: 50 steps with real classifier-free guidance at 4.0. The wrapper skips eight of the 48 active transformer evaluations under both the conditional and unconditional branches (per-branch caching, the v0.4.1 feature). The two portraits are perceptually equivalent at output size; close inspection shows slightly softer microtexture in the wrapped version, which is the expected trade for the skipped steps.

### `flux2-klein-base-9b` — 50 steps, guidance=4.0

|  | Vanilla mflux | mlx-teacache |
|---|---|---|
| Time (median of 3 reps)¹ | 517.6 s | 380.6 s |
| Speedup | — | **1.36×** |
| Steps skipped | 0 of 48 active | 13 of 48 active |
| Threshold | — | rel_l1 = 0.17 |
| Peak memory | ~22 GB | ~10 GB |
| SSIM vs vanilla | — | **0.986** |
| Image | ![vanilla](_artifacts/validation_klein_base_9b_images/vanilla.webp) | ![wrapper](_artifacts/validation_klein_base_9b_images/wrapper.webp) |

¹ Subprocess-per-rep bench on M1 Max 32 GB, bf16, q4 — every (variant, condition, rep) runs in a fresh Python interpreter so each timing starts from a cold MLX allocator. Full report: `_artifacts/v0.6.0_bench_klein_base_9b.json`. v0.5.0 advertised 2.68× on this same recipe; that number was inflated by same-process MLX state leakage in the v0.5.x bench harness — vanilla ran cold while the wrapper inherited warm allocator state. v0.6.0's subprocess isolation exposes the honest 1.36×.

`flux2-klein-base-9b` is the non-distilled FLUX.2 Klein 9B variant (FLUX Non-Commercial license — see [README License obligations](README.md#license-obligations) and accept on the [Hugging Face model page](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) before downloading). Reuses base-4b's polynomial coefficients verbatim — same FLUX.2 Klein architecture family, same calibration recipe. The 1.36× combined speedup decomposes into 1.34× from gating (the v0.4.1 effect) and 1.02× from `mx.compile`-path avoidance (the v0.4 effect; small on M1 Max for this recipe — the peak-memory drop from 22 GB to 10 GB is the tell that the wrapper bypasses mflux's compiled `_predict`).

## What is excluded and why

- `flux1-schnell` and `flux2-klein-4b` / `flux2-klein-9b` distilled schedules: the residual change between adjacent steps is too large for the gate to engage at any reasonable threshold. They skip zero steps and the wrapper would only add a ~1-2% gating tax. Run them through vanilla mflux instead.
- `flux2-klein-base-4b` at `guidance=1.0`: the base model is not guidance-distilled, so running it without CFG produces low-quality output and the wrapper has no CFG branch to cache. Misuse of the model, not a meaningful comparison row.

## Reproducing these numbers

The `flux1-dev` and `klein-base-4b` rows above:

```bash
uv run python scripts/bench_comparison.py
```

Writes images into `_artifacts/comparison/<variant>/{vanilla,wrapper}.webp` and the full JSON to `_artifacts/comparison_report.json`. Expect about 42 minutes total wall time on an M1 Max: roughly 12 minutes for flux1-dev (six 25-step reps) and 30 minutes for klein-base-4b at the CFG schedule (six 50-step reps with CFG doubling the per-step cost).

The `klein-base-9b` row:

```bash
uv run python scripts/bench_speedup.py --variant klein-base-9b --three-way --reps 3 --report _artifacts/v0.6.0_bench_klein_base_9b.json
```

Subprocess-per-rep (every rep gets a fresh Python interpreter), three-way decomposition (vanilla / wrapped-no-gate / wrapped-gated). Expect roughly 70 minutes on an M1 Max (9 cold generations, each loading the 9B model once).

Override the hardware label in the bench_comparison.py JSON if you are running on a different chip:

```bash
uv run python scripts/bench_comparison.py --machine-label "Apple M3 Max" --ram-gb 64
```

To re-run a single variant after a partial bench_comparison.py run, pass `--only <slug>` (one of `flux1-dev`, `klein-base-4b-cfg`).
