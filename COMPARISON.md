# Side-by-side: vanilla mflux vs mlx-teacache

Visual showcase of what the wrapper does on real generations. Every number on this page comes from `scripts/bench_comparison.py` and the JSON report at `_artifacts/comparison_report.json`. The images are committed alongside this file.

Only non-distilled variants are listed here. Distilled schedules (`flux1-schnell`, `flux2-klein-4b`, `flux2-klein-9b`) skip zero steps and gain nothing from the wrapper. See the "When to use mlx-teacache" section in the README for the recommendation.

## Test machine

Apple M1 Max, 32 GB unified memory, macOS Darwin 25.4.0. Models loaded at `quantize=4` in bf16 via mflux 0.17.5. mlx-teacache 0.4.1.

Each variant is run as two subprocesses (one vanilla, one wrapped). Inside each subprocess we record three reps: rep 1 is genuinely cold (the model just loaded), reps 2 and 3 are warm. The "cold" timing on the page is rep 1; the "warm" timing is the median of reps 2 and 3.

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

## What is excluded and why

- `flux1-schnell` and `flux2-klein-4b` / `flux2-klein-9b` distilled schedules: the residual change between adjacent steps is too large for the gate to engage at any reasonable threshold. They skip zero steps and the wrapper would only add a ~1-2% gating tax. Run them through vanilla mflux instead.
- `flux2-klein-base-4b` at `guidance=1.0`: the base model is not guidance-distilled, so running it without CFG produces low-quality output and the wrapper has no CFG branch to cache. Misuse of the model, not a meaningful comparison row.

## Reproducing these numbers

```bash
uv run python scripts/bench_comparison.py
```

The script writes images into `_artifacts/comparison/<variant>/{vanilla,wrapper}.webp` and the full JSON to `_artifacts/comparison_report.json`. Expect about 42 minutes total wall time on an M1 Max: roughly 12 minutes for flux1-dev (six 25-step reps) and 30 minutes for klein-base-4b at the CFG schedule (six 50-step reps with CFG doubling the per-step cost).

Override the hardware label in the JSON if you are running on a different chip:

```bash
uv run python scripts/bench_comparison.py --machine-label "Apple M3 Max" --ram-gb 64
```

To re-run a single variant after a partial run, pass `--only <slug>` (one of `flux1-dev`, `klein-base-4b-cfg`).
