# Klein-base-4b g=1.0 variant dropped from COMPARISON.md

**Date:** 2026-05-18
**Affects:** `docs/superpowers/specs/2026-05-17-comparison-doc-and-cleanup-design.md`, `docs/superpowers/plans/2026-05-17-comparison-doc-and-cleanup.md`, `scripts/bench_comparison.py`

## Decision

The `klein-base-4b-g1` row (25 steps, guidance=1.0) is removed from the comparison doc. Only two non-distilled entries remain:

1. `flux1-dev` — 25 steps, guidance=3.5
2. `flux2-klein-base-4b` (CFG) — 50 steps, guidance=4.0

## Why

The first bench run on 2026-05-18 produced a washed-out, foggy portrait for `klein-base-4b-g1/vanilla.webp`. Web research on `black-forest-labs/FLUX.2-klein-base-4B` confirmed the cause:

- **klein-base-4b is NOT guidance-distilled.** It is the base variant exposed for fine-tuning and inference flexibility. Running it at `guidance=1.0` collapses classifier-free guidance and produces low-quality output.
- **Upstream recommendation:** 25-50 steps, guidance 4.0-7.5. The 50-step / g=4.0 row already in the spec is the canonical setting.
- **Wrapper engagement:** TeaCache requires non-distilled schedules to skip steps. At g=1.0 there is no CFG branch and the gate has nothing meaningful to skip — the wrapper would report 0 skips, identical to a misuse of any distilled variant.

The g=1.0 row therefore measured a configuration nobody should use and produced a comparison image that misrepresented both the model and the wrapper. Dropping it is the right call.

## Artifact cleanup

The bad output was moved to Trash (per the never-rm rule):

```
mv _artifacts/comparison/klein-base-4b-g1 ~/.Trash/klein-base-4b-g1-misconfigured-2026-05-18
```

## Spec / plan deviation

The spec and plan as written reference three entries. They remain as historical records — this note is the correction. Task 4 (write COMPARISON.md) uses the actual `comparison_report.json` produced by the corrected bench, which has two `variants` keys. The COMPARISON.md template in the plan section "Task 4" should be applied with the third row omitted.

## Sources

- Black Forest Labs FLUX.2-klein-base-4B model card (HuggingFace)
- fal.ai FLUX.2 [klein] user guide
- RunDiffusion FLUX.2 Klein guide (Base vs Distilled)
- Runware FLUX.2 [klein] 4B API docs
