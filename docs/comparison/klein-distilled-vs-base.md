# Distilled Klein vs Base Klein

[← Comparison](../../COMPARISON.md)

FLUX.2 Klein ships two ways. The distilled model draws an image in four steps and ignores guidance. The base model runs the full 50-step schedule with real classifier-free guidance, which is what negative prompts, guidance control, and most fine-tunes need. So a fair question, and the one raised in [mflux issue 113](https://github.com/mflux-community/mflux/issues/113): if you are going to run Base, does TeaCache make it a real alternative to the distilled model?

On speed, no. Base with TeaCache is still far slower than distilled, because the distilled schedule does a small fraction of the work. Base earns its place when you need guidance, negative prompts, or a fine-tune the distilled model cannot run, and there TeaCache takes 15 to 30 percent off the wall clock.

All three conditions use one shared portrait prompt and seed: distilled at its four-step default, base at 50 steps with guidance 4.0, and base with the wrapper at its default threshold. This is the same portrait prompt the comparison page used before it moved to a tennis scene, so the images here don't match the ones there.

### 4B, 768×1024, q4

| Condition | Steps | Wall-clock (s) | vs base | Peak (GB) | SSIM vs distilled | SSIM vs base |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| distilled | 4 | 26 | 22.52× | 12.7 | 1.000 | 0.376 |
| base | 50 | 584 | 1.00× | 13.3 | 0.376 | 1.000 |
| base+TeaCache | 50 (−8 skipped) | 487 | 1.20× | 8.1 | 0.384 | 0.933 |

Images (distilled, base, base + TeaCache):

![distilled](../../_artifacts/klein_base_vs_distilled/4b_q4/distilled.webp) ![base](../../_artifacts/klein_base_vs_distilled/4b_q4/base.webp) ![base + TeaCache](../../_artifacts/klein_base_vs_distilled/4b_q4/base-teacache.webp)

### 9B, 512×512, q4

| Condition | Steps | Wall-clock (s) | vs base | Peak (GB) | SSIM vs distilled | SSIM vs base |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| distilled | 4 | 25 | 20.75× | 22.4 | 1.000 | 0.541 |
| base | 50 | 523 | 1.00× | 21.9 | 0.541 | 1.000 |
| base+TeaCache | 50 (−14 skipped) | 369 | 1.42× | 9.8 | 0.553 | 0.975 |

Images (distilled, base, base + TeaCache):

![distilled](../../_artifacts/klein_base_vs_distilled/9b_q4_512x512/distilled.webp) ![base](../../_artifacts/klein_base_vs_distilled/9b_q4_512x512/base.webp) ![base + TeaCache](../../_artifacts/klein_base_vs_distilled/9b_q4_512x512/base-teacache.webp)

The 9B row renders at 512×512 rather than the page's 768×1024. At q4 the 9B model already peaks near 22 GB on this 32 GB machine, and the larger image pushes it past what the machine runs cleanly. The three 9B conditions share the lower resolution, so they stay comparable to each other, while the 4B row keeps the page's portrait size.

The distilled and base images are not the same picture (SSIM 0.38 at 4B, 0.54 at 9B): a different model on a different schedule, so moving between them changes the look, not only the speed. Base with TeaCache stays close to plain Base (0.93 and 0.98), which is the point, since it is the same base image for less compute. The peak-memory drop on the wrapped row, 13 GB to 8 GB at 4B and 22 GB to 10 GB at 9B, is the eager wrapper bypassing mflux's compiled predict path, the same effect the klein-base-9b row in the comparison page shows.

The recommendation is the plain one. Run the distilled model for everyday work, and reach for Base when you need what guidance and fine-tunes give you, with TeaCache to soften the cost.

Measured on an M1 Max 32 GB, mflux 0.18.0, mlx 0.31.2, q4. The 4B row is three reps (cold, then the warm median); the 9B row is two reps, which agree within 2%. Full reports: `_artifacts/klein_base_vs_distilled_4b_q4.json` and `_artifacts/klein_base_vs_distilled_9b_q4_512x512.json`.

### Run it on your Mac

```bash
uv run python scripts/bench_klein_base_vs_distilled.py --size 4b --quantize 4 --reps 3
uv run python scripts/bench_klein_base_vs_distilled.py --size 9b --quantize 4 --reps 3
```

Each condition and repeat runs in its own process and writes its result as soon as it finishes, so an interrupted run picks up where it left off. The tables above are q4, which is what fits a 32 GB Mac. On a 64 GB or larger Mac, run it at `--quantize 8` or `--quantize none` and post your table: the higher-precision numbers are the ones this machine cannot measure.
