# FLUX.2 [klein] base 9B

[← Comparison](../../COMPARISON.md)

50 steps, guidance 4.0, q4, 768×1024, seed 42; text encoders freed once the prompt is encoded.

## Prompt

> A beautiful young woman plays tennis on an outdoor hard court at sunset. She is caught just after a forehand, racket following through across her body, ponytail swinging, weight on her front foot. Her face shows focused, joyful determination: flushed cheeks, bright eyes, a light sheen of sweat on her forehead and temples. She wears a fitted white tennis dress with navy trim, a white visor, a terry wristband, small gold stud earrings, and white tennis shoes with navy laces. The green court's painted white lines lead back to a chain-link fence, the net, and silhouetted trees against an orange-pink sky. Low, warm sunlight rakes across the court, casting long soft shadows and a golden rim light on her hair. A yellow tennis ball hangs in the air just off the racket strings. The mood is cozy, warm and nostalgic. Photorealistic, full-frame camera, 85mm lens, shallow depth of field, natural skin texture, sharp focus on her face.

<!-- COMPARISON:klein-base-9b:sheets START -->
**A: TeaCache off**, every step in order

![Steps, TeaCache off](../../_artifacts/comparison/klein-base-9b/steps-a.jpg)

**B: TeaCache on**, skipped steps framed in orange

![Steps, TeaCache on](../../_artifacts/comparison/klein-base-9b/steps-b.jpg)
<!-- COMPARISON:klein-base-9b:sheets END -->

<!-- COMPARISON:klein-base-9b:details START -->
|  | A: TeaCache off | B: TeaCache on |
|---|---|---|
| Model load (weights evaluated) | 8.8 s | 9.4 s |
| Prompt encoding | 2.2 s | 2.1 s |
| Generation, wall clock | 1025.6 s | 810.5 s |
| Median computed step | 20.3 s | 20.5 s |
| Median skipped step | — | 0.0 s |
| Preview decoding, total | 8.0 s | 7.5 s |
| Steps computed / skipped | 50 / 0 | 39 / 11 |
| Skip pattern (S = skipped) | — | `CCCSCSCSCCCCSCCSCCSCSCCCCSCCSCCSCCSCCCCCCCCCCCCCCC` |
| Threshold (rel_l1) | — | 0.17 |
| MLX peak: load / encode / generation | 4.7 GiB / 5.7 GiB / 8.5 GiB | 4.7 GiB / 5.7 GiB / 8.7 GiB |
| MLX active + cache, peak | 9.7 GiB | 9.6 GiB |
| Process footprint (macOS), peak | 11.1 GiB | 10.8 GiB |

Speedup on this run: 1.27× wall clock, 1.27× with preview decoding left out, 1.28× per step after the first. SSIM of B against A: 0.860, measured on the lossless outputs.

Recipe: 50 steps, guidance 4.0, q4, 768×1024, seed 42, text encoder freed once the prompt is encoded. Checkpoint `black-forest-labs/FLUX.2-klein-base-9B`; preview decoder `taef2`. mlx-teacache 0.11.1 with this branch's changes (commit `bcaac6f`), mflux 0.20.0, MLX 0.32.2, mlx-taef 0.8.3. Multi-run measurement of this model: [bench report](../../_artifacts/v0.10.0_bench_klein_base_9b.json).
<!-- COMPARISON:klein-base-9b:details END -->

### Notes

At its matching 512² recipe, footnote ⁵ splits this variant's 1.37× multi-run speedup into 1.34× from skipped steps and a small 1.02× from avoiding a compiled step. Plain mflux runs this model's prediction step compiled; the wrapper runs it eagerly instead. That same footnote also reports a peak-memory drop from about 22 GB to 9.5 GB at that recipe; it doesn't show up on this run, where the weights are evaluated before generation and the text encoder is freed once the prompt is encoded, so A and B peak within about a gigabyte of each other in the table above (8.5 GiB vs 8.7 GiB).

On this run the gate skips 11 of 50 steps. Same player, same pose, same light between A and B; B redraws the court lines behind her.

### Reproduce

```bash
uv run python scripts/bench_comparison.py --only klein-base-9b --max-workers 1
uv run python scripts/bench_comparison.py --only klein-base-9b --max-workers 1
uv run python scripts/bench_comparison.py --only klein-base-9b --finalize
```
