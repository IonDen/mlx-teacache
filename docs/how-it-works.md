# How mlx-teacache works

## What it does

An image model starts from random noise and cleans it up over 20 to 50 steps. Each step runs the same large neural network (a transformer), and that is where most of the time goes. Between neighboring steps, the network's output often changes very little.

TeaCache estimates how much the output will change before running the full network. It measures how much a cheap early part of the network changed since the previous step and feeds that number into a small formula fitted in advance for the model. While the estimated changes stay small, TeaCache skips the expensive part of the network and reuses what that part added on the last full step. When the estimates add up past a threshold, `rel_l1_thresh`, it runs the network again and starts counting from zero. A lower threshold skips fewer steps and stays closer to the plain image. Most models ship their own default threshold, chosen by comparing images at several values; FLUX.1 dev and schnell and the distilled Klein models use the package default, 0.20.

mlx-teacache is the first MLX port of TeaCache. [How it works](#how-it-works) has the details and a link to the original paper.

## How it works

TeaCache observes that in diffusion denoising, consecutive transformer outputs change very little between most pairs of adjacent steps. The expensive transformer body (all the joint and single attention blocks) produces a residual that's added to the input, and that residual stays roughly stable for stretches of the denoising trajectory.

TeaCache trains a tiny polynomial that predicts how much the output will change given how much the input has changed, measured as the relative L1 distance between each step's modulated block-0 input and the immediately previous step's — matching how the polynomials are calibrated, and the comparison anchor advances on every step, including skipped ones. TeaCache accumulates the predicted change across consecutive skips; once that running total would cross `rel_l1_thresh`, TeaCache recomputes the transformer body instead of reusing the cached residual, and resets the running total. For the polynomials calibrated in this repo (FLUX.2 Klein base, Z-Image, Qwen-Image, Qwen-Image-2512), the input change is first held to the range the calibration actually saw, so a change smaller than any calibrated step is priced like the smallest one rather than as almost no change, and a change larger than any calibrated step always recomputes. Only the cheap prelude (embeddings) and tail (norm + projection) still run on a skipped step.

mlx-teacache implements this for mflux on Apple Silicon. For FLUX.1 we replace `flux.transformer` with a per-instance proxy; for FLUX.2 we replace `flux._predict` with an instance-level closure, which keeps gating live on chips where mflux would otherwise wrap `_predict` in `mx.compile`. The FLUX.1 polynomial coefficients are vendored from upstream. The FLUX.2 coefficients are derived in-repo by `scripts/calibrate_flux2.py` (distilled klein-4b and klein-9b, base-4b each have their own calibrated tuple; klein-base-9b cross-imports base-4b's tuple because the architectures share the calibration recipe). See [`docs/calibration.md`](calibration.md) for procedure and provenance, and the per-variant docs under `docs/variants/` for each tuple's R² and origin date. The original method is described in the TeaCache paper at https://liewfeng.github.io/TeaCache/.

For the full contributor's map — the variant registry, the mflux-free import contract, the per-family patch strategies, the gate internals, and the teardown contract — see [`docs/architecture.md`](architecture.md).
