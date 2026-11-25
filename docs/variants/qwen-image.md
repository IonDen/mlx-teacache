# qwen-image

Qwen-Image base (Alibaba) — a ~20B dual-stream MMDiT, non-distilled, run at the official 50-step CFG recipe. It's the first variant on the FLUX.1 proxy-transformer pattern that also runs true two-pass CFG, and TeaCache skips about two-thirds of its denoising steps at the shipped default as of v0.10.0 (about half under 0.9.x), holding SSIM 0.967 against vanilla.

## Construct via mflux

```python
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.common.config.model_config import ModelConfig

flux = QwenImage(quantize=4, model_config=ModelConfig.qwen_image())
```

On mflux 0.17.5 and 0.18 the `qwen-image` alias is `Qwen/Qwen-Image`, the checkpoint every number on this page was measured on, and the detector matches its aliases `"qwen-image"` and `"qwen"`. On mflux 0.19 and later the same alias loads `Qwen/Qwen-Image-2512` (aliases `qwen-image`, `qwen`, `qwen-image-2512`, `qwen-2512`), a newer checkpoint. Since 0.14.0 that checkpoint has its own variant, [`qwen-image-2512`](qwen-image-2512.md), with its own calibration and default threshold, so `apply_teacache` no longer warns about it. On those mflux releases the original checkpoint keeps this variant only when you build it in Python with its repo id (`Qwen/Qwen-Image`, in any letter case) as the `model_config`:

```python
flux = QwenImage(quantize=4, model_config=ModelConfig.from_name("Qwen/Qwen-Image"))
```

Every other route resolves against the 2512 config. `mflux-generate-qwen --model Qwen/Qwen-Image` is accepted and loads the original weights on the 2512 config. On that command, `--base-model` accepts `qwen-image`, `qwen`, `qwen-image-2512`, `qwen-2512` and `Qwen/Qwen-Image-2512`, and refuses `Qwen/Qwen-Image` and the names of other model families. The command has no `--model-path` flag: a local directory or a mirror is passed as `--model PATH` on the command line or as `model_path` in Python. The command never runs TeaCache, so what follows applies to a Python `apply_teacache` call on a model loaded that way. A `model_path` resolves to the 2512 config with the default `model_config` (per mflux 0.23.0's config resolution), and passing `model_config=ModelConfig.from_name("Qwen/Qwen-Image")` keeps the original variant. That config records nothing about the original checkpoint, so such weights are detected as Qwen-Image-2512 and get that variant's coefficients and 0.25 default, with no warning. To keep this variant's calibration for them, pass both its tuple and its threshold: `apply_teacache(flux, coefficients=COEFFICIENTS, rel_l1_thresh=0.30)`, with `COEFFICIENTS` imported from `mlx_teacache.variants.qwen_image.config`. Both are needed, because with your own coefficients the default threshold becomes 0.20 and the calibrated-range clamp is off.

The uncalibrated-checkpoint warning covers less than its name suggests. mflux records the base model in every config it resolves, so a finetune or a mirror loaded through mflux counts as the checkpoint itself and gets no warning. Only a config that names neither the matched variant's checkpoint nor that checkpoint as its base model raises `TeaCacheUncalibratedCheckpointWarning` once, and the model is still patched. Calibrate such a model with `scripts/calibrate_qwen.py --model <repo>` and pass `coefficients=`, which also silences the warning.

Qwen-Image-Edit (aliases `"qwen-image-edit"` / `"qwen-edit"` / `"qwen-edit-plus"` / `"qwen-edit-2509"`) is a separate pipeline, not a registered variant; it falls through to `IncompatibleModelError`. So does Qwen-Image-2.1 (Qwen Research License, non-commercial; aliases `"qwen-image-2.1"` / `"qwen-2.1"` / `"qwen-image-21"`, mflux 0.20 and later), a different architecture: the detector tests alias membership, and none of those equals `"qwen-image"` or `"qwen"`.

## Recipe + defaults

- Default recipe: 50 steps, `guidance=4.0` (CFG), quantize=4, 768×768
- Default `rel_l1_thresh`: **0.30** (per-variant default, set from the threshold sweep)
- skip-window defaults: `skip_first_n_steps=1`, `skip_last_n_steps=1`

On the comparison page's single run at 672×896, q4, the wrapper took the generation from 693.6 s to 336.7 s (2.06× on this run, preview decoding included), skipping 26 of 50 steps at SSIM 0.92 against vanilla. The record is `_artifacts/comparison/report.json`, from mflux 0.20.0 and MLX 0.32.2; it names a commit for each condition but does not say whether each working tree was clean. See [the comparison page](https://github.com/IonDen/mlx-teacache/blob/main/docs/comparison/qwen-image.md) for the images and the step-by-step preview sheets.

## Image quality on consumer memory

Every number on this page except the comparison-page run above is a uniform 4-bit build on mflux 0.18.0. From mflux 0.19, mflux's 4-bit Qwen build keeps every block's `img_mod_linear` modulation layer at 8 bits (a source change, mflux commit [acad1f9](https://github.com/mflux-community/mflux/commit/acad1f90bf38ae17332013da5d94a672478acf34), [mflux PR #517](https://github.com/mflux-community/mflux/pull/517); not in the 0.19.0 release notes), which a comment in mflux's source puts at about 1.8 GB more memory. TeaCache leaves that as mflux ships it, so a q4 run on a newer mflux is not the recipe measured here, and the gate's input now comes from 8-bit modulation layers that the shipped coefficients were not fitted on.

Stock uniform 4-bit quantization over-quantizes Qwen-Image's quantization-sensitive layers and produces a grainy, low-detail skin texture on a 32 GB Mac — a Qwen + q4 limitation, independent of TeaCache (the wrapper faithfully reproduces whatever the base model generates). The earlier comparison page used a mixed-precision build that keeps the first/last transformer blocks at 8-bit and the embeddings + final projection at bf16, which clears the artifact at a higher peak (above the recommended working set of a 32 GB Mac, from a different run than the stock q4 ones; [`_artifacts/comparison_report.json`](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/comparison_report.json)); the current page uses stock 4-bit instead (17.6 GiB peak), so its portrait keeps a light speckle where the mixed-precision build would not. mlx-teacache stays quantization-agnostic; this is a model-construction choice. To reproduce the mixed-precision build's quality, install the predicate before constructing the model:

```python
import mlx.nn as nn
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

_PROTECT = set(range(6)) | set(range(54, 60))           # first 6 + last 6 of 60 blocks -> q8
_BF16 = ("img_in", "txt_in", "time_text_embed", "proj_out", "norm_out")  # -> bf16

def _mixed(path, module):
    if not hasattr(module, "to_quantized"):
        return False
    if any(path == p or path.startswith(p + ".") for p in _BF16):
        return False                                    # keep bf16
    if path.startswith("transformer_blocks."):
        idx = int(path.split(".")[1])
        if idx in _PROTECT:
            return {"group_size": 64, "bits": 8}        # q8
    return True                                         # default q4

QwenWeightDefinition.quantization_predicate = staticmethod(_mixed)
flux = QwenImage(quantize=4, model_config=ModelConfig.qwen_image())
```

The gate signal (Signal A) is unaffected enough by this that the shipped coefficients — calibrated on stock q4 — are expected to transfer to the mixed-precision build. No record of a sweep on that build was committed, so this rests on the stock-q4 calibration alone (see [Threshold sweep](#threshold-sweep)).

## Threshold sweep

`scripts/sweep_threshold_qwen.py` sweeps `rel_l1_thresh` at the 768×768/50-step recipe (red-apple prompt, seed 42, guidance 4.0, stock q4) and records skip count, longest skip streak and SSIM against a vanilla reference per threshold, one subprocess per threshold; some run freed the text encoders once the prompt was encoded (the committed record's `memory_saver` flag says so; which runs is not in it). Re-run under the current gate, recorded in `_artifacts/sweeps/qwen-image_plain-q4_2026-09-06.json`:

| `rel_l1_thresh` | Skipped (of 48 active) | Longest streak | SSIM vs vanilla | Single-rep speedup |
|---|---|---|---|---|
| 0.15 | 24 | 2 | 0.980 | 1.71× |
| 0.20 | 26 | 2 | 0.980 | 1.93× |
| 0.25 | 30 | 3 | 0.976 | 2.55× |
| 0.30 | 33 | 4 | 0.967 | 3.01× |

The speed column is one run per threshold against one plain run of 920 s. The three-rep bench below has a plain median of 643 s, so the 0.25 and 0.30 rows sit above what their skipped steps allow (50 steps over 20 computed gives 2.50×, over 17 gives 2.94×). Read the column as single timings; the bench below is the headline.

What the record leaves out: its date, 2026-09-06, is the file time on the machine when the record was made, and the file name carries it (git keeps no file times). It names no MLX or mflux version, hardware, commit or uncommitted-changes flag; `uv.lock` at 3b86c8b pins mflux 0.18.0 and MLX 0.31.2. mflux 0.22.0 loads the bias of Qwen-Image's last normalization layer again, and its release notes say the images change visibly, so these scores hold for mflux 0.18.0. The plain-mflux reference image comes from a first attempt that was stopped when the host ran out of memory (see the 0.10.1 notes in the CHANGELOG), and the gated runs followed hours later. The script has been in the repository since June 2026. The version that frees the text encoders was committed during the run and reached main in 3b86c8b, so the reference and the 0.15 run predate it, and the record's `memory_saver: true` flag says only that some run freed them. The record does not show which runs freed the text encoders. The images behind the SSIM scores are not published. [`docs/calibration.md`](../calibration.md#threshold-sweep-records) lists the gaps for all four sweep records.

SSIM still degrades gracefully, with no cliff, and every point clears the 0.95 floor the parity gate enforces. 0.30 stays the default: it is the fastest point on the curve that holds visual equivalence. The operating point of the 0.9.x releases is one setting away at `rel_l1_thresh=0.20`, which now gives 26 skips at 0.980. The default was first set by a June 2026 sweep under the 0.9.x gate, which gave equal or higher SSIM. No record of it was committed, so its figures are not given here, and the table above replaces it.

## Benchmark

The 2.68× below comes from [`_artifacts/v0.11.0_bench_qwen_image.json`](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/v0.11.0_bench_qwen_image.json). The record has no date field or MLX version; 2026-09-06 comes from its chunk timestamps. It names commit 1536f28 and `git_dirty: true`. The harness writes both when it writes the report, after the last chunk, and the report does not list the uncommitted files. Commit 1536f28 was made about 97 seconds after the first chunk began. `scripts/` and `src/` are byte-identical at 46612df (the last commit before the first chunk), at 1536f28 and at 97f4385 (the commit that carries the report). All three are public in [PR #35](https://github.com/IonDen/mlx-teacache/pull/35), and the report reached main as 3b86c8b on 2026-09-06. An uncommitted edit during the run cannot be ruled out.

The record has no MLX version. The lock file at those commits resolves MLX 0.31.2. It stores neither the checkpoint nor the threshold. The bench built the model from mflux's `qwen-image` alias, which on mflux 0.18.0 loads the original Qwen/Qwen-Image; the scripts name the repo explicitly only since 2026-10-01 (commit 50b3023). Measured on mflux 0.18.0: from mflux 0.19.0 the alias loads Qwen-Image-2512, and the 4-bit build keeps the image modulation linear (`img_mod_linear`) at 8 bits (a source change, mflux commit [acad1f9](https://github.com/mflux-community/mflux/commit/acad1f90bf38ae17332013da5d94a672478acf34); not in the 0.19.0 release notes), so the figure holds for mflux 0.18.0. mflux 0.22.0 also restored a bias in the model's last layer, which changes Qwen-Image's images visibly, so the figure holds for mflux 0.18.0. The README's [Benchmarks footnote ¹](../../README.md#benchmarks) is the short form.

Run: `scripts/bench_speedup.py --variant qwen --three-way --reps 3`, red-apple prompt, seed 42, 50 steps, guidance 4.0, 768×768, stock q4, one fresh subprocess per (condition, rep) so every timing starts cold, chunks interleaved rep-outer, text encoders freed after prompt encoding, medians of three (M1 Max 32 GB, macOS 26, mflux 0.18.0, 2026-09-06; `_artifacts/v0.11.0_bench_qwen_image.json`):

| Condition | Rep seconds | Median | Skipped (of 48 active) |
|---|---|---|---|
| vanilla | 610.9 / 642.6 / 715.9 | 642.6 s | — |
| wrapper, no gate (`rel_l1_thresh=0.0`) | 615.6 / 611.5 / 692.5 | 615.6 s | 0 |
| wrapper, default 0.30 | 229.3 / 240.7 / 239.4 | 239.4 s | 33 in every rep, longest streak 4 |

That run gave 2.68× combined against plain mflux, of which 2.57× is gating. The no-gate wrapper's 1.04× over plain mflux is inside the spread between the vanilla reps. In every rep 33 of the 48 active steps skipped, and the gated skip pattern is identical in the 0.10.0 and 0.11.0 reports. The peak, 24.08–24.13 GiB in every condition, is the active peak read after model load. The report's load phase reads about zero because mflux materialises weights on first use, so it is also the whole-run peak. The cache pool read 1.06 GiB after decode. Adding that reading gives roughly 25.1–25.2 GiB, an estimate computed rather than observed. It is not an upper bound, because MLX's cache pool can overshoot its limit, and the cache at the peak was not measured. The 24.96 GiB used for comparison is the default that the comparison recipes hard-code for a 32 GB M1 Max (`scripts/_comparison_recipes.py`); it is not a recorded device value.

The 0.10.0 report ([`_artifacts/v0.10.0_bench_qwen_image.json`](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/v0.10.0_bench_qwen_image.json)) has no date, commit field, dirty flag or MLX version; its install-time version string names commit 90fca33. The README at the v0.10.0 release dates the run "Measured 2026-09-01". The report was committed on the pull-request branch as 15250be on 2026-09-01 and merged to main as fe06b48 on 2026-09-02. The benchmark's Qwen-Image support entered the same PR #33 branch as 18c8feb at 17:27:40 UTC on 2026-09-01, 24 seconds before 15250be. The report's own timings add up to about 95 minutes, and the run had to end before the report was committed, so it began before 18c8feb. The code that ran was not yet committed when the run started, and the history does not show which code ran. Its peak was above the 24.96 GiB default, and its vanilla time (850.3 s) gave 3.02×, which is above the 2.94× that 33 skipped steps of 50 allow (50/17). The record does not say why. It does not explain the 1.10× no-gate gap either: 79.6 s against a 2.4 s spread in its vanilla reps. The later median ratio, 2.68×, is below 3.02×. The two runs differ in six things: mflux's `MemorySaver` freeing the text encoders, a 1 GiB bound on MLX's cache pool, a peak reset, a different chunk order, a new abort watchdog, and different mlx-teacache library code. No run isolates any one of them as the cause of the lower ratio or the lower peak. Power state was not recorded in either run.

Both records come from an Apple M1 Max with 32 GB, Darwin 25.6.0 (macOS 26; the reports store the kernel version), mflux 0.18.0, uniform 4-bit, bf16. Neither records the MLX version.

## CFG (guidance > 1.0)

Qwen-Image always runs CFG: `QwenImage.generate_image` calls the transformer twice per step (positive then negative caption) and combines the two predictions **outside** the transformer. So the proxy forward fires once per branch, and a small state machine threads one shared gate decision and two cached residuals across the pair. The decision is computed on the positive call's gate signal and reused on the negative call; each branch keeps its own cached residual. The gate signal is branch-independent (it depends on the latents and timestep, not the caption), so sharing one decision across both branches is exact, not an approximation. At threshold 0 the wrapper never skips, and its latents match plain mflux's with cosine similarity of at least 0.99 (`tests/test_parity_qwen.py`). The patched forward runs a different sequence of operations from mflux's, so float rounding can differ, and nothing here has been measured bit for bit. The parity gate compares the latents by cosine similarity and requires 0 skips, and the image-level checks in this repository compare decoded images by SSIM rather than by bytes.

## Coefficient provenance

Qwen-Image is FLUX-shaped: each block applies a 6×dim adaLN modulation, so the FLUX-canonical TeaCache signal is available — the modulated block-0 image input. The calibration captured two candidate signals and fit a degree-4 origin-constrained polynomial for each:

- **Signal A** — modulated block-0 image input rel-L1 (the FLUX-canonical signal; caption-independent). **Selected**: R² = 0.849, held-out 0.845.
- **Signal B** — first-block image-stream residual rel-L1 (caption-dependent). R² = 0.881, held-out 0.890.

Signal B's R² is marginally higher, but Signal A is the better choice on two counts: it's caption-independent (the shared CFG decision is exact rather than approximate), and it's cheaper on a skip step (Signal A comes from the prelude; Signal B would require running block 0). The R² (~0.85) is lower than a 512×512/20-step fit would give because the heavier 768×768/50-step recipe samples a finer, noisier per-step rel-L1 distribution — still well above Z-Image's 0.40 and the FLUX.2 family's 0.11–0.47, and the sweep confirms it skips strongly at high SSIM.

- `scripts/calibrate_qwen.py --fit-mode origin` (chunked: one worker subprocess per prompt, resumable)
- 10 prompts (7 fit / 3 held-out) × 50 steps × seed=42 on M1 Max 32 GB, q4, 768×768, guidance=4.0 (CFG)
- A first-step self-check asserts the capturing re-walk's per-branch noise matches the unwrapped `QwenTransformer.__call__` at cosine ≥ 0.999 (faithful-port guard)
- Stored verbatim in `src/mlx_teacache/variants/qwen_image/config.py::COEFFICIENTS`

Full report: `scripts/_calibration_qwen.json`.

## Cache contract

The integration replaces `flux.transformer` with a proxy whose forward re-walks `QwenTransformer.__call__` in four parts: a prelude (`img_in` + the timestep-only text embedding), the gate signal (the modulated block-0 image input), the body (the 60 dual-stream blocks over the image stream), and the tail (`norm_out` + `proj_out`). On a compute step it caches `cached_residual = body_out − img_in(latents)` per CFG branch; on a skip it reconstructs `body_out = img_in(latents) + cached_residual` from the current step's freshly computed prelude, then runs the tail. A skip runs the prelude, the gate signal, and the tail — but none of the 60 blocks, and not the rotary-embedding computation (which only the blocks need).

## License

[`Apache-2.0`](https://huggingface.co/Qwen/Qwen-Image). Permissive, not gated — no acceptance flow required.

## Quirks

- **Default threshold is 0.30, not the package fallback 0.20.** The default comes from the variant's `DEFAULT_THRESH`.
- **768×768, the official Qwen recipe.** At stock q4 on the 768×768 bench recipe the 20B model peaked above 24.96 GiB, the working-set default the comparison recipes use for a 32 GB M1 Max ([`_artifacts/v0.10.0_bench_qwen_image.json`](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/v0.10.0_bench_qwen_image.json)). The mixed-precision showcase build peaked higher still, on a different mflux version and prompt ([`_artifacts/comparison_report.json`](https://github.com/IonDen/mlx-teacache/blob/v0.14.1/_artifacts/comparison_report.json)). The two peaks come from different runs, and neither record names an MLX version. They complete only while host memory pressure stays low: the wired cap bounds only the non-pageable allocation, and the pageable excess pages rather than failing, which under sustained pressure can stall the run or panic the machine instead of raising a clean error. Treat 32 GB as the floor, confirm swap is near empty before a run, and expect the showcase build to be the tighter of the two. As of 0.10.1 the bench and sweep workers bound MLX's cache pool (1 GiB for this variant) and abort themselves the moment active plus cached memory exceeds physical memory minus 4 GiB, writing an `*.aborted.json` instead of paging; on a 32 GB Mac that ceiling is 28 GiB, so the stock-q4 bench recipe fits and the mixed-precision showcase build does not. Since 0.11.0 those workers also free the text encoders once the prompt is encoded (mflux's `MemorySaver`). The stock-q4 run that did so had an active peak of 24.08–24.13 GiB after model load (the load phase reads about zero, so this is also the whole-run peak), under 24.96 GiB. Adding the 1.06 GiB cache reading gives roughly 25.1–25.2 GiB, an estimate computed rather than observed and not an upper bound, since MLX's cache pool can overshoot its limit. The run differs from the 0.10.0 run in six things (the encoders freed, a 1 GiB bound on MLX's cache pool, a peak reset, a different chunk order, the abort watchdog and different mlx-teacache library code), so the lower peak is not credited to any one of them. 1024×1024 needs tiled VAE decode to fit; 768×768 is the comfortable default.
- **Proxy-transformer pattern, not the `_predict` closure.** Qwen-Image has no `_predict` factory and no `mx.compile`, so the variant proxies `flux.transformer` (like FLUX.1) rather than replacing `_predict` (like FLUX.2 / Z-Image). As with FLUX.1, calling `flux.parameters()` at the parent level can miss transformer parameters while the wrapper is active — use `flux.transformer.parameters()` or `handle.restore()` first.
- **Calibrated range (0.12.0).** The gate evaluates the signal-A polynomial only inside the range of deltas it was fitted on, 0.0468 to 0.5878 (`CALIBRATED_RANGE`). A smaller delta is priced at 0.067, the fit's value at the bottom of that range, and a larger one always computes. At the 0.30 default this allows at most four skips in a row, the longest run the bench shows. The 768×768 bench recipe's decisions are unchanged; see `docs/calibration.md`.
- **mflux 0.22 (0.13.2).** mflux 0.22 loads the output norm's bias when the checkpoint has one; earlier releases left it out. The forward calls mflux's own `norm_out`, so TeaCache gets the bias too. Images loaded from the original Hugging Face weights can therefore differ from 0.21 at the same seed, with or without TeaCache; a model saved by an earlier mflux has no bias and renders as before. This release did not measure the difference, and the skip counts on 0.22 were not re-measured. The gate reads block 0's modulated input, before that norm, but the trajectory can still shift. 0.22 also adds a `zero_cond_t` path for Qwen-Image-Edit-2511, which modulates reference-image tokens with the timestep-0 embedding. Text-to-image never turns it on; `apply_teacache` refuses a model that has it, and `match_variant` returns `None` for its config.
- **Memory saver (0.13.3).** When mflux's memory saver drops the transformer after the denoising loop (`--low-ram` or `--pid-decode`, for a single seed), TeaCache no longer keeps it alive, so the decode runs without the transformer's weights. Before 0.13.3 the handle held the original transformer for `restore()`. This was checked with weight-free tests that run gated forward steps through a small real mflux Qwen transformer, and since 0.14.0 once on real Qwen-Image-2512 weights, which go through the same code (see the [Qwen-Image-2512 page](qwen-image-2512.md)). The original checkpoint was not run on real weights for this check. `restore()` after the drop leaves the slot empty.
- **Self-contained.** The variant defines its own internal handle, branch-pairing state machine, and forward, importing no sibling variant — only the model-agnostic `_kernel/`, the public handle, and the shared mflux lifecycle helpers.
- The comparison harness clears the MLX buffer cache between reps for this variant; without the cache clear, accumulation across reps in one process would OOM the second rep. Its soft memory limit is advisory and never stopped an allocation; the watchdog described above is what bounds the run.
