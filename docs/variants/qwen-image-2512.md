# qwen-image-2512

Qwen-Image-2512 (Alibaba) is a newer checkpoint of the Qwen-Image architecture, and on mflux 0.19 and later it is what the `qwen-image` alias loads. It has the same transformer as [`qwen-image`](qwen-image.md), so this variant runs the same proxy forward, taps the same gate signal and handles CFG the same way. The polynomial and the default threshold are its own. The original checkpoint's coefficients do not fit this one: scored on 2512's calibration data they give R² −0.35 on the fit prompts and −0.20 on the held-out ones, worse than predicting a constant.

## Facts

- Model: [`Qwen/Qwen-Image-2512`](https://huggingface.co/Qwen/Qwen-Image-2512), not gated.
- License: Apache-2.0.
- Recipe: 50 steps, guidance 4.0 (CFG), 768×768, 4-bit, the same recipe as the original Qwen-Image.
- Default `rel_l1_thresh`: **0.25**, set from the threshold sweep below. The original checkpoint's 0.30 drops below the 0.95 SSIM floor on this one.
- Skip window: `skip_first_n_steps=1`, `skip_last_n_steps=1`.
- Patch strategy: the Qwen-Image proxy transformer (`ProxyQwenTransformer` + `qwen_forward_with_gate`), so the README's `flux.parameters()` caveat applies here too.
- Every number on this page was measured on a MacBook Pro M1 Max 32 GB, macOS 27, mflux 0.23.0, MLX 0.32.2, with the model quantized to 4 bits at load (`quantize=4`).

## Construct via mflux

```python
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.common.config.model_config import ModelConfig

flux = QwenImage(quantize=4, model_config=ModelConfig.qwen_image())  # Qwen-Image-2512 on mflux 0.19 and later
```

On mflux 0.19 and later, the config behind `qwen-image`, `qwen`, `qwen-image-2512` and `qwen-2512` loads Qwen-Image-2512, and `apply_teacache` picks this variant for it. Releases before 0.14.0 used the original checkpoint's coefficients here and raised `TeaCacheUncalibratedCheckpointWarning`; this variant does not warn. `match_variant` returns `qwen-image-2512` with `calibrated=True` for the same config. Detecting 2512 needs mflux 0.19 or later: mflux 0.17.5 and 0.18 have no 2512 config, so Qwen-Image-2512 loaded there by its repo id is resolved as a copy of `Qwen/Qwen-Image` and gets the original variant, reported as calibrated.

The rule that separates the two Qwen variants: a Qwen-Image config whose aliases include `qwen-image-2512` is this variant, unless its model name is `Qwen/Qwen-Image` (in any letter case), which stays the original variant. So on mflux 0.19 and later the alias names `qwen-image` and `qwen` select 2512. On mflux 0.17.5 and 0.18 the alias is the original Qwen-Image and nothing changes there. The Qwen-Image-Edit pipelines (including Edit-2511's `zero_cond_t` path) and Qwen-Image-2.1 match neither variant and raise `IncompatibleModelError`.

### Original Qwen-Image weights on mflux 0.19 and later

On mflux 0.19 and later the original checkpoint keeps its own variant in one case only: you build the model in Python and pass `ModelConfig.from_name("Qwen/Qwen-Image")` as its `model_config`, with no `model_path`. Every other route resolves against the `qwen-image` config, which is Qwen-Image-2512:

- `mflux-generate-qwen` refuses `--model Qwen/Qwen-Image` (the command accepts only the config's aliases) and `--base-model Qwen/Qwen-Image`.
- Any `--model-path` (in Python, `model_path`), whether a local directory or a downloaded mirror, resolves to the 2512 config, with or without `--base-model qwen-image`.
- In Python, `ModelConfig.from_name()` on a mirror such as `mlx-community/Qwen-Image-4bit` with no base model also infers a 2512 copy.

Nothing in the resolved config records that the weights came from the original checkpoint, so TeaCache detects them as 2512, uses this variant's coefficients and 0.25 default, and does not warn. To run original weights loaded that way on the original calibration, pass both its coefficients and its threshold:

```python
from mlx_teacache import apply_teacache
from mlx_teacache.variants.qwen_image.config import COEFFICIENTS as ORIGINAL_COEFFICIENTS

handle = apply_teacache(flux, coefficients=ORIGINAL_COEFFICIENTS, rel_l1_thresh=0.30)
```

You need both: with your own coefficients the default threshold becomes 0.20, and the calibrated-range clamp is off, so the skips can differ a little from what the `qwen-image` variant does with the same tuple.

## Coefficient provenance

`scripts/calibrate_qwen.py --model Qwen/Qwen-Image-2512` ran the same capture as the original checkpoint's calibration: 10 prompts (7 for the fit, 3 held out), 768×768, 50 steps, guidance 4.0, seed 42, 4-bit, one subprocess per prompt. It records Signal A, the modulated block-0 image input that the original variant gates on, and fits a degree-4 polynomial through the origin.

- Fit R² 0.818, held-out R² 0.785.
- The fit covers input changes from 0.0809 to 0.4706. The gate clamps a smaller change to the bottom of that range and always computes on a larger one (`CALIBRATED_RANGE`, see `docs/calibration.md`).
- The fit and the per-step pairs are committed as `scripts/_calibration_qwen_2512.json`, and a test pins the config's tuple and range to it.

Each calibration prompt peaked at 25.93 GiB. That is above the roughly 25 GiB recommended working set of a 32 GB M1 Max. The run finished with mflux's memory saver freeing the text encoders after the prompt was encoded and with other apps closed.

## Threshold sweep

`scripts/sweep_threshold_qwen.py --model Qwen/Qwen-Image-2512 --plain-q4`, red-apple prompt ("a red apple on a wooden table"), seed 42, 768×768, 50 steps, guidance 4.0, 4-bit, one subprocess per threshold, each image compared with plain mflux by SSIM:

| `rel_l1_thresh` | Skipped (of 48 active) | Longest run of skips | SSIM vs plain mflux |
|---|---|---|---|
| 0.15 | 22 | 3 | 0.991 |
| 0.20 | 27 | 4 | 0.978 |
| 0.25 | 29 | 4 | 0.969 |
| 0.30 | 32 | 5 | 0.947 |
| 0.40 | 35 | 6 | 0.928 |

The default is the highest swept threshold that keeps SSIM at or above 0.95, keeps the longest run of skips under the cap of 8 and shows no visible artefact. At 0.25 the image looks the same as plain mflux, with slightly softer fine detail on the apple skin. Each sweep run peaked at 25.64 GiB. Replaying the shipped gate over the calibration traces, with no weights loaded, gives the same 29 skips and longest run of 4 at 0.25.

This release has no speed benchmark for Qwen-Image-2512. The sweep ran one generation per threshold, so its timings are not reported as a speedup.

## Memory saver

When mflux's memory saver drops the transformer after the denoising loop (`--low-ram` for a single seed), TeaCache keeps no reference to it, and the decode runs without the transformer's weights in memory. We checked this once on Qwen-Image-2512 at 4 bits, 768×768 and 8 steps, with `MemorySaver(keep_transformer=False)`, which is what `--low-ram` registers for one seed. With TeaCache applied, MLX's active memory going into the decode was 0.00 GiB. A control run that kept one reference to the transformer, the way 0.13.2 did, had 12.30 GiB.

## Real-weights check

Three parity tests ran on real Qwen-Image-2512 weights (`-k 2512`, mflux 0.23.0, MLX 0.32.2, M1 Max 32 GB, 768×768, 4-bit) and passed:

- At threshold 0 the latents have a cosine similarity of at least 0.99 with plain mflux's latents, and `restore()` leaves no trace.
- At the 0.25 default the image keeps SSIM at or above 0.95 against plain mflux, and the skip count falls inside the 24–34 band the test allows (the sweep counted 29 at this recipe).
- The high-threshold skip path gives a finite latent.

The original checkpoint's parity tests (`-k original`) were not run for this release.

## Reproduce

```bash
uv run python scripts/calibrate_qwen.py --model Qwen/Qwen-Image-2512     # one subprocess per prompt, resumes past finished prompts
uv run python scripts/sweep_threshold_qwen.py --model Qwen/Qwen-Image-2512 --plain-q4 --thresholds 0.15,0.20,0.25,0.30,0.40 --max-units 1
uv run pytest tests/test_parity_qwen.py -m parity -k 2512
```

`tests/test_parity_qwen.py` covers both checkpoints and refuses a session that selects both, so run `-k original` and `-k 2512` in separate processes.

## License

[Apache-2.0](https://huggingface.co/Qwen/Qwen-Image-2512).
