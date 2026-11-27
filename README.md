# mlx-teacache

<p align="center">
  <img src="https://raw.githubusercontent.com/IonDen/mlx-teacache/main/docs/assets/mlx-teacache-logo.png" alt="mlx-teacache" width="100%">
</p>

[![PyPI version](https://img.shields.io/pypi/v/mlx-teacache.svg)](https://pypi.org/project/mlx-teacache/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://pypi.org/project/mlx-teacache/)
[![License: Apache 2.0](https://img.shields.io/pypi/l/mlx-teacache.svg)](https://github.com/IonDen/mlx-teacache/blob/main/LICENSE)
[![CI](https://github.com/IonDen/mlx-teacache/actions/workflows/ci.yml/badge.svg)](https://github.com/IonDen/mlx-teacache/actions/workflows/ci.yml)

Image models such as FLUX and Qwen-Image build a picture over many steps. By default, [mflux](https://github.com/mflux-community/mflux) runs every step in full on Apple Silicon with MLX, so a picture takes as long as all its steps. `mlx-teacache` skips the costly part of steps that barely change the picture and reuses the previous step's result there ([measured speed-ups](https://github.com/IonDen/mlx-teacache/blob/main/docs/benchmarks.md)).

You need this if you generate images with mflux on an Apple Silicon Mac, with a model that is not distilled (distilled models are trained to finish in a few steps and skip none), at the 25 to 50 steps we measured.

You can:

- Wrap a model in one call, `apply_teacache(flux)`; nothing is retrained or downloaded.
- Get an image very close to plain mflux's.
- Set one threshold, `rel_l1_thresh`: higher is faster, lower stays closer to plain mflux.
- Check support before loading; read per-run stats.

## Try it

```bash
pip install "mlx-teacache[mflux]"
```

This needs Python 3.11 or newer on Apple Silicon; version notes are under [Install](#install). FLUX.2 Klein base 4B is a good first test: it is small, Apache-2.0 and needs no license click-through.

```python
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mlx_teacache import apply_teacache

flux = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_base_4b())
with apply_teacache(flux) as handle:  # this model's default rel_l1_thresh=0.17
    generated = flux.generate_image(
        prompt="a red apple on a wooden table",
        seed=42,
        num_inference_steps=50,
        guidance=4.0,
        width=512,
        height=512,
    )
    print(f"Skipped {handle.stats.skipped_count} steps, speedup ≈ {handle.stats.speedup_estimate:.2f}×")
generated.image.save("apple.png")
```

The first run downloads the weights; per-model timings are on the [benchmarks page](https://github.com/IonDen/mlx-teacache/blob/main/docs/benchmarks.md). Every supported model is wrapped the same way; [`docs/variants/`](https://github.com/IonDen/mlx-teacache/tree/main/docs/variants) has each constructor and recipe. Qwen-Image at 4-bit and 768 by 768 pixels is at the limit of the GPU memory a 32 GB Mac gives. That was measured on the original Qwen-Image checkpoint ([record](https://github.com/IonDen/mlx-teacache/blob/main/docs/variants/qwen-image.md#benchmark)); the Qwen-Image-2512 peak is not recorded.

## Install

```bash
pip install "mlx-teacache[mflux]"
# or with uv:
uv add "mlx-teacache[mflux]"
```

```bash
pip install "mlx-teacache[mflux]==0.18.1"  # pin for reproducibility
```

Requires Python ≥ 3.11 and Apple Silicon. The `[mflux]` extra pulls in `mflux>=0.24`. Version notes:

- mlx-teacache 0.18.1 has no upper bound on mflux. A newer mflux installs and is patched but untested, and `apply_teacache` warns until it is. [docs/stability.md](https://github.com/IonDen/mlx-teacache/blob/main/docs/stability.md) describes what 1.0 will promise.
- mflux 0.24 needs Pillow 13, which fixes a critical Pillow security advisory ([GHSA-5x8w-8jj5-q52h](https://github.com/advisories/GHSA-5x8w-8jj5-q52h); Pillow 12.3 and older are affected). The `[mflux]` extra therefore installs the fixed Pillow. The base install pulls in neither mflux nor Pillow.
- Python 3.10 and mflux 0.17.5 through 0.23.x are no longer supported. If you need either, stay on the 0.14 line: `pip install "mlx-teacache[mflux]<0.15"`. Its last release, 0.14.1, runs on mflux 0.24 but warns that it is untested there.
- mflux 0.24 needs MLX 0.32 (below 0.33), torch 2.13 or newer, opencv-python, Pillow 13 or newer and Python 3.11 or newer.
- For mlx-taef's live preview, use mlx-taef 0.9.0 or later. `mlx-teacache[mflux]` and `mlx-taef[mflux]>=0.9.0` install together.
- From mflux 0.19 on, the `qwen-image` alias loads `Qwen/Qwen-Image-2512`, which has its own variant, so `apply_teacache` does not warn about it. The original `Qwen/Qwen-Image` loads only from Python, with `ModelConfig.from_name("Qwen/Qwen-Image")`; the command line, any `model_path` and any mirror resolve to 2512. Qwen-Image-2.1 (mflux 0.20 and later, non-commercial) is not supported, and `apply_teacache` raises `IncompatibleModelError` for it.

## From the mflux command line

If you run mflux from the terminal rather than from Python, the `mflux-teacache` plugin adds a TeaCache version of the Z-Image command. It takes every option of `mflux-generate-z-image` and prints how many steps it skipped.

Run `pip install mflux-teacache` in an environment that already has mflux. For a fresh setup, follow the [Install section of the plugin's README](https://github.com/mflux-community/mflux-plugins/tree/main/teacache#install).

```bash
pip install mflux-teacache
mflux-generate-z-image-teacache --prompt "a lighthouse on a cliff at dusk" --steps 50 --guidance 4.0 --seed 42 -q 8
```

It is the only command today, and it covers Z-Image base only. Before it loads weights, it refuses an unsupported or distilled model and a run with too few steps. It has no step-cache option, so mflux's own step cache never comes into play. It needs mflux 0.22 or newer and mlx-teacache 0.14.0 or newer. The [plugin's README](https://github.com/mflux-community/mflux-plugins/tree/main/teacache#readme) lists the options and versions.

## img2img

mlx-teacache supports mflux's image-to-image generation. Pass `image_path` and `image_strength > 0` to `flux.generate_image()` with TeaCache active:

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

Caching covers only the active denoising window, `num_inference_steps - init_time_step`. So `image_strength=0.7` with `num_inference_steps=25` leaves 8 predict calls to cache. At `image_strength=1.0` mflux runs no denoising, so TeaCache does nothing.

## Threshold guide

`rel_l1_thresh` trades speed against closeness to the plain image: a lower value skips fewer steps and stays closer. The default differs by model (0.25 on Qwen-Image-2512, 0.30 on Qwen-Image and FLUX.1 Krea [dev], 0.17 on FLUX.2 Klein base, 0.12 on Z-Image); FLUX.1 dev and schnell and the distilled Klein models use the package default, 0.20. On FLUX.1 dev, 0.10 and 0.15 skipped nothing in single runs, so they give no gain, and at 0.25 text prompts that plain mflux renders as neon tubes can come out as dot-matrix; no record of those three runs was committed. The [FLUX.1 dev sweep](https://github.com/IonDen/mlx-teacache/blob/main/docs/benchmarks.md#threshold-sweep-for-flux1-dev) has the table and the two records behind the 0.20 default; the [variant pages](https://github.com/IonDen/mlx-teacache/tree/main/docs/variants) cover the other models.

## Supported models

Only the models below are supported; `apply_teacache` refuses any other. Distilled models finish in 4 to 8 steps and skip none of them; see [When to use mlx-teacache](#when-to-use-mlx-teacache). In the table, g is the guidance scale. For Qwen-Image, FLUX.2 Klein base and Z-Image, g above 1.0 adds a negative-prompt pass each step (classifier-free guidance, CFG); FLUX.1 dev and Krea take g as an input to the network and still run one pass.

The table is generated from the variant registry by `docs/_generate_supported_models.py`, grouped by family and ordered by how widely each family is used on Macs ([how that was checked](https://github.com/IonDen/mlx-teacache/blob/main/docs/benchmarks.md#per-variant-notes)). Per-variant detail lives in [`docs/variants/<id>.md`](https://github.com/IonDen/mlx-teacache/tree/main/docs/variants).

<!-- SUPPORTED_MODELS_START -->
| Variant id | Display name | Distilled? | Default recipe | License |
|---|---|---|---|---|
| `qwen-image-2512` | Qwen-Image-2512 | no | 50 steps, g=4.0 | [Apache-2.0](https://huggingface.co/Qwen/Qwen-Image-2512) |
| `qwen-image` | Qwen-Image | no | 50 steps, g=4.0 | [Apache-2.0](https://huggingface.co/Qwen/Qwen-Image) |
| `flux2-klein-base-9b` | FLUX.2 Klein base 9B | no | 50 steps, g=4.0 | [FLUX Non-Commercial](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) |
| `flux2-klein-base-4b` | FLUX.2 Klein base 4B | no | 50 steps, g=4.0 | [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) |
| `flux2-klein-4b` | FLUX.2 Klein 4B | yes | 8 steps, g=1.0 | [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| `flux2-klein-9b` | FLUX.2 Klein 9B | yes | 8 steps, g=1.0 | [FLUX Non-Commercial](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) |
| `z-image-base` | Z-Image base | no | 50 steps, g=4.0 | [Apache-2.0](https://huggingface.co/Tongyi-MAI/Z-Image) |
| `flux1-dev` | FLUX.1 dev | no | 25 steps, g=3.5 | [FLUX.1-dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| `flux1-krea-dev` | FLUX.1 Krea [dev] | no | 28 steps, g=4.5 | [FLUX.1-dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev) |
| `flux1-schnell` | FLUX.1 schnell | yes | 4 steps, g=1.0 | [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |
<!-- SUPPORTED_MODELS_END -->

### Per-variant notes

Each variant has its own page under [`docs/variants/`](https://github.com/IonDen/mlx-teacache/tree/main/docs/variants) with its mflux constructor, recipe, license obligations and quirks. The [benchmarks page](https://github.com/IonDen/mlx-teacache/blob/main/docs/benchmarks.md#per-variant-notes) has the measured notes for each.

- Original Qwen-Image weights loaded through the command line, a `model_path` or a mirror resolve to Qwen-Image-2512 on mflux 0.19 and later. To run them on the original calibration, pass `coefficients=` with the original tuple and `rel_l1_thresh=0.30` (see the [Qwen-Image-2512 page](https://github.com/IonDen/mlx-teacache/blob/main/docs/variants/qwen-image-2512.md)).
- Stock 4-bit Qwen-Image is grainy on a 32 GB Mac. The [Qwen-Image page](https://github.com/IonDen/mlx-teacache/blob/main/docs/variants/qwen-image.md) documents a mixed-precision build that clears it; mlx-teacache does not change quantization.
- FLUX.2 Klein base 9B is non-commercial (see [License obligations](#license-obligations)). Accept the license on its [Hugging Face page](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) before downloading.

## When to use mlx-teacache

TeaCache helps when neighboring steps look alike. So far that held for non-distilled models at 25 to 50 steps, as measured.

- Use it for `qwen-image` at 50 steps with CFG, `flux2-klein-base-9b` and `flux2-klein-base-4b` at 25-50 steps with or without CFG, `z-image-base` at 50 steps with CFG, `flux1-dev` at 25-50 steps, and `flux1-krea-dev` at its 28-step recipe. [COMPARISON.md](https://github.com/IonDen/mlx-teacache/blob/main/COMPARISON.md) shows one scene per model. `qwen-image-2512` is calibrated but has no speed measurement yet.
- Do not use it on the distilled models: `flux1-schnell`, `flux2-klein-4b` and `flux2-klein-9b`. The gate skips zero steps there and adds a small overhead (not measured on its own), so run plain mflux; `apply_teacache` warns. The [research note](https://ineshin.space/papers/why-teacache-does-not-engage-on-short-distilled-schedules/) explains why.

On FLUX.2 and Z-Image there is also a small effect that is separate from skipping steps. On Macs where mflux compiles its prediction step (Max, Ultra, M3 and newer), the wrapper sidesteps that compile. On base and Pro M1 and M2, mflux already runs that step uncompiled, so there is nothing to sidestep. There, Klein base should keep the gain from skipped steps, and distilled Klein, which skips nothing, is about neutral or slightly slower than plain mflux; neither was measured on those chips. Speed on M3 and newer is untested, and on M5 the eager wrapper can lose some or all of the Neural Accelerator gain. FLUX.1, Krea and Qwen-Image should gain the same on every chip (measured on an M1 Max only; [per-chip table](https://github.com/IonDen/mlx-teacache/blob/main/docs/benchmarks.md#performance-by-chip)).

The distilled Klein model stays the everyday choice. Base with TeaCache is for when you need guidance, negative prompts or a fine-tune ([comparison](https://github.com/IonDen/mlx-teacache/blob/main/docs/benchmarks.md#distilled-klein-against-base)).

## Combining with mlx-taef

For live previews while generating, add [`mlx-taef`](https://github.com/IonDen/mlx-taef), tiny preview decoders in MLX. It works alongside TeaCache.

```python
from mlx_taef.integrations.mflux import LivePreviewCallback
from mlx_teacache import apply_teacache

preview = LivePreviewCallback(variant="taef2", every=5, save_to="preview.png",
                              latent_height=32, latent_width=32)
flux.callbacks.register(preview)

with apply_teacache(flux):  # the variant's default rel_l1_thresh
    flux.generate_image(prompt="...", seed=42, num_inference_steps=25)
```

## Checking support before loading

```python
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.z_image.variants.z_image import ZImage

from mlx_teacache import check_step_window, match_variant

info = match_variant(ModelConfig.from_name("z-image"), ZImage)
print(info.variant_id, info.default_thresh)  # z-image-base 0.12

check_step_window(50)  # returns; check_step_window(2) raises InvalidStepWindowError
```

`match_variant` returns `None` for a model or pipeline that `apply_teacache` would refuse. Otherwise it returns a `VariantInfo`: variant id, display name, default threshold, detection names and `calibrated`. `default_thresh` is `None` for the distilled Klein 4B and 9B, where the gate skips nothing. `calibrated` is `False` only for a config that names neither the checkpoint the coefficients were fitted on nor that checkpoint as its base model; `apply_teacache` still patches such a model and warns. mflux records the base model in every config it resolves, so a finetune or mirror loaded through mflux reports `calibrated=True`: the field cannot tell it from the checkpoint itself.

`check_step_window` raises when a run has too few denoising steps for the skip window. By default, one step at each end always computes, so 3 or fewer active steps skip nothing and 2 or fewer are refused. A 3-step run passes `check_step_window`, but `apply_teacache` warns that it cannot skip anything. For image-to-image, pass the number of steps mflux actually runs. A count of 0 is refused too, which is stricter than `apply_teacache`, where a zero-step run is a valid no-op. Neither call reads weights, and both use the same window rule as `apply_teacache`.

## Inspecting stats

```python
handle = apply_teacache(flux)
flux.generate_image(prompt="apple", seed=42, num_inference_steps=25)
print(handle.stats.computed_count, handle.stats.skipped_count)
print(handle.stats.last_generation.decisions[5])  # per-step record
print(handle.decision_rule_revision, handle.mflux_version)
```

The last line prints the revision of the gate's decision rule and the mflux version the patch was applied on. Skip counts can differ between minor releases or mflux versions, and these two values show which of them changed.

## Custom coefficients

```python
custom = [...]  # length 5, all finite
apply_teacache(flux, coefficients=custom, rel_l1_thresh=0.20)  # use your own threshold
```

Pass a `rel_l1_thresh` with your coefficients. A threshold is tuned for one polynomial, so the model's default does not carry over to yours. If you leave it out, `apply_teacache` picks 0.30 on FLUX.1 Krea [dev] and 0.20 on every other model, and warns with a `FutureWarning`; from 1.0 the same call raises an error. To find a threshold for a new polynomial, run `scripts/sweep_threshold_qwen.py`, `scripts/sweep_threshold_z_image.py`, `scripts/sweep_threshold_klein_base_4b.py`, or `scripts/calibrate_flux1.py --sweep` for the FLUX.1 models. Your polynomial is evaluated on the raw input change. The calibrated-range clamp described on the [how-it-works page](https://github.com/IonDen/mlx-teacache/blob/main/docs/how-it-works.md) applies only to a variant's own built-in tuple.

## How it works

TeaCache runs only a cheap early part of the network and uses it to estimate how much a step's output will change. While that estimate stays under a threshold, it skips the expensive part. The [how-it-works page](https://github.com/IonDen/mlx-teacache/blob/main/docs/how-it-works.md) has the mechanism. The original method is in the [TeaCache paper](https://liewfeng.github.io/TeaCache/).

## Benchmarks

The [benchmarks page](https://github.com/IonDen/mlx-teacache/blob/main/docs/benchmarks.md) has the multi-run speed figures and the image-similarity figures behind them, the record each comes from, what that record does not name, and a command to reproduce each row. Single-run comparisons are in [COMPARISON.md](https://github.com/IonDen/mlx-teacache/blob/main/COMPARISON.md).

## Limitations

The [limitations page](https://github.com/IonDen/mlx-teacache/blob/main/docs/limitations.md) has the full notes. These change how you call the library:

- img2img reuses the txt2img coefficients.
- FLUX.2 output is close to plain mflux's but not bit-identical, even at threshold 0.
- On an mflux newer than 0.24.0 (0.24.1 included), `apply_teacache` emits a `TeaCacheUntestedMfluxWarning` once per process and still patches the model. If a run fails or looks wrong there, open an issue with both version numbers.
- On FLUX.1 and Qwen-Image the wrapper replaces `flux.transformer` with a proxy that the parent model's parameter tree does not reach. While it is active, `flux.parameters()`, `nn.quantize(flux)` or `flux.update(...)` on the parent can miss the transformer. Use `flux.transformer.parameters()` directly, or call `handle.restore()` first. Calls on the transformer itself work: `flux.transformer.load_weights(...)`, `set_dtype`, `eval()` and `nn.quantize(flux.transformer)`.
- When mflux's memory saver drops the transformer after the denoising loop (`model.transformer = None`), TeaCache keeps no copy, so a later `handle.restore()` leaves `flux.transformer` as `None`, the state plain mflux leaves.

## License obligations

The FLUX.1 variants (`flux1-dev`, `flux1-schnell`) and `flux2-klein-4b` come with their own upstream weight licenses; the wrapper this library applies does not change those terms.

`flux2-klein-9b` and `flux2-klein-base-9b` are both distributed under the FLUX.2 Klein license (non-commercial use + BFL safety-filter obligations). These terms flow with the weights, not with mlx-teacache. If you call `apply_teacache` on either variant (`Flux2Klein(model_config=ModelConfig.flux2_klein_9b())` or `Flux2Klein(model_config=ModelConfig.flux2_klein_base_9b())`), you are responsible for ensuring your use complies with the upstream license, including the safety-filter requirements the BFL model cards describe. See the official model cards at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B and https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B for the full terms.

## Research notes

Four short research notes (Qwen-Image mixed precision on a 32 GB Mac, why the gate skips nothing on distilled schedules, why byte-exact parity is a poor test, and benchmarking above the recommended working set) are listed on the [benchmarks page](https://github.com/IonDen/mlx-teacache/blob/main/docs/benchmarks.md#research-notes).

## Contributing

Questions and results go to [GitHub Discussions](https://github.com/IonDen/mlx-teacache/discussions); bugs go to [Issues](https://github.com/IonDen/mlx-teacache/issues). If you are adding a variant or changing the gate, [`docs/architecture.md`](https://github.com/IonDen/mlx-teacache/blob/main/docs/architecture.md) is the contributor's map and [`docs/calibration.md`](https://github.com/IonDen/mlx-teacache/blob/main/docs/calibration.md) covers coefficient calibration.

## License + acknowledgements

Apache-2.0. See `LICENSE` and `NOTICE`.

- [ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache): upstream method and FLUX.1 coefficients.
- [mflux-community/mflux](https://github.com/mflux-community/mflux): the MLX diffusion runner this library integrates with (moved from `filipstrand/mflux` in August 2026; the old URL redirects).
- [Apple ML Explore](https://github.com/ml-explore/mlx): MLX.

## Sister projects

Other MLX libraries for Apple Silicon:

- [mlx-taef](https://github.com/IonDen/mlx-taef): tiny autoencoders for fast diffusion-latent previews and low-memory decode (FLUX / SD).
- [mlx-model-doctor](https://github.com/IonDen/mlx-model-doctor): validate an MLX / Hugging Face model repo before you load it (config, tokenizer, safetensors, memory).
- [mlx-quant-fidelity](https://github.com/IonDen/mlx-quant-fidelity): measure how much quality a quantization costs (KL divergence, top-token flips, perplexity delta).

---

By Denis Ineshin · [ineshin.space](https://ineshin.space)
