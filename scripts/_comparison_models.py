"""mflux model loading for the comparison harness: staged eager load, pre-encoded prompts, encoders freed.

mflux loads weights lazily (mx.load mmap + nn.quantize, no eval), so the harness evaluates them before its timers.
The order matters for memory: text encoders first, then the prompt is encoded and evaluated, then (where the recipe
frees encoders) the encoders are released, and only then the transformer and VAE are evaluated. Evaluating
everything at once would hold Qwen's bf16 text encoder (~16 GB, never quantized) and its transformer together.
"""

import gc
from collections.abc import Callable, Sequence
from typing import Any

from _comparison_recipes import Recipe

ENCODER_ATTRS: tuple[str, ...] = ("clip_text_encoder", "t5_text_encoder", "text_encoder")
DENOISER_ATTRS: tuple[str, ...] = ("transformer", "vae")
_CACHE_LOADERS = ("flux1-dev", "flux1-krea-dev", "qwen-image-original")


def qwen_original_config() -> Any:
    from mflux.models.common.config.model_config import ModelConfig

    return ModelConfig.from_name("Qwen/Qwen-Image")


def load_model(recipe: Recipe) -> Any:
    from mflux.models.common.config.model_config import ModelConfig

    if recipe.loader in ("flux1-dev", "flux1-krea-dev"):
        from mflux.models.flux.variants.txt2img.flux import Flux1

        config = ModelConfig.dev() if recipe.loader == "flux1-dev" else ModelConfig.krea_dev()
        flux = Flux1(quantize=recipe.quantize, model_config=config)
    elif recipe.loader in ("klein-base-4b", "klein-base-9b"):
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

        config = (
            ModelConfig.flux2_klein_base_4b()
            if recipe.loader == "klein-base-4b"
            else ModelConfig.flux2_klein_base_9b()
        )
        flux = Flux2Klein(quantize=recipe.quantize, model_config=config)
    elif recipe.loader == "z-image":
        from mflux.models.z_image.variants.z_image import ZImage

        flux = ZImage(quantize=recipe.quantize, model_config=ModelConfig.z_image())
    elif recipe.loader == "qwen-image-original":
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

        flux = QwenImage(quantize=recipe.quantize, model_config=qwen_original_config())
        if flux.model_config.model_name != "Qwen/Qwen-Image":
            raise RuntimeError(
                f"expected the original Qwen/Qwen-Image, loaded {flux.model_config.model_name!r}"
            )
    else:
        raise ValueError(f"unknown loader {recipe.loader!r}")
    flux.freeze()
    return flux


def evaluate_modules(flux: Any, names: Sequence[str]) -> None:
    import mlx.core as mx

    for name in names:
        module = getattr(flux, name, None)
        if module is not None:
            mx.eval(module.parameters())


def _cached_call(result: Any, key: dict[str, Any], label: str) -> Callable[..., Any]:
    def _call(**kwargs: Any) -> Any:
        if kwargs != key:
            raise RuntimeError(f"{label}: prompt-encoding cache miss (got {kwargs!r})")
        return result

    return _call


def precompute_prompt(flux: Any, recipe: Recipe, prompt: str) -> None:
    import mlx.core as mx

    if recipe.loader in ("flux1-dev", "flux1-krea-dev"):
        from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder

        embeds = PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=flux.prompt_cache,
            t5_tokenizer=flux.tokenizers["t5"],
            clip_tokenizer=flux.tokenizers["clip"],
            t5_text_encoder=flux.t5_text_encoder,
            clip_text_encoder=flux.clip_text_encoder,
        )
        mx.eval(embeds)
    elif recipe.loader in ("klein-base-4b", "klein-base-9b"):
        key = {"prompt": prompt, "negative_prompt": " ", "guidance": recipe.guidance}  # flux2_klein.py:76-80
        result = flux._encode_prompt_pair(**key)
        mx.eval([x for x in result if x is not None])
        flux._encode_prompt_pair = _cached_call(result, key, recipe.slug)
    elif recipe.loader == "z-image":
        key = {"prompt": prompt, "negative_prompt": None, "guidance": recipe.guidance}  # z_image.py:97-101
        result = flux._encode_prompts(**key)
        mx.eval([x for x in result if x is not None])
        flux._encode_prompts = _cached_call(result, key, recipe.slug)
    elif recipe.loader == "qwen-image-original":
        from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder

        embeds = QwenPromptEncoder.encode_prompt(
            prompt=prompt,
            negative_prompt=" ",
            prompt_cache=flux.prompt_cache,
            qwen_tokenizer=flux.tokenizers["qwen"],
            qwen_text_encoder=flux.text_encoder,
        )
        mx.eval(embeds)
    else:
        raise ValueError(f"unknown loader {recipe.loader!r}")


def assert_prompt_cache_hit(flux: Any, recipe: Recipe) -> None:
    if recipe.loader in _CACHE_LOADERS:
        n = len(flux.prompt_cache)
        if n != 1:
            raise RuntimeError(
                f"{recipe.slug}: prompt cache holds {n} entries; generation re-encoded the prompt"
            )


def release_text_encoders(flux: Any) -> list[str]:
    import mlx.core as mx

    released = [name for name in ENCODER_ATTRS if getattr(flux, name, None) is not None]
    for name in released:
        setattr(flux, name, None)
    gc.collect()
    mx.clear_cache()
    return released
