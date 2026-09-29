# tests/_variant_fakes.py
"""Fake mflux pipelines for tests that must run the real TeaCache lifecycle.

Each factory returns an object the variant detector accepts (a bare instance of
the real mflux class with just enough attributes set) whose ``generate_image``
drives callbacks in mflux 0.18 order: before-loop callbacks, one transformer
call per denoising step, after-loop callbacks, then the result. The transformer
is the fake inner below, so the real proxy and the real gate run on tiny
zero tensors and nothing loads weights.

Test utility only: nothing in ``src/`` imports this module. Factories import
mflux lazily, so importing this module is safe in the pure-core lane; calling a
factory needs the ``[mflux]`` extra.
"""

from types import SimpleNamespace
from typing import Any

import mlx.core as mx

from tests._fakes import FaithfulCallbackRegistry


class _FakeJointBlock:
    def norm1(self, hidden_states: mx.array, text_embeddings: mx.array) -> tuple[mx.array, ...]:
        return (hidden_states, mx.zeros((1,)), mx.zeros((1,)), mx.zeros((1,)), mx.zeros((1,)))


class _FakeFlux1Inner:
    """The Transformer surface `flux1_forward_with_gate` touches, on tiny zero tensors."""

    text_seq = 2
    img_seq = 4
    dim = 8

    def __init__(self) -> None:
        self.transformer_blocks: list[Any] = [_FakeJointBlock()]
        self.single_transformer_blocks: list[Any] = [object()]
        self.pos_embed = object()
        self.time_text_embed = object()

    def __call__(self, **_kw: Any) -> mx.array:
        # What the unpatched model does when a step calls it directly.
        return mx.zeros((1, self.img_seq, self.dim))

    def x_embedder(self, hidden_states: mx.array) -> mx.array:
        return mx.zeros((1, self.img_seq, self.dim))

    def context_embedder(self, prompt_embeds: mx.array) -> mx.array:
        return mx.zeros((1, self.text_seq, self.dim))

    def compute_text_embeddings(
        self, t: int, pooled: mx.array, time_text_embed: Any, config: Any
    ) -> mx.array:
        return mx.zeros((1, self.dim))

    def compute_rotary_embeddings(
        self, prompt_embeds: mx.array, pos_embed: Any, config: Any, kontext_image_ids: Any
    ) -> mx.array:
        return mx.zeros((1, self.text_seq + self.img_seq, self.dim // 2))

    def _apply_joint_transformer_block(self, *, hidden_states, encoder_hidden_states, **_kw):  # noqa: ANN001, ANN003, ANN202
        return encoder_hidden_states, hidden_states

    def _apply_single_transformer_block(self, *, hidden_states, **_kw):  # noqa: ANN001, ANN003, ANN202
        return hidden_states

    def norm_out(self, x: mx.array, text_embeddings: mx.array) -> mx.array:
        return x

    def proj_out(self, x: mx.array) -> mx.array:
        return x


def make_flux1_fake(alias: str = "dev") -> Any:
    """A FLUX.1 fake whose ``generate_image(seed, prompt, num_inference_steps, ...)`` runs the
    mflux 0.18 callback order and returns the string ``"image"``.

    ``_extra_transformer_calls=N`` makes the loop call the transformer N extra times, so the
    number of recorded step decisions differs from the active step count.
    """
    from mflux.models.flux.variants.txt2img.flux import Flux1

    flux = Flux1.__new__(Flux1)
    flux.model_config = SimpleNamespace(model_name="black-forest-labs/FLUX.1-" + alias, aliases=[alias])
    flux.transformer = _FakeFlux1Inner()
    flux.callbacks = FaithfulCallbackRegistry()

    def generate_image(
        seed: int = 0,
        prompt: str = "",
        num_inference_steps: int = 4,
        _extra_transformer_calls: int = 0,
        **_kw: Any,
    ) -> str:
        config = SimpleNamespace(num_inference_steps=num_inference_steps, init_time_step=0)
        latents = mx.zeros((1, 4, 8))
        for cb in list(flux.callbacks.before_loop):
            cb.call_before_loop(seed=seed, prompt=prompt, latents=latents, config=config)
        for t in range(num_inference_steps + _extra_transformer_calls):
            flux.transformer(
                t=t,
                config=config,
                hidden_states=latents,
                prompt_embeds=mx.zeros((1, 2, 8)),
                pooled_prompt_embeds=mx.zeros((1, 8)),
            )
        for cb in list(flux.callbacks.after_loop):
            cb.call_after_loop(seed=seed, prompt=prompt, latents=latents, config=config)
        return "image"

    flux.generate_image = generate_image
    return flux


# Alias that each registry id's detector looks for, and the family it belongs to.
_FLUX1_ALIASES = {"flux1-dev": "dev", "flux1-schnell": "schnell", "flux1-krea-dev": "krea-dev"}
_DUCK_ALIASES = {
    "flux2-klein-4b": "flux2-klein-4b",
    "flux2-klein-9b": "flux2-klein-9b",
    "flux2-klein-base-4b": "flux2-klein-base-4b",
    "flux2-klein-base-9b": "flux2-klein-base-9b",
    "z-image-base": "z-image",
    "qwen-image": "qwen-image",
}


def fake_for_variant(variant_id: str) -> Any:
    """A model that ``variant_id``'s detector matches and whose ``apply()`` can patch without weights.

    Covers every id in the variant registry. FLUX.1 ids get the real-class fake from
    ``make_flux1_fake`` (needs the ``[mflux]`` extra); the FLUX.2, Z-Image and Qwen-Image ids get a
    duck-typed namespace (the detectors only read ``model_config.aliases``). Qwen-Image declares the
    checkpoint the built-in polynomial was fitted on, so applying it raises no uncalibrated warning.
    The fake is patched and restored only; it is not a working generator except for FLUX.1.

    Distilled ids (``flux2-klein-4b``, ``-9b``) still emit ``TeaCacheNoBenefitWarning`` on apply
    with built-in coefficients; the caller must expect it.
    """
    if variant_id in _FLUX1_ALIASES:
        return make_flux1_fake(_FLUX1_ALIASES[variant_id])
    if variant_id not in _DUCK_ALIASES:
        raise KeyError(f"no fake for variant {variant_id!r}")
    model_config: dict[str, Any] = {"aliases": [_DUCK_ALIASES[variant_id]]}
    if variant_id == "qwen-image":
        from mlx_teacache.variants.qwen_image.config import META

        model_config["model_name"] = META["hf_model_id"]
    return SimpleNamespace(
        model_config=SimpleNamespace(**model_config),
        transformer=SimpleNamespace(name="real-transformer"),
        callbacks=FaithfulCallbackRegistry(),
        generate_image=lambda **kw: "image",
    )
