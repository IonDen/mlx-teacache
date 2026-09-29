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
