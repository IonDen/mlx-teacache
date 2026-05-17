# src/mlx_teacache/integrations/mflux/flux2.py
"""FLUX.2 _predict replacement.

`flux._predict = make_teacache_predict_factory(handle)` makes
self._predict(self.transformer) (per Flux2Klein.generate_image:86) invoke
our factory, which returns a per-generation predict closure. The closure
runs eager Python — never wrapped in mx.compile — so step-level gating
remains live (per spike F3/F4 findings).

v0.4.1+: CFG steps are routed through flux2_cfg_forward_with_gate (gated,
per-branch caching). Skip-window validation runs on the first gated call
regardless of CFG, so an all-CFG generation with a bad window now raises
InvalidStepWindowError (behavior change documented in CHANGELOG)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx

from mlx_teacache.errors import InvalidStepWindowError, MissingGenerationContextError
from mlx_teacache.integrations.mflux.forward import (
    flux2_cfg_forward_with_gate,
    flux2_forward_with_gate,
)

PredictFn = Callable[
    [mx.array, mx.array, mx.array, mx.array, "mx.array | None", "mx.array | None", float, mx.array],
    Any,
]
PredictFactory = Callable[[Any], PredictFn]


def _vanilla_flux2_cfg_predict(
    transformer: Any,
    latents: mx.array,
    latent_ids: mx.array,
    prompt_embeds: mx.array,
    text_ids: mx.array,
    negative_prompt_embeds: mx.array,
    negative_text_ids: mx.array,
    guidance: float,
    timestep: mx.array,
) -> Any:
    """Bit-exact mirror of mflux's Flux2Klein._predict closure CFG path
    (flux2_klein.py:259-276). v0.4.1+: this function is a **test-only
    diagnostic reference** used by tests/test_parity_flux2.py to verify
    that flux2_cfg_forward_with_gate at rel_l1_thresh<=0 produces the
    same output as vanilla CFG math. It is no longer called from the
    production predict closure."""
    noise = transformer(
        hidden_states=latents,
        encoder_hidden_states=prompt_embeds,
        timestep=timestep,
        img_ids=latent_ids,
        txt_ids=text_ids,
        guidance=None,
    )
    negative_noise = transformer(
        hidden_states=latents,
        encoder_hidden_states=negative_prompt_embeds,
        timestep=timestep,
        img_ids=latent_ids,
        txt_ids=negative_text_ids,
        guidance=None,
    )
    return negative_noise + guidance * (noise - negative_noise)


def make_teacache_predict_factory(handle: Any) -> PredictFactory:
    """Return a callable assignable to `flux._predict`. Mflux will call it
    as `predict = self._predict(self.transformer)` at the top of each
    generate_image."""

    def predict_factory(transformer: Any) -> PredictFn:
        # Closure-local — fresh per generation. Survives mid-loop crashes of
        # previous runs because Flux2Klein.generate_image builds a new closure
        # every time it calls self._predict(self.transformer).
        context_consumed = False

        def predict(
            latents: mx.array,
            latent_ids: mx.array,
            prompt_embeds: mx.array,
            text_ids: mx.array,
            negative_prompt_embeds: mx.array | None,
            negative_text_ids: mx.array | None,
            guidance: float,
            timestep: mx.array,
        ) -> Any:
            nonlocal context_consumed
            ctx = handle._gen_ctx

            # 1. Consume context on the first call of THIS generation's closure.
            if not context_consumed:
                if ctx.active_num_steps is None or ctx.consumed_at_token == ctx.token:
                    raise MissingGenerationContextError()
                ctx.consumed_at_token = ctx.token
                context_consumed = True

            # 2. Lazy skip-window validation. v0.4.1: lifted up so it runs on
            #    the first gated call regardless of CFG. In v0.4.0 the CFG
            #    branch bypassed this; an all-CFG generation with a bad
            #    window silently ran vanilla. This is now a behavior change
            #    documented in CHANGELOG.
            if not handle._state.cache.skip_window_validated:
                if handle.skip_first_n_steps + handle.skip_last_n_steps >= ctx.active_num_steps:
                    raise InvalidStepWindowError(
                        skip_first=handle.skip_first_n_steps,
                        skip_last=handle.skip_last_n_steps,
                        num_steps=ctx.active_num_steps,
                    )
                handle._state.cache.skip_window_validated = True

            # 3. CFG branch: gated per-branch caching.
            cfg_active = negative_prompt_embeds is not None and negative_text_ids is not None
            if cfg_active:
                assert negative_prompt_embeds is not None
                assert negative_text_ids is not None
                handle._state.stats._staging.cfg_was_active = True
                return flux2_cfg_forward_with_gate(
                    transformer,
                    handle,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_text_ids=negative_text_ids,
                    guidance=guidance,
                    timestep=timestep,
                    img_ids=latent_ids,
                )

            # 4. Non-CFG branch: unchanged from v0.4.0.
            return flux2_forward_with_gate(
                transformer,
                handle,
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                img_ids=latent_ids,
                txt_ids=text_ids,
            )

        return predict

    return predict_factory
