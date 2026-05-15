# src/mlx_teacache/integrations/mflux/flux2.py
"""FLUX.2 _predict replacement.

`flux._predict = make_teacache_predict_factory(handle)` makes
self._predict(self.transformer) (per Flux2Klein.generate_image:86) invoke
our factory, which returns a per-generation predict closure. The closure
runs eager Python — never wrapped in mx.compile — so step-level gating
remains live (per spike F3/F4 findings).

CFG fallback: when negative embeds are passed, we run vanilla mflux CFG
(both transformer calls + manual combination) and record cfg-fallback
StepDecisions. Skip-window validation runs lazily on first non-CFG call."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mlx.core as mx

from mlx_teacache.errors import InvalidStepWindowError, MissingGenerationContextError
from mlx_teacache.integrations.mflux.forward import flux2_forward_with_gate
from mlx_teacache.stats import StepDecision

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
    (flux2_klein.py:259-276). Used when TeaCache auto-falls-back."""
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
                handle._state.cache.reset_for_new_generation(num_steps=ctx.active_num_steps)
                ctx.consumed_at_token = ctx.token
                context_consumed = True

            timestep_val = float(timestep.flatten()[0]) if hasattr(timestep, "flatten") else float(timestep)

            # 2. CFG fallback path (no skip-window validation here — all-CFG
            #    generations should not raise InvalidStepWindowError).
            cfg_active = negative_prompt_embeds is not None and negative_text_ids is not None
            if cfg_active:
                noise = _vanilla_flux2_cfg_predict(
                    transformer,
                    latents,
                    latent_ids,
                    prompt_embeds,
                    text_ids,
                    negative_prompt_embeds,
                    negative_text_ids,  # type: ignore[arg-type]
                    guidance,
                    timestep,
                )
                handle._state.stats.record(
                    StepDecision(
                        step_idx=handle._state.cache.step_counter,
                        timestep=timestep_val,
                        rel_l1=None,
                        accumulated_distance=handle._state.cache.accumulated_distance,
                        decision="cfg-fallback",
                    )
                )
                handle._state.cache.step_counter += 1
                handle._state.cache.last_timestep = timestep_val
                return noise

            # 3. Gated non-CFG path. Validate skip-window lazily here (first
            #    non-CFG call of the generation), per §5.6.
            if not handle._state.cache.skip_window_validated:
                if handle.skip_first_n_steps + handle.skip_last_n_steps >= ctx.active_num_steps:
                    raise InvalidStepWindowError(
                        skip_first=handle.skip_first_n_steps,
                        skip_last=handle.skip_last_n_steps,
                        num_steps=ctx.active_num_steps,
                    )
                handle._state.cache.skip_window_validated = True

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
