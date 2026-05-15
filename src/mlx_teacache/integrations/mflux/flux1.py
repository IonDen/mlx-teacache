# src/mlx_teacache/integrations/mflux/flux1.py
"""Per-instance proxy that intercepts flux.transformer(...) calls and runs
TeaCache-gated forward (see forward.flux1_forward_with_gate).

Inherits nn.Module so mflux's component-level operations (Flux1.freeze ->
self.transformer.freeze(), ModelSaver -> component.parameters()) keep working.
Parent-level traversal (flux.parameters() at the Flux1 level) does NOT
recurse into _inner because MLX's valid_parameter_filter excludes
underscore-prefixed keys — this is documented as a v0.1 limitation."""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

from mlx_teacache.integrations.mflux.forward import flux1_forward_with_gate


class ProxyFlux1Transformer(nn.Module):  # type: ignore[misc,name-defined]
    def __init__(self, inner: Any, handle: Any) -> None:
        super().__init__()
        # Store inner under a leading-underscore name. This makes _inner
        # accessible via attribute lookup and our delegated methods, but
        # MLX's valid_parameter_filter excludes underscore-prefixed keys,
        # so parent-level flux.parameters() does NOT recurse into _inner.
        # We rely on mflux calling flux.transformer.parameters() (the
        # proxy's overridden method) directly during save, which delegates
        # to inner.parameters() correctly. See limitations note below.
        # Use object.__setattr__ to bypass nn.Module.__setattr__, which calls
        # hasattr(self, key) -> __getattr__ -> getattr(self._inner, ...) before
        # _inner is set, causing infinite recursion. Both _inner and _handle are
        # stored directly on the instance __dict__ this way.
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_handle", handle)

    def __call__(
        self,
        t: int,
        config: Any,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        **kwargs: Any,
    ) -> Any:
        return flux1_forward_with_gate(
            self._inner,
            self._handle,
            t=t,
            config=config,
            hidden_states=hidden_states,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            **kwargs,
        )

    # --- explicit method delegation for mflux-used ops ---
    # MLX's nn.Module exposes parameters() and trainable_parameters() as
    # methods (verified against mlx/nn/layers/base.py), not properties.

    def freeze(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.freeze(*args, **kwargs)

    def parameters(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._inner.parameters())

    def trainable_parameters(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._inner.trainable_parameters())

    def __getattr__(self, name: str) -> Any:
        # nn.Module's __getattr__ handles dict children + parameters. Fall
        # back to the inner module for anything else (x_embedder,
        # transformer_blocks, time_text_embed, etc.).
        # Use object.__getattribute__ to fetch _inner to avoid re-entering
        # __getattr__ if _inner hasn't been set yet.
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = object.__getattribute__(self, "_inner")
            return getattr(inner, name)
