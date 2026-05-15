# src/mlx_teacache/cache.py
"""Per-handle mutable cache state. Pure data; methods only for reset."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class TeaCacheState:
    step_counter: int = 0
    previous_mod_input: mx.array | None = None
    cached_residual: mx.array | None = None
    accumulated_distance: float = 0.0
    last_timestep: float | None = None
    skip_window_validated: bool = False
    num_steps: int | None = None

    def reset_for_new_generation(self, *, num_steps: int) -> None:
        """Clear all per-generation fields. Called by:
        - ``GenerationContextCallback.call_before_loop`` (in
          ``integrations/mflux/lifecycle.py``) — the sole owner of per-generation
          cache reset as of v0.2.0. Previously this was called from FLUX.1's
          forward (`if t == 0:`) and FLUX.2's predict closure (`if not
          context_consumed:`); both call sites were removed when lifecycle took
          exclusive ownership so img2img generations (which start at `t > 0`)
          reset correctly.
        """
        self.step_counter = 0
        self.previous_mod_input = None
        self.cached_residual = None
        self.accumulated_distance = 0.0
        self.last_timestep = None
        self.skip_window_validated = False
        self.num_steps = num_steps
