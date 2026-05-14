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
        - FLUX.1 wrapper on every t == 0 transformer call (§5.2)
        - FLUX.2 predict closure on the first call of a new generation (§5.5)
        - api.restore_fn via discard path
        """
        self.step_counter = 0
        self.previous_mod_input = None
        self.cached_residual = None
        self.accumulated_distance = 0.0
        self.last_timestep = None
        self.skip_window_validated = False
        self.num_steps = num_steps
