# src/mlx_teacache/_kernel/cache.py
"""Canonical home for the TeaCacheState dataclass (extracted in v0.6.0).

Per-handle mutable cache state. Pure data; methods only for reset.

Fields
------
step_counter          : int
previous_mod_input    : mx.array | None
cached_residual       : mx.array | None  — positive branch residual
cached_residual_neg   : mx.array | None  — negative branch residual (CFG)
accumulated_distance  : float
last_timestep         : float | None
skip_window_validated : bool
num_steps             : int | None
consecutive_skips     : int
"""

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class TeaCacheState:
    step_counter: int = 0
    previous_mod_input: mx.array | None = None
    cached_residual: mx.array | None = None
    cached_residual_neg: mx.array | None = None
    accumulated_distance: float = 0.0
    last_timestep: float | None = None
    skip_window_validated: bool = False
    num_steps: int | None = None
    consecutive_skips: int = 0

    def reset_for_new_generation(self, *, num_steps: int) -> None:
        """Clear all per-generation fields. Called by:
        - ``GenerationContextCallback.call_before_loop`` (in
          ``integrations/mflux/lifecycle.py``) — the sole owner of per-generation
          cache reset as of v0.2.0. Previously this was called from FLUX.1's
          forward (`if t == 0:`) and FLUX.2's predict closure (`if not
          context_consumed:`); both call sites were removed when lifecycle took
          exclusive ownership so img2img generations (which start at `t > 0`)
          reset correctly.

        Clears both ``cached_residual`` (positive branch) and
        ``cached_residual_neg`` (negative branch) to prevent cross-generation
        pollution under CFG (v0.4.1+).
        """
        self.step_counter = 0
        self.previous_mod_input = None
        self.cached_residual = None
        self.cached_residual_neg = None
        self.accumulated_distance = 0.0
        self.consecutive_skips = 0
        self.last_timestep = None
        self.skip_window_validated = False
        self.num_steps = num_steps

    def store_residuals(self, *, pos: mx.array | None = None, neg: mx.array | None = None) -> None:
        """Materialise and keep the body residual(s).

        ``body_out - body_in`` is a sibling of the step's output, not an
        ancestor, so mflux's per-step ``mx.eval(latents)`` never forces it. Left
        lazy it pins both operands across the step boundary (twice its own size,
        four times under CFG). Evaluate here, once, at the only place the
        residual is written."""
        arrays = [a for a in (pos, neg) if a is not None]
        if arrays:
            mx.eval(*arrays)
        if pos is not None:
            self.cached_residual = pos
        if neg is not None:
            self.cached_residual_neg = neg

    def release_arrays(self) -> None:
        """Drop the three body-sized arrays without touching the counters.

        Called after the denoising loop (before VAE decode, where peak memory
        lands) and on ``handle.restore()``, so a handle kept for its stats does
        not pin them for the process lifetime. Dropping the references hands the
        buffers back to MLX's allocator pool, where the decode can reuse them;
        whether the pool itself shrinks is governed by ``mx.set_cache_limit``,
        which the library leaves to the application."""
        self.previous_mod_input = None
        self.cached_residual = None
        self.cached_residual_neg = None
