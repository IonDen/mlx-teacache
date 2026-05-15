# src/mlx_teacache/stats.py
"""Public stats with a private staging buffer.

Public counters (computed_count, forced_count, skipped_count, numerical_miss_count,
cfg_fallback_steps, generations, last_generation) reflect only committed
generations. record() updates a staging buffer; finalize_last_generation()
commits + snapshots; discard_current_generation() zeroes the staging buffer
(called by the generate_image try/finally when a run does not complete naturally,
per spec §4.5 v2.5). This makes "failed runs leave no trace in public stats"
mechanical rather than aspirational."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


class StatsFrozenError(Exception):
    """Raised when attempting to mutate a frozen TeaCacheStats. Frozen by handle.restore()."""


Decision = Literal["computed", "forced", "skipped", "numerical-miss", "cfg-fallback"]


@dataclass(frozen=True)
class StepDecision:
    step_idx: int
    timestep: float
    rel_l1: float | None
    accumulated_distance: float
    decision: Decision


@dataclass(frozen=True)
class GenerationStats:
    num_steps: int
    cfg_was_active: bool
    decisions: tuple[StepDecision, ...]


@dataclass
class _Staging:
    computed: int = 0
    forced: int = 0
    skipped: int = 0
    numerical_miss: int = 0
    cfg_fallback: int = 0
    decisions: list[StepDecision] = field(default_factory=list)

    def clear(self) -> None:
        self.computed = 0
        self.forced = 0
        self.skipped = 0
        self.numerical_miss = 0
        self.cfg_fallback = 0
        self.decisions.clear()


@dataclass
class TeaCacheStats:
    """Public aggregate counters (committed). Failed/interrupted generations
    are NOT included in these counters. See `_staging` for the staging buffer
    that holds in-progress step records until commit-or-discard."""

    generations: int = 0
    computed_count: int = 0
    forced_count: int = 0
    skipped_count: int = 0
    numerical_miss_count: int = 0
    cfg_fallback_steps: int = 0
    last_generation: GenerationStats | None = None
    _staging: _Staging = field(default_factory=_Staging)
    _frozen: bool = False

    @property
    def total_active_steps(self) -> int:
        """Committed steps that ran under TeaCache (gated or forced). Excludes
        CFG fallback steps where mlx-teacache ran vanilla mflux."""
        return self.computed_count + self.forced_count + self.skipped_count + self.numerical_miss_count

    @property
    def total_steps_seen(self) -> int:
        """All committed steps across all generations, including CFG fallbacks."""
        return self.total_active_steps + self.cfg_fallback_steps

    @property
    def speedup_estimate(self) -> float:
        """Coarse estimate over TeaCache-active steps only:
            active_steps / (active_steps - skipped_count)
        Wall-clock will be slightly less due to ~1-2% gating overhead.

        Returns 1.0 if no active steps yet OR if every step was CFG fallback
        (in which case TeaCache did not actually run; vanilla mflux did).
        Does NOT account for CFG fallback steps."""
        active = self.total_active_steps
        if active == 0:
            return 1.0
        denom = active - self.skipped_count
        if denom <= 0:
            return 1.0
        return active / denom

    def record(self, decision: StepDecision) -> None:
        """Append a step decision to the staging buffer. No public counter changes."""
        if self._frozen:
            raise StatsFrozenError("TeaCacheStats is frozen (handle.restore() was called)")
        st = self._staging
        st.decisions.append(decision)
        if decision.decision == "computed":
            st.computed += 1
        elif decision.decision == "forced":
            st.forced += 1
        elif decision.decision == "skipped":
            st.skipped += 1
        elif decision.decision == "numerical-miss":
            st.numerical_miss += 1
        elif decision.decision == "cfg-fallback":
            st.cfg_fallback += 1

    def finalize_last_generation(self, *, num_inference_steps: int, cfg_was_active: bool) -> None:
        """Commit staging counters to public counters and snapshot GenerationStats.
        Called by the generate_image wrapper on natural completion only.

        Enforces spec invariant: len(GenerationStats.decisions) == num_inference_steps.
        A mismatch raises InternalStateError WITHOUT committing — the staging is
        discarded so partial commits never leak."""
        if self._frozen:
            raise StatsFrozenError("TeaCacheStats is frozen (handle.restore() was called)")
        from mlx_teacache.errors import InternalStateError

        st = self._staging
        if len(st.decisions) != num_inference_steps:
            actual = len(st.decisions)
            st.clear()
            raise InternalStateError(
                f"GenerationStats length invariant violated: expected {num_inference_steps} "
                f"step decisions, got {actual}. Staging discarded; public counters unchanged. "
                f"This indicates a bug in the integration layer's step bookkeeping."
            )
        self.computed_count += st.computed
        self.forced_count += st.forced
        self.skipped_count += st.skipped
        self.numerical_miss_count += st.numerical_miss
        self.cfg_fallback_steps += st.cfg_fallback
        self.last_generation = GenerationStats(
            num_steps=num_inference_steps,
            cfg_was_active=cfg_was_active,
            decisions=tuple(st.decisions),
        )
        self.generations += 1
        st.clear()

    def discard_current_generation(self) -> None:
        """Zero the staging buffer without committing. Called by the
        generate_image try/finally when the run did not complete naturally."""
        if self._frozen:
            raise StatsFrozenError("TeaCacheStats is frozen (handle.restore() was called)")
        self._staging.clear()

    def _freeze(self) -> None:
        self._frozen = True
