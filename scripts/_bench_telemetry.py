"""Skip-streak telemetry shared by the bench scripts.

Both ``bench_speedup.py`` and ``bench_comparison.py`` record, per gated
generation, the per-step skip pattern and the longest run of consecutive skips
so ``docs/calibration.md``'s streak table can be filled from committed bench
reports. Pure functions over ``TeaCacheStats``; no mlx / mflux imports.
"""

from typing import Any


def skip_pattern(decision_kinds: list[str]) -> str:
    """Per-step pattern string: ``S`` for a skipped step, ``C`` for everything else."""
    return "".join("S" if kind == "skipped" else "C" for kind in decision_kinds)


def max_skip_streak(pattern: str) -> int:
    """Length of the longest run of consecutive ``S`` in a skip pattern (0 if none)."""
    return max((len(run) for run in pattern.split("C")), default=0)


def streak_telemetry(stats: Any) -> dict[str, Any]:
    """Skip pattern + max consecutive-skip streak of the last committed generation."""
    last = stats.last_generation
    if last is None:
        return {"skip_pattern": "", "max_consecutive_skips": 0}
    pattern = skip_pattern([d.decision for d in last.decisions])
    return {"skip_pattern": pattern, "max_consecutive_skips": max_skip_streak(pattern)}
