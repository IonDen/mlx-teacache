"""Skip-streak telemetry shared by the bench scripts.

Both ``bench_speedup.py`` and ``bench_comparison.py`` record, per gated
generation, the per-step skip pattern and the longest run of consecutive skips
so ``docs/calibration.md``'s streak table can be filled from committed bench
reports. It also builds the report stamps (MLX version, code version, repo-relative
paths). Everything here is a pure function; ``mlx.core`` is imported only inside
``mlx_version``.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

_GIT_TIMEOUT_SECONDS = 5


def mlx_version() -> str:
    """Version of the ``mlx.core`` that is imported in this process."""
    import mlx.core as mx

    return str(mx.__version__)


def teacache_version(
    repo_root: Path,
    *,
    fallback: str,
    run: Callable[..., Any] = subprocess.run,
) -> str:
    """``git describe --tags --dirty --always`` for the checkout at ``repo_root``.

    An editable install freezes its dist version at ``uv sync`` time, so the dist
    version can name an older release than the code that ran. Any git failure
    (no git, not a checkout, timeout, empty output) returns ``fallback``.
    """
    argv = ["git", "-C", str(repo_root), "describe", "--tags", "--dirty", "--always"]
    try:
        out = run(argv, capture_output=True, text=True, check=True, timeout=_GIT_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return fallback
    described = str(out.stdout).strip()
    return described or fallback


def repo_relative(path: Path, repo_root: Path) -> str:
    """``path`` relative to ``repo_root`` (POSIX form), or just its name when it lies outside."""
    try:
        return Path(path).resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return Path(path).name


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
