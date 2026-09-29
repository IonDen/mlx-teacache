"""Test-session MLX memory-cap computation and installation, plus the parity-lane
watchdog and wall backstop."""

import json
import os
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GIB = 1024**3
CACHE_CAP_BYTES = 2 * GIB
CACHE_FRACTION_OF_WIRED = 0.25
CACHE_FRACTION_OF_MEMORY = 0.05


def cache_limit_target(wired_bytes: int) -> int:
    """Cache-pool cap: a quarter of the wired cap, at most 2 GiB (the policy
    scripts/_mlx_caps.py applies). MLX parks freed buffers in this pool and its
    default limit is near device memory, so an unbounded pool lets a module of
    many generations grow far past one generation's peak."""
    return min(CACHE_CAP_BYTES, int(wired_bytes * CACHE_FRACTION_OF_WIRED))


def fallback_cache_limit_target(total_bytes: int) -> int:
    """Cache-pool cap when no wired cap can be derived (a device report without
    max_recommended_working_set_size): 5 % of physical memory, at most 2 GiB. Without it
    the pool would stay at MLX's near-device-memory default for the whole session."""
    return min(CACHE_CAP_BYTES, int(total_bytes * CACHE_FRACTION_OF_MEMORY))


def wired_limit_target(total_bytes: int, max_working_set: int) -> int | None:
    """Return a positive wired-memory cap strictly below the MLX ceiling."""
    if total_bytes <= 0 or max_working_set <= 0:
        return None
    desired = int(total_bytes * 0.60)
    ceiling = int(max_working_set * 0.90)
    target = min(desired, ceiling)
    if target <= 0 or target >= max_working_set:
        return None
    return target


def apply_mlx_memory_caps(mx, emit) -> None:
    """Install independent hard and soft MLX caps without propagating errors."""
    try:
        info = mx.device_info()
        total_bytes = int(info.get("memory_size", 0))
        max_working_set = int(info.get("max_recommended_working_set_size", 0))
    except Exception as exc:  # noqa: BLE001
        emit(f"device_info failed ({exc!r})")
        return

    target = wired_limit_target(total_bytes, max_working_set)
    cache: int | None = None
    if target is None:
        emit(f"no safe wired cap (memory_size={total_bytes}, max_working_set={max_working_set})")
        if total_bytes > 0:
            cache = fallback_cache_limit_target(total_bytes)
    else:
        try:
            mx.set_wired_limit(target)
        except Exception as exc:  # noqa: BLE001
            emit(f"set_wired_limit({target}) failed ({exc!r})")
        cache = cache_limit_target(target)
    if cache is not None:
        try:
            mx.set_cache_limit(cache)
        except Exception as exc:  # noqa: BLE001
            emit(f"set_cache_limit({cache}) failed ({exc!r})")

    if total_bytes > 0:
        try:
            mx.set_memory_limit(int(total_bytes * 0.70))
        except Exception as exc:  # noqa: BLE001
            emit(f"set_memory_limit failed ({exc!r})")


PARITY_LABEL = "pytest-parity"
PARITY_WALL_ENV = "PYTEST_PARITY_WALL_S"
DEFAULT_PARITY_WALL_S = 10800.0  # 3 h per session
WATCHDOG_EXIT_CODE = 4  # same value as scripts/_mlx_watchdog.WATCHDOG_EXIT_CODE
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

PARITY_GUARD_ARMED = False


def parity_selected(items: Sequence[Any]) -> bool:
    """True when any collected item carries the ``parity`` marker."""
    return any(item.get_closest_marker("parity") is not None for item in items)


def wall_seconds(environ: Mapping[str, str]) -> float:
    """Wall backstop in seconds: ``PYTEST_PARITY_WALL_S`` if a positive number, else 3 h."""
    try:
        value = float(environ.get(PARITY_WALL_ENV, ""))
    except ValueError:
        return DEFAULT_PARITY_WALL_S
    return value if value > 0 else DEFAULT_PARITY_WALL_S


def start_wall_backstop(
    *,
    wall_s: float,
    on_abort: Callable[[dict[str, Any]], None],
    exit_fn: Callable[[int], None] = os._exit,
    stop: threading.Event | None = None,
) -> threading.Thread:
    """Daemon thread: after ``wall_s`` seconds (unless ``stop`` is set first) call
    ``on_abort`` then ``exit_fn(WATCHDOG_EXIT_CODE)``."""
    halt = stop if stop is not None else threading.Event()

    def _loop() -> None:
        if halt.wait(wall_s):
            return
        try:
            on_abort({"wall_s": wall_s})
        except BaseException:  # noqa: BLE001 - the exit must happen even if the handler fails
            import traceback

            traceback.print_exc(file=sys.stderr)
        exit_fn(WATCHDOG_EXIT_CODE)

    thread = threading.Thread(target=_loop, name="parity-wall-backstop", daemon=True)
    thread.start()
    return thread


def _default_start_memory(on_abort: Callable[[dict[str, int]], None]) -> threading.Thread:
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    import _mlx_watchdog

    return _mlx_watchdog.arm_mlx_watchdog(on_abort=on_abort)


def arm_parity_guard(
    items: Sequence[Any],
    *,
    start_memory: Callable[[Callable[[dict[str, int]], None]], Any] = _default_start_memory,
    start_wall: Callable[..., Any] = start_wall_backstop,
    environ: Mapping[str, str] | None = None,
    exit_fn: Callable[[int], None] = os._exit,
    artifact_dir: Path | None = None,
) -> bool:
    """Arm the active+cache memory watchdog and the wall backstop when ``items``
    include a parity test. Returns whether it armed. A trip writes
    ``<artifact_dir>/pytest-parity.aborted.json`` (reason ``memory`` or ``wall``)
    and exits with WATCHDOG_EXIT_CODE."""
    global PARITY_GUARD_ARMED
    if not parity_selected(items):
        return False
    target = (
        artifact_dir
        if artifact_dir is not None
        else _SCRIPTS_DIR.parent / "tests" / "_artifacts" / "watchdog_aborts"
    )

    def _write(payload: dict[str, Any]) -> None:
        target.mkdir(parents=True, exist_ok=True)
        record = {"label": PARITY_LABEL, "at": datetime.now(timezone.utc).isoformat(), **payload}
        (target / f"{PARITY_LABEL}.aborted.json").write_text(json.dumps(record, indent=2))

    def _on_memory_abort(payload: dict[str, int]) -> None:
        line = (
            f"[watchdog] ABORTED {PARITY_LABEL}: {payload['resident_bytes'] / GIB:.2f} GiB resident "
            f"> {payload['ceiling_bytes'] / GIB:.2f} GiB ceiling"
        )
        print(line, file=sys.stderr, flush=True)
        _write({"reason": "memory", **payload})

    def _on_wall_abort(payload: dict[str, Any]) -> None:
        print(
            f"[watchdog] ABORTED {PARITY_LABEL}: wall backstop {payload['wall_s']} s",
            file=sys.stderr,
            flush=True,
        )
        _write({"reason": "wall", **payload})

    start_memory(_on_memory_abort)
    start_wall(
        wall_s=wall_seconds(os.environ if environ is None else environ),
        on_abort=_on_wall_abort,
        exit_fn=exit_fn,
    )
    PARITY_GUARD_ARMED = True
    return True
