"""Test-session MLX memory-cap computation and installation, plus the parity-lane
watchdog and wall backstop."""

import contextlib
import json
import math
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


def cache_limit_target(wired_bytes: int, *, cap_bytes: int = CACHE_CAP_BYTES) -> int:
    """Cache-pool cap: a quarter of the wired cap, at most ``cap_bytes`` (2 GiB by
    default, the policy scripts/_mlx_caps.py applies). MLX parks freed buffers in this
    pool and its default limit is near device memory, so an unbounded pool lets a
    module of many generations grow far past one generation's peak."""
    return min(cap_bytes, int(wired_bytes * CACHE_FRACTION_OF_WIRED))


def fallback_cache_limit_target(total_bytes: int, *, cap_bytes: int = CACHE_CAP_BYTES) -> int:
    """Cache-pool cap when no wired cap can be derived (a device report without
    max_recommended_working_set_size): 5 % of physical memory, at most ``cap_bytes``.
    Without it the pool would stay at MLX's near-device-memory default for the whole session."""
    return min(cap_bytes, int(total_bytes * CACHE_FRACTION_OF_MEMORY))


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


def apply_mlx_memory_caps(mx, emit, *, cache_cap_bytes: int = CACHE_CAP_BYTES) -> None:
    """Install independent hard and soft MLX caps without propagating errors.
    ``cache_cap_bytes`` bounds the cache pool (2 GiB for the fast lane; the parity
    guard re-applies the caps with 1 GiB)."""
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
            cache = fallback_cache_limit_target(total_bytes, cap_bytes=cache_cap_bytes)
    else:
        try:
            mx.set_wired_limit(target)
        except Exception as exc:  # noqa: BLE001
            emit(f"set_wired_limit({target}) failed ({exc!r})")
        cache = cache_limit_target(target, cap_bytes=cache_cap_bytes)
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
PARITY_HEADROOM_ENV = "PYTEST_PARITY_HEADROOM_GIB"
DEFAULT_PARITY_HEADROOM_GIB = 4.0  # same default as scripts/_mlx_watchdog.DEFAULT_HEADROOM_GIB
# Qwen parity runs at ~26 GiB active; a 1 GiB pool keeps it clear of the 28 GiB
# active+cache ceiling (the fast lane keeps CACHE_CAP_BYTES).
PARITY_CACHE_CAP_BYTES = 1 * GIB
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


def headroom_gib(environ: Mapping[str, str]) -> float:
    """Watchdog headroom in GiB: ``PYTEST_PARITY_HEADROOM_GIB`` if a positive finite
    number, else 4. The ceiling is physical memory minus this headroom."""
    try:
        value = float(environ.get(PARITY_HEADROOM_ENV, ""))
    except ValueError:
        return DEFAULT_PARITY_HEADROOM_GIB
    return value if math.isfinite(value) and value > 0 else DEFAULT_PARITY_HEADROOM_GIB


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


def _default_start_memory(
    on_abort: Callable[[dict[str, int]], None], *, headroom_gib: float
) -> threading.Thread:
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    import _mlx_watchdog

    return _mlx_watchdog.arm_mlx_watchdog(on_abort=on_abort, headroom_gib=headroom_gib)


def _default_install_caps(cache_cap_bytes: int) -> None:
    """Re-apply the session caps with the parity lane's smaller cache pool."""
    try:
        import mlx.core as mx
    except ImportError:
        return
    apply_mlx_memory_caps(
        mx,
        lambda message: print(f"mlx-teacache tests: {message}", file=sys.stderr),
        cache_cap_bytes=cache_cap_bytes,
    )


def _abort_line_writer(capture: Any, stderr_fd: int | None) -> Callable[[str], None]:
    """How an abort handler prints its one line so it reaches the terminal.

    The handlers run just before ``os._exit``, which skips pytest's capture teardown,
    so a line printed into the fd capture is lost. With pytest's capture manager the
    writer suspends the global capture first; without it (``-p no:capture``) nothing
    is captured, so the line goes to a descriptor duplicated from stderr at arm time."""
    if capture is not None:

        def _emit(line: str) -> None:
            with contextlib.suppress(Exception):  # print anyway; the exit follows
                capture.suspend_global_capture(in_=True)
            print(line, file=sys.stderr, flush=True)

        return _emit
    fd = os.dup(2) if stderr_fd is None else stderr_fd

    def _write(line: str) -> None:
        os.write(fd, f"{line}\n".encode())

    return _write


def arm_parity_guard(
    items: Sequence[Any],
    *,
    start_memory: Callable[..., Any] = _default_start_memory,
    start_wall: Callable[..., Any] = start_wall_backstop,
    environ: Mapping[str, str] | None = None,
    exit_fn: Callable[[int], None] = os._exit,
    artifact_dir: Path | None = None,
    install_caps: Callable[[int], Any] = _default_install_caps,
    capture: Any = None,
    stderr_fd: int | None = None,
) -> bool:
    """Arm the active+cache memory watchdog and the wall backstop when ``items``
    include a parity test. Returns whether it armed. Arming removes a stale
    ``pytest-parity.aborted.json``, lowers the MLX cache pool to 1 GiB and reads the
    ``PYTEST_PARITY_HEADROOM_GIB`` / ``PYTEST_PARITY_WALL_S`` overrides. A trip prints
    one line past pytest's output capture (``capture`` is pytest's capture manager),
    writes ``<artifact_dir>/pytest-parity.aborted.json`` (reason ``memory`` or
    ``wall``) and exits with WATCHDOG_EXIT_CODE."""
    global PARITY_GUARD_ARMED
    if not parity_selected(items):
        return False
    env = os.environ if environ is None else environ
    target = (
        artifact_dir
        if artifact_dir is not None
        else _SCRIPTS_DIR.parent / "tests" / "_artifacts" / "watchdog_aborts"
    )
    (target / f"{PARITY_LABEL}.aborted.json").unlink(missing_ok=True)
    install_caps(PARITY_CACHE_CAP_BYTES)
    emit = _abort_line_writer(capture, stderr_fd)

    def _write(payload: dict[str, Any]) -> None:
        target.mkdir(parents=True, exist_ok=True)
        record = {"label": PARITY_LABEL, "at": datetime.now(timezone.utc).isoformat(), **payload}
        (target / f"{PARITY_LABEL}.aborted.json").write_text(json.dumps(record, indent=2))

    def _on_memory_abort(payload: dict[str, int]) -> None:
        line = (
            f"[watchdog] ABORTED {PARITY_LABEL}: {payload['resident_bytes'] / GIB:.2f} GiB resident "
            f"> {payload['ceiling_bytes'] / GIB:.2f} GiB ceiling"
        )
        emit(line)
        _write({"reason": "memory", **payload})

    def _on_wall_abort(payload: dict[str, Any]) -> None:
        emit(f"[watchdog] ABORTED {PARITY_LABEL}: wall backstop {payload['wall_s']} s")
        _write({"reason": "wall", **payload})

    start_memory(_on_memory_abort, headroom_gib=headroom_gib(env))
    start_wall(
        wall_s=wall_seconds(env),
        on_abort=_on_wall_abort,
        exit_fn=exit_fn,
    )
    PARITY_GUARD_ARMED = True
    return True
