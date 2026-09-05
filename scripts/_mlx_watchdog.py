"""Active+cache memory watchdog for heavy MLX workers.

MLX's soft limit is advisory and the wired cap bounds only non-pageable memory,
so a worker can walk past the device's working set into a paging storm with
nothing stopping it. This thread samples ``get_active_memory + get_cache_memory``
(the actual resident footprint: dropped buffers sit in the cache pool, which
``get_active_memory`` alone does not count) every ``poll_s`` and aborts the
process the moment the sum exceeds ``memory_size - headroom``. The poll thread
runs while ``mx.eval`` holds no GIL, so 0.05 s costs nothing measurable.

The abort is ``os._exit(3)`` after ``on_abort(payload)``: the caller prints an
honest artifact (the orchestrator persists it as ``*.aborted.json``) and the
process dies before the kernel does.
"""

import os
import threading
from collections.abc import Callable

GIB = 1024**3
DEFAULT_HEADROOM_GIB = 4.0
ABORT_EXIT_CODE = 3


def ceiling_bytes(memory_size_bytes: int, *, headroom_gib: float = DEFAULT_HEADROOM_GIB) -> int:
    """Resident-memory ceiling: physical memory minus the headroom the OS needs."""
    ceiling = int(memory_size_bytes - headroom_gib * GIB)
    if ceiling <= 0:
        raise ValueError(f"memory_size {memory_size_bytes} leaves no room under {headroom_gib} GiB headroom")
    return ceiling


def over_ceiling(active_bytes: int, cache_bytes: int, ceiling: int) -> bool:
    """True when the resident footprint (active + retained cache) exceeds the ceiling."""
    return active_bytes + cache_bytes > ceiling


def start_watchdog(
    *,
    ceiling: int,
    sample: Callable[[], tuple[int, int]],
    on_abort: Callable[[dict[str, int]], None],
    exit_fn: Callable[[int], None] = os._exit,
    poll_s: float = 0.05,
    stop: threading.Event | None = None,
) -> threading.Thread:
    """Start the daemon poll thread; return it. ``sample`` yields ``(active, cache)``
    bytes; ``on_abort`` receives the payload once, then ``exit_fn(3)`` runs."""
    halt = stop if stop is not None else threading.Event()

    def _loop() -> None:
        while not halt.is_set():
            active, cached = sample()
            if over_ceiling(active, cached, ceiling):
                on_abort(
                    {
                        "active_bytes": active,
                        "cache_bytes": cached,
                        "resident_bytes": active + cached,
                        "ceiling_bytes": ceiling,
                    }
                )
                exit_fn(ABORT_EXIT_CODE)
                return
            halt.wait(poll_s)

    thread = threading.Thread(target=_loop, name="mlx-memory-watchdog", daemon=True)
    thread.start()
    return thread


def arm_mlx_watchdog(
    *, on_abort: Callable[[dict[str, int]], None], headroom_gib: float = DEFAULT_HEADROOM_GIB
) -> threading.Thread:
    """Imperative glue: read the device size and sample the real MLX counters."""
    import mlx.core as mx

    ceiling = ceiling_bytes(int(mx.device_info()["memory_size"]), headroom_gib=headroom_gib)
    return start_watchdog(
        ceiling=ceiling,
        sample=lambda: (int(mx.get_active_memory()), int(mx.get_cache_memory())),
        on_abort=on_abort,
    )
