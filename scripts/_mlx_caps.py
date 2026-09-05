"""Device-derived MLX memory caps shared by the heavy scripts.

Three limits, three different jobs:

* ``mx.set_wired_limit`` is the only hard ceiling. It bounds non-pageable Metal
  allocations and raises when asked for more than the system wired limit, so a
  literal cap tuned on a 32 GB M1 Max fails at startup on a 16 GB or 24 GB Mac.
  Derive it through :func:`clamped_wired_bytes`.
* ``mx.set_memory_limit`` is advisory. It does not stop an allocation; pageable
  memory past it is paged, and a sustained paging storm has kernel-panicked this
  machine. Treat it as a guideline, never as a ceiling.
* ``mx.set_cache_limit`` bounds MLX's retained buffer pool. Dropped buffers go
  there instead of back to the OS, and the default sits near physical RAM
  (30.4 GiB measured on a 32 GB M1 Max), so without this cap the resident
  footprint climbs unopposed. Derive it through :func:`clamped_cache_bytes`.

None of the three aborts a run that is already past the working set. That is
the job of the active+cache watchdog in ``_mlx_watchdog.py``; every script
under ``scripts/`` that installs these caps also arms the watchdog (a test in
``tests/test_scripts_memory_caps.py`` keeps it that way).
"""

GIB = 1024**3
DEFAULT_FRACTION = 0.85  # of max_recommended_working_set_size — same margin scripts/calibrate_qwen.py uses
DEFAULT_CACHE_GB = 2.0
CACHE_FRACTION_OF_WIRED = 0.25


def clamped_wired_bytes(
    requested_gb: float, max_recommended_bytes: int, *, fraction: float = DEFAULT_FRACTION
) -> int:
    """Wired-limit bytes: the requested cap, clamped to ``fraction`` of the device's
    recommended working set so it can never exceed the system wired limit."""
    if requested_gb <= 0:
        raise ValueError(f"requested wired cap must be positive, got {requested_gb} GB")
    return min(int(requested_gb * GIB), int(max_recommended_bytes * fraction))


def clamped_cache_bytes(
    cache_gb: float, wired_bytes: int, *, fraction: float = CACHE_FRACTION_OF_WIRED
) -> int:
    """Cache-pool bytes: the requested size, clamped to ``fraction`` of the wired
    cap so the pool can never dominate the budget the wired cap was sized for."""
    if cache_gb <= 0:
        raise ValueError(f"requested cache cap must be positive, got {cache_gb} GB")
    return min(int(cache_gb * GIB), int(wired_bytes * fraction))


def install_caps(
    *, wired_gb: float, soft_gb: float, cache_gb: float = DEFAULT_CACHE_GB
) -> tuple[int, int, int]:
    """Apply the device-clamped wired cap, the advisory soft cap, and the
    cache-pool cap; return ``(wired_bytes, soft_bytes, cache_bytes)``.

    Call BEFORE any model load. The soft cap is never set below the wired cap."""
    import mlx.core as mx

    max_set = int(mx.device_info()["max_recommended_working_set_size"])
    wired = clamped_wired_bytes(wired_gb, max_set)
    soft = max(int(soft_gb * GIB), wired)
    cache = clamped_cache_bytes(cache_gb, wired)
    mx.set_wired_limit(wired)
    mx.set_memory_limit(soft)
    mx.set_cache_limit(cache)
    if wired < int(wired_gb * GIB):
        print(
            f"[caps] wired cap clamped from {wired_gb:g} GB to {wired / GIB:.2f} GB "
            f"({DEFAULT_FRACTION:.0%} of the {max_set / GIB:.2f} GB recommended working set)",
            flush=True,
        )
    return wired, soft, cache
