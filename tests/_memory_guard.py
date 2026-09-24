"""Test-session MLX memory-cap computation and installation."""

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
