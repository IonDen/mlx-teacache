"""Test-session MLX memory-cap computation and installation."""


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
    if target is None:
        emit(f"no safe wired cap (memory_size={total_bytes}, max_working_set={max_working_set})")
    else:
        try:
            mx.set_wired_limit(target)
        except Exception as exc:  # noqa: BLE001
            emit(f"set_wired_limit({target}) failed ({exc!r})")

    if total_bytes > 0:
        try:
            mx.set_memory_limit(int(total_bytes * 0.70))
        except Exception as exc:  # noqa: BLE001
            emit(f"set_memory_limit failed ({exc!r})")
