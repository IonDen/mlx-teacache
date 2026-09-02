"""Device-derived MLX memory caps shared by the heavy scripts.

``mx.set_wired_limit`` raises when asked for more than the system's wired
limit, so a literal cap tuned on a 32 GB M1 Max (20-24 GB) fails at startup on
a 16 GB or 24 GB Mac. Every model-loading worker should derive its wired cap
through :func:`clamped_wired_bytes` (pure, unit-tested) or call
:func:`install_caps`, which reads the device ceiling and applies both limits.
"""

GIB = 1024**3
DEFAULT_FRACTION = 0.85  # of max_recommended_working_set_size — same margin scripts/calibrate_qwen.py uses


def clamped_wired_bytes(
    requested_gb: float, max_recommended_bytes: int, *, fraction: float = DEFAULT_FRACTION
) -> int:
    """Wired-limit bytes: the requested cap, clamped to ``fraction`` of the device's
    recommended working set so it can never exceed the system wired limit."""
    if requested_gb <= 0:
        raise ValueError(f"requested wired cap must be positive, got {requested_gb} GB")
    return min(int(requested_gb * GIB), int(max_recommended_bytes * fraction))


def install_caps(*, wired_gb: float, soft_gb: float) -> tuple[int, int]:
    """Apply a device-clamped wired cap and the soft cap; return (wired_bytes, soft_bytes).

    Call BEFORE any model load. The soft cap is advisory (``mx.set_memory_limit``);
    it is never set below the wired cap."""
    import mlx.core as mx

    max_set = int(mx.device_info()["max_recommended_working_set_size"])
    wired = clamped_wired_bytes(wired_gb, max_set)
    soft = max(int(soft_gb * GIB), wired)
    mx.set_wired_limit(wired)
    mx.set_memory_limit(soft)
    if wired < int(wired_gb * GIB):
        print(
            f"[caps] wired cap clamped from {wired_gb:g} GB to {wired / GIB:.2f} GB "
            f"({DEFAULT_FRACTION:.0%} of the {max_set / GIB:.2f} GB recommended working set)",
            flush=True,
        )
    return wired, soft
