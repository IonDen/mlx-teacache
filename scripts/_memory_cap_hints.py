"""Per-variant soft memory caps (GB) for the heavy scripts.

A hint is the advisory soft cap a worker passes to ``_mlx_caps.install_caps``
before the model loads; the worker derives its hard wired cap as (hint - 2) GB,
and ``install_caps`` clamps that to the device. It is a cap request sized on a
32 GB M1 Max, not a prediction of the peak. A variant without a hint uses the
script's own default (22 GB) or ``--cap-gb`` where the script has one.

Keyed by registry variant id. This module imports nothing.
"""

MEMORY_CAP_HINT_GB: dict[str, int] = {
    "flux2-klein-base-9b": 24,
    "z-image-base": 22,
    "qwen-image": 22,
    "qwen-image-2512": 22,
}


def memory_cap_hint_gb(variant_id: str) -> int | None:
    """Return the soft memory cap hint in GB for ``variant_id``, or None if it has none."""
    return MEMORY_CAP_HINT_GB.get(variant_id)
