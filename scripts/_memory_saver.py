"""mflux MemorySaver builder shared by the Qwen bench, sweep and calibration scripts.

Standard library only: no PIL, scikit-image, scipy or mflux import here. The
caller passes mflux's MemorySaver class in, so a script can import this module at
load time (before any model is built) without dragging the image stack along.
"""

from typing import Any


def make_memory_saver(flux: Any, saver_cls: Any) -> Any:
    """Build mflux's MemorySaver with the load-bearing keyword arguments pinned.

    keep_transformer=True frees only the text encoders. cache_limit_bytes=None
    matters: MemorySaver's default (1 GB) branch calls mx.set_cache_limit,
    overriding install_caps, calls mx.reset_peak_memory before the load peak is
    read, and switches the VAE to tiled decode, which changes output pixels and
    would make the SSIM numbers and committed images incomparable. MemorySaver
    also runs gc.collect and mx.clear_cache after every loop; that applies to
    every condition alike, so the ratios stand."""
    return saver_cls(model=flux, keep_transformer=True, cache_limit_bytes=None, num_seeds=1)
