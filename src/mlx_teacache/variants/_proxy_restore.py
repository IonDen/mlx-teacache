"""mflux-free rollback shared by the proxy variants (FLUX.1 dev / Krea dev / schnell, Qwen-Image)."""

import weakref
from collections.abc import Callable
from typing import Any


def make_restore_transformer(flux: Any, proxy: Any) -> Callable[[], None]:
    """A rollback that puts the proxy's inner transformer back on `flux` while the proxy is still installed.

    It holds the proxy only weakly, so the proxy in `flux.transformer` is the only strong reference to it and, through
    it, to the original transformer. When mflux's memory saver sets `model.transformer = None` before the decode
    (--low-ram or --pid-decode, for a single seed), both are collected. The slot is restored only while it still holds
    the proxy, so the rollback never brings back a transformer mflux dropped, even if something else still references
    the proxy, and never replaces a transformer the caller installed after apply."""
    proxy_ref = weakref.ref(proxy)

    def _restore_transformer() -> None:
        live = proxy_ref()
        if live is None:
            return
        installed: Any = getattr(flux, "transformer", None)
        if installed is live:
            flux.transformer = live._inner

    return _restore_transformer
