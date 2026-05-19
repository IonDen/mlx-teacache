"""Pure-algorithm primitives have no business pulling mflux. Walks every
module under _kernel/ in a simulated no-mflux environment and confirms
they import cleanly."""

from __future__ import annotations

import importlib
import pkgutil


def test_kernel_subtree_imports_without_mflux(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "mflux", None)

    import mlx_teacache._kernel as kernel_pkg

    importlib.reload(kernel_pkg)
    for _, name, _ in pkgutil.walk_packages(kernel_pkg.__path__, kernel_pkg.__name__ + "."):
        importlib.import_module(name)
