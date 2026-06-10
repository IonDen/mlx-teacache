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


def test_kernel_has_no_mflux_import_nodes_ast():
    """Catches function-local `from mflux import X` that importlib checks miss.
    Immune to docstring mentions (AST inspects import nodes only)."""
    import ast
    from pathlib import Path

    kernel = Path(__file__).resolve().parent.parent.parent / "src" / "mlx_teacache" / "_kernel"
    assert kernel.is_dir(), f"kernel dir not found at {kernel}"
    offenders = []
    for py in kernel.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("mflux"):
                offenders.append(f"{py.name}:{node.lineno} from {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("mflux"):
                        offenders.append(f"{py.name}:{node.lineno} import {alias.name}")
    assert not offenders, f"mflux imports in _kernel/: {offenders}"
