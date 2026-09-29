"""Pure-algorithm primitives have no business pulling mflux. Walks every
module under _kernel/ in a simulated no-mflux environment and confirms
they import cleanly."""

from __future__ import annotations

import subprocess
import sys


def test_kernel_subtree_imports_without_mflux():
    """Bug caught: a top-level mflux import in _kernel/ or the variant registry.
    Runs in a fresh interpreter so the result cannot depend on which modules
    earlier tests already imported (sys.modules cache, test order)."""
    code = (
        "import sys; sys.modules['mflux'] = None; "
        "import mlx_teacache._kernel.gate, mlx_teacache._kernel.cache, "
        "mlx_teacache._kernel.stats, mlx_teacache._kernel.coefficients, "
        "mlx_teacache.variants"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _mflux_free_sources() -> list:
    from pathlib import Path

    pkg = Path(__file__).resolve().parent.parent.parent / "src" / "mlx_teacache"
    assert pkg.is_dir(), f"package dir not found at {pkg}"
    files = set(pkg.joinpath("_kernel").rglob("*.py"))
    files.add(pkg / "__init__.py")
    files.update(pkg.joinpath("variants").glob("*.py"))
    files.update(pkg.joinpath("variants").glob("*/config.py"))
    files.update(pkg.joinpath("variants").glob("*/detect.py"))
    return sorted(files)


def test_mflux_free_modules_have_no_mflux_import_nodes_ast():
    """Catches function-local `from mflux import X` that importlib checks miss.
    Immune to docstring mentions (AST inspects import nodes only). Covers _kernel/,
    the package root, variants/*.py helpers and every variant config.py / detect.py."""
    import ast

    sources = _mflux_free_sources()
    names = {p.name for p in sources}
    assert {"gate.py", "__init__.py", "config.py", "detect.py"} <= names
    offenders = []
    for py in sources:
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("mflux"):
                offenders.append(f"{py}:{node.lineno} from {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("mflux"):
                        offenders.append(f"{py}:{node.lineno} import {alias.name}")
    assert not offenders, f"mflux imports in mflux-free modules: {offenders}"
