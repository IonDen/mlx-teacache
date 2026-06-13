"""The package root must import its public types from the canonical `_kernel`
modules, not from the deprecated top-level compat shims (`mlx_teacache.coefficients`
/ `mlx_teacache.stats`).

The shims re-export the same objects from `_kernel`, so object *identity* is
already correct (pinned elsewhere). This guards the *structural* dependency:
the package entry point should not depend on its own deprecated shims
(backlog 0031 #9). AST-level so a docstring mention can't trip it.
"""

import ast
from pathlib import Path

# public name -> the _kernel module it must be imported from
_KERNEL_SOURCES = {
    "Provenance": "mlx_teacache._kernel.coefficients",
    "GenerationStats": "mlx_teacache._kernel.stats",
    "StatsFrozenError": "mlx_teacache._kernel.stats",
    "StepDecision": "mlx_teacache._kernel.stats",
    "TeaCacheStats": "mlx_teacache._kernel.stats",
}


def _init_path() -> Path:
    return Path(__file__).resolve().parent.parent / "src" / "mlx_teacache" / "__init__.py"


def _import_occurrences(tree: ast.Module) -> list[tuple[str, str]]:
    """Every (name, module) pair where __init__ does `from <module> import <name>`
    for a tracked public name. A list, not a dict, so a duplicate import — the same
    name pulled from both a shim and _kernel — can't be silently collapsed (which
    could otherwise let a shim import slip past)."""
    pairs: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name in _KERNEL_SOURCES:
                    pairs.append((alias.name, node.module))
    return pairs


def test_public_types_imported_from_kernel_not_shims() -> None:
    init = _init_path()
    assert init.is_file(), f"__init__.py not found at {init}"
    pairs = _import_occurrences(ast.parse(init.read_text()))
    seen = {name for name, _ in pairs}

    missing = [name for name in _KERNEL_SOURCES if name not in seen]
    assert not missing, f"public types not imported in __init__.py: {missing}"

    offenders = [(name, module) for name, module in pairs if module != _KERNEL_SOURCES[name]]
    assert not offenders, (
        "package root imports public types from the wrong module — expected the "
        f"_kernel canonical source, got: {offenders}"
    )
