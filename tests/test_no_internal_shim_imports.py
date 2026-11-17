"""Guard: nothing in the repo imports the four root shims except the tests that pin them.

`mlx_teacache.gate`, `.cache`, `.stats` and `.coefficients` are compatibility shims over
`mlx_teacache._kernel`. Once they warn on access, an incidental import of one turns an unrelated
test red under `filterwarnings = error`, and an import from library code warns every user. Code
imports from `mlx_teacache._kernel.*` (or the package root) instead. AST-level, so a docstring or
comment that names a shim cannot trip it.
"""

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCANNED_DIRS = ("src/mlx_teacache", "tests", "scripts", "examples", "docs")
_SHIM_NAMES = frozenset({"gate", "cache", "stats", "coefficients"})
_SHIM_MODULES = frozenset(f"mlx_teacache.{name}" for name in _SHIM_NAMES)

# The tests whose job is to pin the shims themselves, plus this file.
_ALLOWED = frozenset(
    {
        "tests/test_public_api.py",
        "tests/_kernel/test_gate_equivalence.py",
        "tests/_kernel/test_cache_equivalence.py",
        "tests/_kernel/test_stats_equivalence.py",
        "tests/_kernel/test_coefficients_provenance.py",
        "tests/test_shim_deprecation.py",
        "tests/test_no_internal_shim_imports.py",
    }
)


def _resolve_relative(module: str | None, level: int, package: str) -> str:
    """The absolute module a `from <dots><module> import ...` names inside `package`."""
    parts = package.split(".")
    base = parts[: len(parts) - (level - 1)] if level > 1 else parts
    return ".".join([*base, module] if module else base)


def shim_import_lines(source: str, *, package: str | None = None) -> list[int]:
    """Line numbers in `source` that import a root shim.

    `package` is the dotted package the source lives in; it is needed only to resolve relative
    imports, and a relative import in a file outside the package is ignored.
    """
    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(alias.name in _SHIM_MODULES for alias in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if package is None:
                    continue
                module = _resolve_relative(node.module, node.level, package)
            else:
                module = node.module or ""
            if module in _SHIM_MODULES or (
                module == "mlx_teacache" and any(alias.name in _SHIM_NAMES for alias in node.names)
            ):
                lines.append(node.lineno)
    return lines


def _package_of(path: Path) -> str | None:
    """Dotted package of a file under src/, None for files outside the installed package."""
    src = _REPO / "src"
    if not path.is_relative_to(src):
        return None
    return ".".join(path.relative_to(src).parent.parts)


def _scanned_files() -> list[Path]:
    files: list[Path] = []
    for rel in _SCANNED_DIRS:
        files.extend(p for p in (_REPO / rel).rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(files)


def test_no_module_imports_a_root_shim() -> None:
    """Bug: a module imports a root shim, so from 0.16 on an unrelated test fails under
    filterwarnings=error, or library code warns every user."""
    offenders: dict[str, list[int]] = {}
    for path in _scanned_files():
        lines = shim_import_lines(path.read_text(encoding="utf-8"), package=_package_of(path))
        if lines:
            offenders[path.relative_to(_REPO).as_posix()] = lines
    unexpected = {rel: lines for rel, lines in offenders.items() if rel not in _ALLOWED}
    assert not unexpected, (
        "these files import a root shim (mlx_teacache.gate/cache/stats/coefficients); import from "
        "mlx_teacache._kernel.* instead:\n"
        + "\n".join(f"  {rel}: lines {lines}" for rel, lines in sorted(unexpected.items()))
    )


def test_the_scan_covers_library_code_and_tests() -> None:
    """Bug: a wrong repo root or glob makes the scan empty, so the guard above passes vacuously."""
    scanned = {path.relative_to(_REPO).as_posix() for path in _scanned_files()}
    assert {
        "src/mlx_teacache/api.py",
        "src/mlx_teacache/variants/flux1_dev/integration.py",
        "tests/test_gate.py",
        "scripts/calibrate_flux2.py",
        "examples/mflux_teacache_flux1.py",
        "docs/_generate_comparison.py",
    } <= scanned


def test_library_files_get_their_package_for_relative_imports() -> None:
    """Bug: library files are scanned without their package, so relative shim imports go unchecked."""
    src = _REPO / "src" / "mlx_teacache"
    assert _package_of(src / "variants" / "flux1_dev" / "integration.py") == "mlx_teacache.variants.flux1_dev"
    assert _package_of(src / "variants" / "__init__.py") == "mlx_teacache.variants"
    assert _package_of(src / "api.py") == "mlx_teacache"
    assert _package_of(_REPO / "tests" / "test_gate.py") is None


def test_guard_flags_each_import_form() -> None:
    """Bug: the collector misses one import form, or flags the canonical _kernel import."""
    assert shim_import_lines("from mlx_teacache.gate import gate_step") == [1]
    assert shim_import_lines("from mlx_teacache import stats") == [1]
    assert shim_import_lines("import mlx_teacache.cache") == [1]
    assert shim_import_lines("import mlx_teacache.coefficients as c") == [1]
    assert shim_import_lines("from mlx_teacache._kernel.gate import gate_step") == []
    assert shim_import_lines("from mlx_teacache import TeaCacheStats, apply_teacache") == []
    assert shim_import_lines("import mlx_teacache._kernel.cache") == []


def test_guard_resolves_relative_imports_inside_the_package() -> None:
    """Bug: a relative import of a shim from library code (`from ..stats import X`) slips past."""
    variant = "mlx_teacache.variants.flux1_dev"
    assert shim_import_lines("from ...stats import TeaCacheStats", package=variant) == [1]
    assert shim_import_lines("from ... import gate", package=variant) == [1]
    assert shim_import_lines("from .gate import gate_step", package="mlx_teacache") == [1]
    assert shim_import_lines("from .gate import gate_step", package="mlx_teacache._kernel") == []
    assert shim_import_lines("from .config import COEFFICIENTS", package=variant) == []
    assert shim_import_lines("from .gate import gate_step") == []
