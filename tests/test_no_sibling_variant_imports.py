"""Guard: no code module imports a sibling variant.

Each variant package under `mlx_teacache/variants/<variant_id>/` owns its detector, config and integration. Code they
share lives in `variants/_*.py` (mflux-free helpers) or under `integrations/` and `_kernel/`. A variant that imports a
sibling's code couples the two: editing one variant's forward silently changes another. The rules:

- a module in `variants/<pkg>/` imports no `mlx_teacache.variants.<other>`; the one exception is a `config.py`
  reusing a sibling's fitted data by identity, allowed for exactly the pairs in `_CONFIG_REUSE`;
- a shared `variants/_*.py` module imports no variant package;
- nothing under `integrations/` imports a variant package (the shared `variants/_*.py` modules are fine), and
  nothing under `_kernel/` imports `mlx_teacache.variants` at all;
- `detect.py`, `config.py` and the shared `variants/_*.py` modules import neither `mlx_teacache.integrations` nor
  mflux, so detection stays pure-core;
- only the registry (`variants/__init__.py`) imports a variant by name: `importlib.import_module` or `__import__` of
  a `mlx_teacache.variants.<id>` string (or an f-string starting `mlx_teacache.variants.`) anywhere else is flagged.

AST-level, so a docstring or comment that names a module cannot trip it; function-local and `TYPE_CHECKING` imports
count like module-level ones.
"""

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
_VARIANTS = "mlx_teacache.variants"

# (importing variant, variant whose config it reuses): config.py -> <other>.config only.
_CONFIG_REUSE = frozenset({("flux1_schnell", "flux1_dev"), ("flux2_klein_base_9b", "flux2_klein_base_4b")})


def _resolve_relative(module: str | None, level: int, package: str) -> str:
    """The absolute module a `from <dots><module> import ...` names inside `package`."""
    parts = package.split(".")
    base = parts[: len(parts) - (level - 1)] if level > 1 else parts
    return ".".join([*base, module] if module else base)


def _targets(node: ast.Import | ast.ImportFrom, package: str) -> list[str]:
    """The dotted modules (or module attributes) one import statement reaches."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    module = _resolve_relative(node.module, node.level, package) if node.level else node.module or ""
    named = [f"{module}.{alias.name}" for alias in node.names]
    # `from mlx_teacache.variants.<id> import config`: the names decide which of the package's modules it reaches.
    is_variant_package = module.startswith(f"{_VARIANTS}.") and module.count(".") == 2
    return named if is_variant_package else [module, *named]


def _variant_of(target: str, variant_ids: frozenset[str]) -> tuple[str, str | None] | None:
    """(variant id, its submodule or None) when `target` is inside a variant package, else None."""
    parts = target.split(".")
    if parts[:2] != _VARIANTS.split(".") or len(parts) < 3 or parts[2] not in variant_ids:
        return None
    return parts[2], parts[3] if len(parts) > 3 else None


def _under(target: str, prefix: str) -> bool:
    return target == prefix or target.startswith(f"{prefix}.")


def _breaks_rule(module: str, target: str, variant_ids: frozenset[str]) -> bool:
    """True when the module at dotted path `module` may not import `target`."""
    parts = module.split(".")
    hit = _variant_of(target, variant_ids)
    in_variant = parts[:2] == _VARIANTS.split(".") and len(parts) > 3 and parts[2] in variant_ids
    shared = parts[:2] == _VARIANTS.split(".") and len(parts) == 3 and parts[2].startswith("_")
    if in_variant and hit is not None and hit[0] != parts[2]:
        reuses_config = parts[3] == "config" and hit[1] == "config" and (parts[2], hit[0]) in _CONFIG_REUSE
        if not reuses_config:
            return True
    if shared and hit is not None:
        return True
    if _under(module, "mlx_teacache.integrations") and hit is not None:
        return True
    if _under(module, "mlx_teacache._kernel") and _under(target, _VARIANTS):
        return True
    pure_core = shared or (in_variant and parts[3] in {"detect", "config"})
    return pure_core and (_under(target, "mlx_teacache.integrations") or _under(target, "mflux"))


def _imports_a_variant_by_name(node: ast.Call, variant_ids: frozenset[str]) -> bool:
    """True for `import_module(...)` / `__import__(...)` whose first argument names a variant module."""
    callee = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
    if callee not in {"import_module", "__import__"} or not node.args:
        return False
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return _variant_of(first.value, variant_ids) is not None
    if isinstance(first, ast.JoinedStr) and first.values:
        head = first.values[0]
        return isinstance(head, ast.Constant) and str(head.value).startswith(f"{_VARIANTS}.")
    return False


def forbidden_import_lines(source: str, *, module: str, variant_ids: frozenset[str]) -> list[int]:
    """Line numbers in `source` (the module at dotted path `module`) whose import breaks the rules above."""
    package = module.rsplit(".", 1)[0]
    is_registry = module == f"{_VARIANTS}.__init__"
    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        static = isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            _breaks_rule(module, target, variant_ids) for target in _targets(node, package)
        )
        by_name = (
            isinstance(node, ast.Call) and not is_registry and _imports_a_variant_by_name(node, variant_ids)
        )
        if static or by_name:
            lines.append(node.lineno)
    return sorted(lines)


def _variant_ids() -> frozenset[str]:
    root = _SRC / "mlx_teacache" / "variants"
    return frozenset(
        p.name for p in root.iterdir() if (p / "__init__.py").is_file() and not p.name.startswith(("_", "."))
    )


def _module_of(path: Path) -> str:
    """Dotted module of a file under src/ (`pkg/__init__.py` is `pkg.__init__`, so its package is `pkg`)."""
    return ".".join(path.relative_to(_SRC).with_suffix("").parts)


def _library_files() -> list[Path]:
    return sorted(p for p in (_SRC / "mlx_teacache").rglob("*.py") if "__pycache__" not in p.parts)


def test_no_module_imports_a_sibling_variant() -> None:
    """Bug: a variant imports a sibling's code (or an integration imports a variant), so editing one variant
    silently changes another, or detection pulls in mflux."""
    ids = _variant_ids()
    offenders = {
        path.relative_to(_REPO).as_posix(): lines
        for path in _library_files()
        if (
            lines := forbidden_import_lines(
                path.read_text(encoding="utf-8"), module=_module_of(path), variant_ids=ids
            )
        )
    }
    assert not offenders, (
        "these modules break the variant import rules (see this file's docstring):\n"
        + "\n".join(f"  {rel}: lines {lines}" for rel, lines in sorted(offenders.items()))
    )


def test_the_scan_covers_every_layer() -> None:
    """Bug: a wrong root or glob makes the scan (or the variant list) empty, so the guard passes vacuously."""
    scanned = {path.relative_to(_REPO).as_posix() for path in _library_files()}
    assert {
        "src/mlx_teacache/variants/qwen_image_2512/detect.py",
        "src/mlx_teacache/variants/flux1_schnell/config.py",
        "src/mlx_teacache/variants/_pipeline_class.py",
        "src/mlx_teacache/integrations/mflux/forward_qwen.py",
        "src/mlx_teacache/_kernel/step.py",
    } <= scanned
    assert {
        "flux1_dev",
        "flux1_schnell",
        "flux2_klein_base_4b",
        "qwen_image",
        "qwen_image_2512",
        "z_image_base",
    } <= (_variant_ids())
    assert "_pipeline_class" not in _variant_ids()


def test_each_config_reuse_pair_is_still_used() -> None:
    """Bug: an allowlisted pair outlives the import it was for, so a later sibling import there slips through."""
    root = _SRC / "mlx_teacache" / "variants"
    for importer, source in sorted(_CONFIG_REUSE):
        text = (root / importer / "config.py").read_text(encoding="utf-8")
        assert f"{_VARIANTS}.{source}.config" in text, (importer, source)


_IDS = frozenset({"flux1_dev", "flux1_schnell", "qwen_image", "qwen_image_2512", "z_image_base"})
_DETECT_2512 = "mlx_teacache.variants.qwen_image_2512.detect"
_SCHNELL_CONFIG = "mlx_teacache.variants.flux1_schnell.config"


def _lines(source: str, module: str) -> list[int]:
    return forbidden_import_lines(source, module=module, variant_ids=_IDS)


def test_guard_flags_each_sibling_import_form() -> None:
    """Bug: the collector misses one import form (absolute, relative, package attribute)."""
    assert _lines("from mlx_teacache.variants.qwen_image.detect import has_zero_cond_t", _DETECT_2512) == [1]
    assert _lines("import mlx_teacache.variants.qwen_image.detect", _DETECT_2512) == [1]
    assert _lines("from mlx_teacache.variants import qwen_image", _DETECT_2512) == [1]
    assert _lines("from mlx_teacache.variants.qwen_image import detect", _DETECT_2512) == [1]
    assert _lines("from ..qwen_image.detect import is_2512_checkpoint", _DETECT_2512) == [1]
    assert _lines("from .. import qwen_image", _DETECT_2512) == [1]
    assert _lines(
        "def f():\n    from mlx_teacache.variants.qwen_image import integration\n", _DETECT_2512
    ) == [2]


def test_guard_allows_own_variant_and_shared_modules() -> None:
    """Bug: the guard flags a variant's own modules or the shared helpers, so the rule cannot be met."""
    assert _lines("from mlx_teacache.variants.qwen_image_2512.config import META", _DETECT_2512) == []
    assert _lines("from .config import META", _DETECT_2512) == []
    assert _lines("from mlx_teacache.variants._qwen_checkpoint import is_2512_checkpoint", _DETECT_2512) == []
    assert _lines("from .._pipeline_class import config_matches", _DETECT_2512) == []
    assert _lines("from mlx_teacache.variants import match_variant", _DETECT_2512) == []


def test_config_reuse_is_allowed_only_for_the_listed_config_pairs() -> None:
    """Bug: the config exception widens to another module, another target or another pair."""
    reuse = "from mlx_teacache.variants.flux1_dev.config import COEFFICIENTS"
    assert _lines(reuse, _SCHNELL_CONFIG) == []
    assert _lines("from mlx_teacache.variants.flux1_dev import config", _SCHNELL_CONFIG) == []
    assert _lines(reuse, "mlx_teacache.variants.flux1_schnell.integration") == [1]
    assert _lines("from mlx_teacache.variants.flux1_dev.detect import matches", _SCHNELL_CONFIG) == [1]
    assert _lines("from mlx_teacache.variants.flux1_dev import COEFFICIENTS", _SCHNELL_CONFIG) == [1]
    assert _lines(reuse, "mlx_teacache.variants.z_image_base.config") == [1]


def test_integrations_and_kernel_import_no_variant() -> None:
    """Bug: a shared forward imports one variant's module, so that variant's edits reach every user of the forward."""
    forward = "mlx_teacache.integrations.mflux.forward_qwen"
    assert _lines(
        "from mlx_teacache.variants.qwen_image.detect import is_calibrated_checkpoint", forward
    ) == [1]
    assert (
        _lines("from mlx_teacache.variants._qwen_checkpoint import is_calibrated_checkpoint", forward) == []
    )
    assert _lines(
        "from mlx_teacache.variants._pipeline_class import config_matches", "mlx_teacache._kernel.step"
    ) == [1]
    assert _lines("from mlx_teacache._kernel.step import step_decision", "mlx_teacache._kernel.resolve") == []


def test_shared_variant_modules_import_no_variant() -> None:
    """Bug: a shared helper imports a variant, so the helper's other users depend on that variant."""
    shared = "mlx_teacache.variants._qwen_checkpoint"
    assert _lines("from mlx_teacache.variants.qwen_image.detect import same_repo_id", shared) == [1]
    assert _lines("from ._pipeline_class import config_matches", shared) == []


def test_detection_modules_stay_free_of_integrations_and_mflux() -> None:
    """Bug: a detect, config or shared helper imports the integrations or mflux, so `import mlx_teacache` (or
    detection on a machine without mflux) pulls in the runner."""
    for module in (_DETECT_2512, _SCHNELL_CONFIG, "mlx_teacache.variants._qwen_checkpoint"):
        assert _lines("from mlx_teacache.integrations._session import CfgBranchPairer", module) == [1], module
        assert _lines("from mlx_teacache import integrations", module) == [1], module
        assert _lines("import mflux.models", module) == [1], module
        assert _lines("from mlx_teacache.errors import CalibrationError", module) == [], module
    integration = "mlx_teacache.variants.qwen_image_2512.integration"
    assert _lines("from mlx_teacache.integrations.mflux.forward_qwen import apply_qwen", integration) == []


_REGISTRY_MODULE = "mlx_teacache.variants.__init__"
_FORWARD_QWEN = "mlx_teacache.integrations.mflux.forward_qwen"


def test_guard_flags_string_imports_of_a_variant_outside_the_registry() -> None:
    """Bug: a module reaches a variant through `importlib.import_module("mlx_teacache.variants.<id>...")`, which
    is no import statement, so the AST collector never sees it and the coupling ships."""
    assert _lines('importlib.import_module("mlx_teacache.variants.qwen_image.detect")', _DETECT_2512) == [1]
    assert _lines(
        'from importlib import import_module\nmod = import_module("mlx_teacache.variants.qwen_image")\n',
        _FORWARD_QWEN,
    ) == [2]
    assert _lines('__import__("mlx_teacache.variants.qwen_image.detect")', _FORWARD_QWEN) == [1]
    assert _lines(
        'def f(vid):\n    return importlib.import_module(f"mlx_teacache.variants.{vid}.integration")\n',
        _FORWARD_QWEN,
    ) == [2]


def test_guard_allows_string_imports_in_the_registry_and_of_other_modules() -> None:
    """Bug: the string-import check flags the registry, which loads every variant by name on purpose, or a
    string import of a non-variant module, so the rule cannot be met."""
    assert (
        _lines('integration = importlib.import_module(f"{module_name}.integration")', _REGISTRY_MODULE) == []
    )
    assert (
        _lines('importlib.import_module("mlx_teacache.variants.qwen_image.detect")', _REGISTRY_MODULE) == []
    )
    assert _lines('importlib.import_module("mlx_teacache.errors")', _DETECT_2512) == []
    assert _lines('importlib.import_module("mlx_teacache.variants._qwen_checkpoint")', _FORWARD_QWEN) == []
    assert _lines("getattr(importlib.import_module(sources[name]), name)", "mlx_teacache._deprecation") == []
