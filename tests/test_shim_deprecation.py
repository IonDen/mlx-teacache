"""The four root modules `mlx_teacache.gate`, `.cache`, `.stats` and `.coefficients` are deprecated.

Every name they exported in 0.15 still resolves to the same object, with a DeprecationWarning that
points at the user's line. Unknown names raise a plain AttributeError and never warn, so `hasattr`
probes stay quiet. Importing the package itself never touches a deprecated module.
"""

import importlib
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType

import pytest

import mlx_teacache

# The 0.15 exports of each module, and where each name really lives.
_SHIMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "mlx_teacache.gate": (
        "mlx_teacache._kernel.gate",
        ("GateDecision", "GateKind", "gate_step", "mean_abs_rel_l1", "poly_eval"),
    ),
    "mlx_teacache.cache": ("mlx_teacache._kernel.cache", ("TeaCacheState",)),
    "mlx_teacache.stats": (
        "mlx_teacache._kernel.stats",
        ("Decision", "GenerationStats", "StatsFrozenError", "StepDecision", "TeaCacheStats"),
    ),
    "mlx_teacache.coefficients": ("mlx_teacache._kernel.coefficients", ("Provenance", "validate_custom")),
}

# Names a user can import from the package root instead.
_ROOT_PUBLIC = {"TeaCacheStats", "GenerationStats", "StepDecision", "StatsFrozenError", "Provenance"}

_CASES = [(shim, kernel, name) for shim, (kernel, names) in _SHIMS.items() for name in names]
# Not in `__all__`, but reachable through the stats module in 0.15.
_CASES.append(("mlx_teacache.stats", "mlx_teacache._kernel.stats", "_Staging"))
_CASE_IDS = [f"{shim.rsplit('.', 1)[1]}.{name}" for shim, _, name in _CASES]

# Code the fresh interpreter runs in test_importing_the_package_does_not_warn.
_PACKAGE_IMPORT_CODE = "import mlx_teacache; import mlx_teacache.variants"


def _shim(name: str) -> ModuleType:
    return importlib.import_module(name)


@pytest.mark.parametrize(("shim", "kernel", "name"), _CASES, ids=_CASE_IDS)
def test_every_shim_name_warns_and_is_the_kernel_object(shim: str, kernel: str, name: str) -> None:
    module = _shim(shim)
    expected = getattr(importlib.import_module(kernel), name)
    with pytest.warns(DeprecationWarning, match="deprecated") as rec:
        value = getattr(module, name)
    assert value is expected
    assert len(rec) == 1
    assert rec[0].category is DeprecationWarning
    assert rec[0].filename == __file__


def test_from_import_statement_warns_at_the_import_line() -> None:
    code = compile("from mlx_teacache.gate import gate_step", "user_script.py", "exec")
    with pytest.warns(DeprecationWarning, match="deprecated") as rec:
        exec(code, {})
    assert len(rec) == 1
    assert rec[0].filename == "user_script.py"
    assert rec[0].lineno == 1


def test_first_import_in_a_fresh_process_warns_at_the_import_line(tmp_path: Path) -> None:
    script = tmp_path / "user_script.py"
    script.write_text("import sys\nfrom mlx_teacache.stats import TeaCacheStats\n")
    result = subprocess.run(
        [sys.executable, "-W", "always::DeprecationWarning", str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert f"{script}:2: DeprecationWarning: mlx_teacache.stats.TeaCacheStats is deprecated" in result.stderr
    assert result.stderr.count("DeprecationWarning") == 1, result.stderr


@pytest.mark.parametrize(("shim", "kernel", "name"), _CASES, ids=_CASE_IDS)
def test_message_names_the_right_replacement(shim: str, kernel: str, name: str) -> None:
    with pytest.warns(DeprecationWarning) as rec:
        getattr(_shim(shim), name)
    message = str(rec[0].message)
    assert message.startswith(f"{shim}.{name} is deprecated")
    if name in _ROOT_PUBLIC:
        assert f"import {name} from mlx_teacache instead" in message
        assert "no public replacement" not in message
    else:
        assert "no public replacement" in message
        assert "from mlx_teacache instead" not in message
    assert "will be removed in 2.0" in message
    assert "_kernel" not in message


def test_root_public_names_are_exported_at_the_root() -> None:
    from mlx_teacache import _deprecation

    assert _deprecation._ROOT_PUBLIC == _ROOT_PUBLIC
    shim_names = {name for _, _, name in _CASES}
    assert shim_names & set(mlx_teacache.__all__) == _ROOT_PUBLIC
    for _, kernel, name in _CASES:
        if name in _ROOT_PUBLIC:
            assert getattr(mlx_teacache, name) is getattr(importlib.import_module(kernel), name)


@pytest.mark.parametrize("shim", _SHIMS)
def test_unknown_attribute_raises_attribute_error_without_warning(shim: str) -> None:
    module = _shim(shim)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(AttributeError, match=rf"^module '{shim}' has no attribute 'nope'$"):
            module.nope  # noqa: B018
        assert not hasattr(module, "__path__")
        assert not hasattr(module, "_kernel")


@pytest.mark.parametrize("shim", _SHIMS)
def test_dir_lists_the_public_names(shim: str) -> None:
    module = _shim(shim)
    _, names = _SHIMS[shim]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert dir(module) == sorted(names)
        assert sorted(module.__all__) == sorted(names)


def test_importing_the_package_does_not_warn() -> None:
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", _PACKAGE_IMPORT_CODE],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
