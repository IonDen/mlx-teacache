"""Public API surface snapshot. Locks v0.5.x → v0.6.0 compatibility."""

from __future__ import annotations

import subprocess
import sys


def test_root_package_exports() -> None:
    import mlx_teacache

    for name in [
        "__version__",
        "apply_teacache",
        "TeaCacheHandle",
        "TeaCacheStats",
        "GenerationStats",
        "StepDecision",
        "Provenance",
        "TeaCacheError",
        "TeaCacheValueError",
        "AlreadyPatchedError",
        "CalibrationError",
        "IncompatibleModelError",
        "InternalStateError",
        "InvalidStepWindowError",
        "MissingGenerationContextError",
        "StatsFrozenError",
        "TeaCacheDisabledWarning",
        "TeaCacheNoBenefitWarning",
        "TransformerShapeError",
    ]:
        assert hasattr(mlx_teacache, name), f"missing public export: {name}"


def test_stats_submodule_paths() -> None:
    from mlx_teacache.stats import (  # noqa: F401
        GenerationStats,
        StatsFrozenError,
        StepDecision,
        TeaCacheStats,
    )

    s = TeaCacheStats()
    assert s.computed_count == 0
    assert s.speedup_estimate == 1.0


def test_coefficients_provenance_path() -> None:
    from mlx_teacache.coefficients import Provenance

    assert Provenance.for_user_supplied().source == "user"


def test_gate_module_path() -> None:
    from mlx_teacache._kernel.gate import GateDecision as _KernelGD
    from mlx_teacache._kernel.gate import gate_step as _kernel_gate_step
    from mlx_teacache.gate import GateDecision, gate_step

    assert GateDecision is _KernelGD, "gate.GateDecision must be the _kernel original"
    assert gate_step is _kernel_gate_step, "gate.gate_step must be the _kernel original"


def test_cache_module_path() -> None:
    from mlx_teacache._kernel.cache import TeaCacheState as _KernelTCS
    from mlx_teacache.cache import TeaCacheState

    assert TeaCacheState is _KernelTCS, "cache.TeaCacheState must be the _kernel original"


def test_base_import_without_mflux() -> None:
    """Audit F4: base-package import must work without [mflux] extra."""
    code = (
        "import sys\n"
        "sys.modules['mflux'] = None\n"
        "import mlx_teacache\n"
        "from mlx_teacache import apply_teacache\n"
        "assert callable(apply_teacache)\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert "OK" in result.stdout, f"stderr={result.stderr}"
    assert result.returncode == 0


def test_apply_teacache_docstring_documents_per_variant_defaults():
    from mlx_teacache import apply_teacache

    doc = apply_teacache.__doc__
    assert doc is not None
    # per-variant default resolution is spelled out
    assert "0.17" in doc  # base-4b / base-9b default
    assert "0.12" in doc  # z-image-base default
    assert "0.20" in doc  # package fallback
    # the resolved effective value is pointed at
    assert "handle.rel_l1_thresh" in doc
