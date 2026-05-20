"""Public API surface snapshot. Locks v0.5.x → v0.6.0 compatibility."""
from __future__ import annotations

import inspect
import subprocess
import sys


def test_root_package_exports() -> None:
    import mlx_teacache
    for name in [
        "__version__", "apply_teacache", "TeaCacheHandle", "TeaCacheStats",
        "GenerationStats", "StepDecision", "Provenance",
        "TeaCacheError", "AlreadyPatchedError", "CalibrationError",
        "IncompatibleModelError", "InternalStateError", "InvalidStepWindowError",
        "MissingGenerationContextError", "StatsFrozenError",
        "TeaCacheNoBenefitWarning", "TransformerShapeError",
    ]:
        assert hasattr(mlx_teacache, name), f"missing public export: {name}"


def test_stats_submodule_paths() -> None:
    from mlx_teacache.stats import GenerationStats, StatsFrozenError, StepDecision, TeaCacheStats  # noqa: F401
    s = TeaCacheStats()
    assert s.computed_count == 0
    assert s.speedup_estimate == 1.0


def test_coefficients_provenance_path() -> None:
    from mlx_teacache.coefficients import Provenance
    assert Provenance.for_user_supplied().source == "user"


def test_gate_module_path() -> None:
    from mlx_teacache.gate import GateDecision, gate_step  # noqa: F401


def test_cache_module_path() -> None:
    from mlx_teacache.cache import TeaCacheState  # noqa: F401


def test_apply_teacache_signature() -> None:
    """All four explicit kwargs must survive (audit F3)."""
    from mlx_teacache import apply_teacache
    sig = inspect.signature(apply_teacache)
    for name in ("rel_l1_thresh", "coefficients",
                 "skip_first_n_steps", "skip_last_n_steps"):
        assert name in sig.parameters, f"public kwarg missing: {name}"


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
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert "OK" in result.stdout, f"stderr={result.stderr}"
    assert result.returncode == 0
