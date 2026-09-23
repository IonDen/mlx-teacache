"""Public API surface snapshot. Locks v0.5.x → v0.6.0 compatibility."""

from __future__ import annotations

import re
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


def test_apply_teacache_docstring_points_at_the_resolved_threshold():
    from mlx_teacache import apply_teacache

    doc = apply_teacache.__doc__
    assert doc is not None
    assert "handle.rel_l1_thresh" in doc


def _docstring_default_rows(doc: str) -> list[tuple[list[str], str]]:
    """The `- <names> ..... <value>` rows of apply_teacache's per-variant default table."""
    rows = []
    for m in re.finditer(r"^\s*- (?P<names>[a-z0-9][a-z0-9, -]*?) \.{3,} (?P<rest>.+)$", doc, re.MULTILINE):
        rows.append(([n.strip() for n in m.group("names").split(",")], m.group("rest")))
    return rows


def _names_variant(token: str, variant_id: str) -> bool:
    # "flux2-klein-base-4b, -base-9b" abbreviates the second id to its suffix.
    return token == variant_id or (token.startswith("-") and variant_id.endswith(token))


def test_apply_teacache_docstring_default_table_matches_the_registry():
    """Every registered variant has exactly one row in the docstring's default table, and the
    row shows the variant's actual DEFAULT_THRESH (or says it has none). Goes red when a
    recalibration changes a config default without touching the docstring, or when a new
    variant ships without a row."""
    from mlx_teacache import apply_teacache
    from mlx_teacache.variants import _REGISTRY

    rows = _docstring_default_rows(apply_teacache.__doc__ or "")
    assert rows, "no per-variant default table found in the apply_teacache docstring"
    for variant_id, entry in _REGISTRY.items():
        matching = [rest for names, rest in rows if any(_names_variant(n, variant_id) for n in names)]
        assert len(matching) == 1, f"{variant_id}: {len(matching)} docstring rows"
        default = entry["default_thresh"]
        if default is None:
            assert matching[0].startswith("no per-variant default"), (variant_id, matching[0])
        else:
            assert matching[0].startswith(f"{default:.2f}"), (variant_id, default, matching[0])
