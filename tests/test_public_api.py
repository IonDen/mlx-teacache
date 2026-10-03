"""Public API surface snapshot. Locks v0.5.x → v0.6.0 compatibility."""

from __future__ import annotations

import re
import subprocess
import sys

EXPECTED_ALL = {
    "__version__",
    "apply_teacache",
    "TeaCacheHandle",
    "TeaCacheStats",
    "GenerationStats",
    "StepDecision",
    "StatsFrozenError",
    "Provenance",
    "TeaCacheError",
    "TeaCacheValueError",
    "TeaCacheDisabledWarning",
    "TeaCacheNoBenefitWarning",
    "TeaCacheUncalibratedCheckpointWarning",
    "TeaCacheUntestedMfluxWarning",
    "IncompatibleModelError",
    "AlreadyPatchedError",
    "CalibrationError",
    "TransformerShapeError",
    "InternalStateError",
    "InvalidStepWindowError",
    "MissingGenerationContextError",
}


def test_all_is_exactly_the_public_surface() -> None:
    """Bug caught: dropping TeaCacheUncalibratedCheckpointWarning from __init__ passes the hasattr list."""
    import mlx_teacache

    assert set(mlx_teacache.__all__) == EXPECTED_ALL
    for name in EXPECTED_ALL:
        getattr(mlx_teacache, name)


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


def _stale_row_names(rows: list[tuple[list[str], str]], variant_ids) -> list[str]:  # noqa: ANN001
    """Every name in a docstring row that no live variant answers to, in row order."""
    ids = list(variant_ids)
    return [n for names, _ in rows for n in names if not any(_names_variant(n, v) for v in ids)]


def test_apply_teacache_docstring_default_table_matches_the_registry():
    """Every registered variant has exactly one row in the docstring's default table, and the
    row shows the variant's actual DEFAULT_THRESH (or says it has none). Goes red when a
    recalibration changes a config default without touching the docstring, when a new
    variant ships without a row, or when a row outlives the variant it names."""
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
    # The converse, per name: a row (or one name in a grouped row) that outlives its variant.
    assert _stale_row_names(rows, _REGISTRY) == []


def test_stale_row_names_catches_one_dropped_name_in_a_grouped_row():
    """bug caught: checking a grouped row with any() instead of per name. Dropping only
    flux2-klein-base-9b while flux2-klein-base-4b stays registered must still flag the
    row's `-base-9b` token as stale."""
    from mlx_teacache import apply_teacache
    from mlx_teacache.variants import _REGISTRY

    rows = _docstring_default_rows(apply_teacache.__doc__ or "")
    assert _stale_row_names(rows, _REGISTRY) == []
    without_base_9b = [v for v in _REGISTRY if v != "flux2-klein-base-9b"]
    assert _stale_row_names(rows, without_base_9b) == ["-base-9b"]


_NAMES = r"[a-z0-9][a-z0-9-]*(?: / -[a-z0-9-]+| / [a-z0-9][a-z0-9-]*)*"


def _expand_names(group: str) -> list[str]:
    """``"flux2-klein-base-4b / -base-9b"`` -> both full ids. A ``-suffix`` token
    replaces as many trailing dash-segments of the group's first name."""
    tokens = [t.strip() for t in group.split(" / ")]
    first = tokens[0]
    out = [first]
    for tok in tokens[1:]:
        if tok.startswith("-"):
            keep = first.split("-")[: -(tok.count("-"))]
            out.append("-".join(keep) + tok)
        else:
            out.append(tok)
    return out


def _package_doc_defaults(doc: str) -> tuple[dict[str, str], list[str], str | None]:
    """Parse the root docstring's default paragraph into (id -> threshold text,
    distilled ids, distilled fallback text)."""
    para = doc[doc.index("rel_l1_thresh defaults to") :].split("\n\n")[0]
    para = " ".join(para.split())
    per_id: dict[str, str] = {}
    for m in re.finditer(r"(?<![\w.])(\d\.\d\d) (" + _NAMES + ")", para):
        for name in _expand_names(m.group(2)):
            assert name not in per_id, f"{name} named twice"
            per_id[name] = m.group(1)
    distilled = re.search(r"Distilled (" + _NAMES + r") have none and use (\d\.\d\d)", para)
    if distilled is None:
        return per_id, [], None
    return per_id, _expand_names(distilled.group(1)), distilled.group(2)


def test_package_docstring_states_each_variant_with_its_own_default() -> None:
    """Bug caught: the root docstring pairs a variant with another variant's number
    (e.g. "0.30 flux1-dev" while flux1-dev defaults to 0.20 and Krea to 0.30), drops a
    variant, or stops naming the distilled pair that has no default of its own."""
    import mlx_teacache
    from mlx_teacache.variants import _REGISTRY

    doc = mlx_teacache.__doc__ or ""
    per_id, distilled, fallback = _package_doc_defaults(doc)
    with_default = {
        vid: f"{e['default_thresh']:.2f}" for vid, e in _REGISTRY.items() if e["default_thresh"] is not None
    }
    without_default = sorted(vid for vid, e in _REGISTRY.items() if e["default_thresh"] is None)
    assert per_id == with_default
    assert sorted(distilled) == without_default == ["flux2-klein-4b", "flux2-klein-9b"]
    assert fallback == "0.20"
    assert "Provenance.default_thresh" not in doc


def test_package_doc_parser_pairs_each_name_with_its_own_number() -> None:
    """Bug caught: the parser credits a grouped or shared number to the wrong ids,
    so the association test above passes a swapped docstring."""
    swapped = (
        "rel_l1_thresh defaults to a per-variant value:\n"
        "    0.30 flux1-dev / flux1-schnell, 0.20 flux1-krea-dev, 0.17 flux2-klein-base-4b /\n"
        "    -base-9b. Distilled flux2-klein-4b / -9b have none and use 0.20.\n"
    )
    per_id, distilled, fallback = _package_doc_defaults(swapped)
    assert per_id == {
        "flux1-dev": "0.30",
        "flux1-schnell": "0.30",
        "flux1-krea-dev": "0.20",
        "flux2-klein-base-4b": "0.17",
        "flux2-klein-base-9b": "0.17",
    }
    assert distilled == ["flux2-klein-4b", "flux2-klein-9b"]
    assert fallback == "0.20"
    assert _package_doc_defaults("rel_l1_thresh defaults to 0.20 flux1-dev.")[1:] == ([], None)
