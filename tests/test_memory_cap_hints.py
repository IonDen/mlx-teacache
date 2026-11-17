"""Per-variant soft memory caps for the heavy scripts live in ``scripts/_memory_cap_hints.py``.

The table is script configuration, not library data: the variant configs no longer carry it, and the
three scripts that size their workers from it read it through ``memory_cap_hint_gb``.
``_memory_cap_hints`` imports nothing, so this module runs in the pure-core lane.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
VARIANTS = ROOT / "src" / "mlx_teacache" / "variants"

sys.path.insert(0, str(SCRIPTS))
import _memory_cap_hints as hints  # noqa: E402

HINT_KEY = "memory_cap_hint_gb"
# The scripts that size a worker's caps from the per-variant hint.
READERS = ("bench_speedup.py", "calibrate_flux2.py", "sweep_threshold_klein_base_4b.py")


def _string_constants(path: Path) -> list[int]:
    """Line numbers where the hint key appears as a string literal (a dict key or a lookup)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno for node in ast.walk(tree) if isinstance(node, ast.Constant) and node.value == HINT_KEY
    ]


def _calls(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    )


def test_hint_table_is_the_measured_caps() -> None:
    """Bug caught: a cap value drifts during the move (base-9b silently drops from 24 GB to 22 GB)."""
    assert hints.MEMORY_CAP_HINT_GB == {
        "flux2-klein-base-9b": 24,
        "z-image-base": 22,
        "qwen-image": 22,
        "qwen-image-2512": 22,
    }


def test_hinted_variants_return_their_cap() -> None:
    """Bug caught: the lookup ignores the table (returns None or the 22 GB default for base-9b)."""
    assert hints.memory_cap_hint_gb("flux2-klein-base-9b") == 24
    assert hints.memory_cap_hint_gb("z-image-base") == 22


def test_unknown_variant_has_no_hint() -> None:
    """Bug caught: the lookup invents a cap for an unhinted or unknown id, so the script's own default
    and its ``--cap-gb`` fallback path are never reached."""
    assert hints.memory_cap_hint_gb("flux2-klein-4b") is None
    assert hints.memory_cap_hint_gb("not-a-variant") is None


def test_every_hint_names_a_registered_variant() -> None:
    """Bug caught: a renamed variant id silently loses its cap and the script falls back to the default."""
    from mlx_teacache.variants import _REGISTRY

    assert sorted(set(hints.MEMORY_CAP_HINT_GB) - set(_REGISTRY)) == []


def test_no_variant_config_carries_the_hint() -> None:
    """Bug caught: two sources of truth, a config META still carrying the cap the scripts no longer read."""
    configs = sorted(VARIANTS.glob("*/config.py"))
    assert len(configs) == 10
    carrying = {p.parent.name: _string_constants(p) for p in configs if _string_constants(p)}
    assert carrying == {}


def test_no_script_reads_the_hint_from_the_registry() -> None:
    """Bug caught: a script still looks the hint up in a variant META, gets None, and runs base-9b
    under the 22 GB default."""
    readers = {name: _string_constants(SCRIPTS / name) for name in READERS}
    assert readers == {name: [] for name in READERS}


def test_each_reader_sizes_its_worker_from_the_table() -> None:
    """Bug caught: a reader drops the hint lookup and always uses its 22 GB default.

    The readers run inside model-loading workers, so this checks the call is present rather than
    running it."""
    assert {name: _calls(SCRIPTS / name, "memory_cap_hint_gb") for name in READERS} == {
        name: 1 for name in READERS
    }
