"""Every heavy script must derive its memory caps through _mlx_caps, never a
literal, and must not touch the deprecated mx.metal namespace (its deprecation
notice goes to stderr, invisible to filterwarnings=error)."""

import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RAW = re.compile(r"mx\.(set_wired_limit|set_memory_limit|set_cache_limit)\(")


def test_no_script_calls_the_raw_cap_apis_directly() -> None:
    # bug caught: a new script pasting `mx.set_wired_limit(int(20 * 1024**3))`
    offenders = sorted(
        p.name for p in SCRIPTS.glob("*.py") if p.name != "_mlx_caps.py" and RAW.search(p.read_text())
    )
    assert offenders == [], offenders


def test_no_script_uses_the_deprecated_metal_namespace() -> None:
    # bug caught: mx.metal.get_peak_memory (stderr deprecation, invisible to filterwarnings)
    offenders = sorted(p.name for p in SCRIPTS.glob("*.py") if "mx.metal." in p.read_text())
    assert offenders == [], offenders
