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


def test_every_script_that_installs_caps_also_arms_the_watchdog() -> None:
    # bug caught: a model-loading worker with caps but no ceiling — the wired cap
    # prevents only the wired-exhaustion panic; a paging storm needs the watchdog
    installs = sorted(
        p.name
        for p in SCRIPTS.glob("*.py")
        if p.name not in ("_mlx_caps.py", "_mlx_watchdog.py") and "install_caps(" in p.read_text()
    )
    unguarded = [name for name in installs if "arm_mlx_watchdog(" not in (SCRIPTS / name).read_text()]
    assert installs, "no script installs caps — the glob is wrong"
    assert unguarded == [], unguarded
