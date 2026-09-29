"""Text that ships inside the package (help(), tracebacks) must make sense to a user."""

import re
from pathlib import Path

import mlx_teacache
from mlx_teacache.errors import MissingGenerationContextError

TRACKER = re.compile(
    r"audit (medium|high|low|F\d)|\(audit\b|\bTask \d+|spec §|PORTED VERBATIM|per 00\d\d|Phase [A-Z]\b|v2\.5"
    r"|Byte-for-byte port|Ported from scripts/"
)


def test_no_tracker_words_in_shipped_source() -> None:
    """Bug caught: help() shows 'per audit medium #4' / 'Task 18' a user cannot resolve."""
    root = Path(mlx_teacache.__file__).parent
    hits = [
        f"{p.relative_to(root)}:{i}"
        for p in sorted(root.rglob("*.py"))
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if TRACKER.search(line)
    ]
    assert hits == []


def test_missing_context_message_is_family_neutral() -> None:
    """Bug caught: a FLUX.1 user is told 'FLUX.2 generation started'."""
    assert "FLUX.2" not in str(MissingGenerationContextError())
    msg = str(MissingGenerationContextError("callback gone"))
    assert "callback gone" in msg and "(detail:" not in msg
