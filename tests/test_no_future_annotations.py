"""Guard: `from __future__ import annotations` is a project anti-pattern (it
breaks runtime annotation introspection and is being phased out post-PEP 749;
see the workspace Python standards). Only the hatch-vcs-generated `_version.py`
is exempt. AST-level so a docstring/comment mention can't trip it (backlog 0028).
"""

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src" / "mlx_teacache"
_EXEMPT = {"_version.py"}


def test_no_future_annotations_import_in_src() -> None:
    offenders = []
    for py in sorted(_SRC.rglob("*.py")):
        if py.name in _EXEMPT:
            continue
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "__future__"
                and any(alias.name == "annotations" for alias in node.names)
            ):
                offenders.append(f"{py.relative_to(_SRC)}:{node.lineno}")
    assert not offenders, (
        "`from __future__ import annotations` is banned in src (only _version.py is "
        f"exempt); found in: {offenders}"
    )
