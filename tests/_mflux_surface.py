"""Weight-free introspection of the installed mflux: which attributes a class's
``__init__`` assigns, which tuple arities a function returns, and a
whitespace/comment/docstring-insensitive fingerprint of a function body.

The contract pins are built on these so a mflux minor that renames or rewrites
what the integrations touch goes red in CI without loading weights."""

import ast
import hashlib
import inspect
import textwrap
from collections.abc import Callable
from typing import Any


def _source_ast(obj: Any) -> ast.AST:
    return ast.parse(textwrap.dedent(inspect.getsource(obj)))


def assigned_attributes(cls: type) -> frozenset[str]:
    """Names assigned as ``self.<name> = ...`` anywhere inside ``cls.__init__``."""
    tree = _source_ast(cls.__init__)
    names: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            targets = [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                names.add(target.attr)
    return frozenset(names)


def return_tuple_arities(fn: Callable[..., Any]) -> frozenset[int]:
    """Lengths of every literal tuple ``fn`` returns (a bare value counts as 1)."""
    arities: set[int] = set()
    for node in ast.walk(_source_ast(fn)):
        if isinstance(node, ast.Return) and node.value is not None:
            arities.add(len(node.value.elts) if isinstance(node.value, ast.Tuple) else 1)
    return frozenset(arities)


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            del body[0]
    return tree


def ast_fingerprint(fn: Callable[..., Any]) -> str:
    """Short sha256 of the function's signature and body AST with docstrings
    removed: the name, decorators, comments, blank lines and formatting do not
    move it; any statement or argument change does."""
    tree = _strip_docstrings(_source_ast(fn))
    fn_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef))
    dumped = (
        ast.dump(fn_node.args, include_attributes=False)
        + "|"
        + "|".join(ast.dump(stmt, include_attributes=False) for stmt in fn_node.body)
    )
    return hashlib.sha256(dumped.encode()).hexdigest()[:16]
