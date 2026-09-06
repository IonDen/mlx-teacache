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


# Fields whose presence or text depends on the interpreter, not on the code:
# positions; `type_comment`; `type_params` (new in 3.12, printed as `[]` by
# ast.dump on every FunctionDef); `kind` on Constant; `ctx` on Name/Attribute/
# Subscript. ast.dump itself is not used: 3.13 changed its defaults so that
# empty lists and None fields are omitted, which moved every digest.
_INTERPRETER_FIELDS = frozenset(
    {"lineno", "col_offset", "end_lineno", "end_col_offset", "type_comment", "type_params", "kind", "ctx"}
)


def _normalized(node: Any) -> str:
    if isinstance(node, ast.AST):
        parts = [type(node).__name__]
        for field in node._fields:
            if field in _INTERPRETER_FIELDS:
                continue
            value = getattr(node, field, None)
            if value is None or value == []:
                continue
            parts.append(f"{field}={_normalized(value)}")
        return "(" + ",".join(parts) + ")"
    if isinstance(node, list):
        return "[" + ",".join(_normalized(item) for item in node) + "]"
    return repr(node)


def fingerprint_function_node(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Short sha256 of a function node's arguments and body, normalised so the
    same source gives the same digest on every CPython from 3.10 to 3.14; the
    name, decorators, docstrings (strip them first), comments and formatting do
    not move it; any statement or argument change does."""
    dumped = _normalized(fn_node.args) + "|" + "|".join(_normalized(stmt) for stmt in fn_node.body)
    return hashlib.sha256(dumped.encode()).hexdigest()[:16]


def ast_fingerprint(fn: Callable[..., Any]) -> str:
    """`fingerprint_function_node` over a live function's source (docstrings removed)."""
    tree = _strip_docstrings(_source_ast(fn))
    fn_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef))
    return fingerprint_function_node(fn_node)
