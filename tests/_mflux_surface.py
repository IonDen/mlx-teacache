"""Weight-free introspection of the installed mflux: which attributes a class's
``__init__`` assigns, which tuple arities a function returns, and a
whitespace/comment/docstring-insensitive fingerprint of a function body.

The contract pins are built on these so a mflux minor that renames or rewrites
what the integrations touch goes red in CI without loading weights."""

import ast
import hashlib
import inspect
import json
import textwrap
from collections.abc import Callable, Iterable, Mapping
from importlib.metadata import PackageNotFoundError, distribution
from typing import Any

import pytest

from mlx_teacache._mflux_versions import release_tuple


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


def drift_reference(installed: str, known: Iterable[str]) -> str | None:
    """The verified release whose fingerprints an installed mflux is held to.

    Its own row when it has one; the newest row when it is a later release than every row (an
    unrecorded new release is judged on whether the copied functions moved, not on its number);
    None for an unrecorded release inside the verified span or a string with no release number."""
    versions = list(known)
    if installed in versions:
        return installed
    mine = release_tuple(installed)
    ranked = [(parsed, v) for v in versions if (parsed := release_tuple(v)) is not None]
    if mine is None or not ranked:
        return None
    newest_parsed, newest = max(ranked)
    return newest if mine > newest_parsed else None


def drift_failures(
    installed: str, fresh: Mapping[str, str], known: Mapping[str, Mapping[str, str]]
) -> list[str]:
    """One message per reason the installed mflux fails the drift guard; empty when it passes."""
    reference = drift_reference(installed, known)
    if reference is None:
        return [
            f"mflux {installed} has no verified row and is not newer than the newest one; "
            f"current fingerprints: {dict(fresh)}"
        ]
    return [
        f"{label} changed in mflux {installed} (verified {reference}: {expected}, now {fresh.get(label)}); "
        "diff it against the copy that feeds on it and re-verify before recording a row"
        for label, expected in known[reference].items()
        if fresh.get(label) != expected
    ]


def _is_self_predict_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_predict"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    )


_SELF_TRANSFORMER = ast.dump(ast.parse("self.transformer", mode="eval").body)


def unbound_predict_keywords(source: str, factory: Callable[..., Any]) -> list[str]:
    """What `factory` cannot take from the ``self._predict(...)`` calls in the ``generate_image`` of ``source``
    (a whole module, a class or the method alone); empty when every call binds.

    Each call's first positional argument must be ``self.transformer``, the module the installed factory gates.
    Each call is bound against ``inspect.signature(factory)`` with its own positional count and keyword names.
    A keyword that lands in a ``**kwargs`` catch-all is reported too: the installed factory refuses those at
    run time, so only keywords it names count as accepted. A ``generate_image`` with no such call is reported,
    and so is a ``**mapping`` splat, whose keys the source does not show."""
    tree = ast.parse(textwrap.dedent(source))
    generates = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == "generate_image"
    ]
    if not generates:
        return ["source has no generate_image"]
    calls = [n for fn in generates for n in ast.walk(fn) if _is_self_predict_call(n)]
    if not calls:
        return ["generate_image never calls self._predict"]
    signature = inspect.signature(factory)
    catch_all = [p.name for p in signature.parameters.values() if p.kind is inspect.Parameter.VAR_KEYWORD]
    problems: list[str] = []
    for call in calls:
        assert isinstance(call, ast.Call)
        if call.args and ast.dump(call.args[0]) != _SELF_TRANSFORMER:
            problems.append("the first argument is not self.transformer")
        names = [kw.arg for kw in call.keywords if kw.arg is not None]
        if len(names) != len(call.keywords):
            problems.append("a **mapping splat whose keywords cannot be checked")
        try:
            bound = signature.bind(*[None] * len(call.args), **dict.fromkeys(names))
        except TypeError as error:
            unknown = [name for name in names if name not in signature.parameters or name in catch_all]
            problems.extend(unknown or [f"{len(call.args)} positional arguments: {error}"])
            continue
        for name in catch_all:
            problems.extend(bound.arguments.get(name, {}))
    return problems


def installed_predict_factory(variant_id: str) -> Callable[..., Any]:
    """The object `apply_teacache` assigns to `flux._predict` for `variant_id`, taken from a duck-typed fake."""
    # Imported here: the pure-core lane imports this module without mflux and without the test fakes' MLX models.
    from mlx_teacache import apply_teacache
    from tests._variant_fakes import fake_for_variant

    fake = fake_for_variant(variant_id)
    handle = apply_teacache(fake)
    factory: Callable[..., Any] = vars(fake)["_predict"]
    handle.restore()
    return factory


def installed_vcs_commit(dist_name: str = "mflux") -> str | None:
    """The git commit an installed distribution was built from, read from its PEP 610
    ``direct_url.json``; None for a wheel or PyPI install, a missing file, or a non-git source."""
    try:
        text = distribution(dist_name).read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not text:
        return None
    try:
        vcs_info = json.loads(text).get("vcs_info", {})
        if vcs_info.get("vcs") != "git":
            return None
        commit = vcs_info.get("commit_id")
    except (ValueError, AttributeError):
        return None
    return commit if isinstance(commit, str) and commit else None


def require_or_skip(has_feature: bool, *, vcs_commit: str | None, feature: str) -> None:
    """Return when ``has_feature``; otherwise skip on a release (it predates the feature) but fail on a
    git install of mflux (``vcs_commit`` given), where a missing feature means main dropped or renamed
    something TeaCache was reviewed against and a skip would hide it."""
    if has_feature:
        return
    if vcs_commit is not None:
        pytest.fail(
            f"mflux main ({vcs_commit[:7]}) has no {feature}: it dropped or renamed the feature "
            "TeaCache was reviewed against"
        )
    pytest.skip(f"this mflux has no {feature}")


def drift_failures_for_install(
    installed: str,
    vcs_commit: str | None,
    fresh: Mapping[str, str],
    known: Mapping[str, Mapping[str, str]],
    reviewed_main: Mapping[str, Any],
) -> list[str]:
    """`drift_failures` for a wheel; for a git install (a commit is given), the installed fingerprints are
    compared with the reviewed-main row, because a main checkout keeps the last release's version string.

    A wheel newer than every recorded release may carry the already-reviewed main change (the release cut
    from that main), so it passes when it matches the newest row or the reviewed-main row, and otherwise
    fails with the differences against both. Recorded and in-span unrecorded versions keep the strict rule."""
    short = str(reviewed_main["commit"])[:7]
    main_fingerprints: Mapping[str, str] = reviewed_main["fingerprints"]

    def against_main(prefix: str, label: str, expected: str) -> str:
        return (
            f"{label} changed {prefix} (was {expected}, now {fresh.get(label)}); "
            "diff it against the copy that feeds on it and re-verify before recording a new reviewed commit"
        )

    if vcs_commit is not None:
        return [
            against_main(f"on mflux main (installed {vcs_commit[:7]}, reviewed {short})", label, expected)
            for label, expected in main_fingerprints.items()
            if fresh.get(label) != expected
        ]
    release = drift_failures(installed, fresh, known)
    if installed in known or drift_reference(installed, known) is None:
        return release
    main = [
        against_main(f"against reviewed main {short}", label, expected)
        for label, expected in main_fingerprints.items()
        if fresh.get(label) != expected
    ]
    return [] if not release or not main else release + main
