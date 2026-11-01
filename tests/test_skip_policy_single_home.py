"""Guard: the per-step skip policy lives only in `mlx_teacache._kernel`.

The threshold-zero fast path, the gating-signal shape check, the step-window check, the
gate-to-stats record, the residual write and the reconstruction of a skipped step are the
helpers of `_kernel/step.py`. A forward that re-inlines one of them keeps a private copy, so
the next change to the gate or the window rule misses it. AST-level, so a docstring or comment
naming a helper cannot trip it.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_teacache.integrations._session import InternalHandle
from mlx_teacache.integrations.mflux import lifecycle

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src" / "mlx_teacache"

_POLICY_CALLS = frozenset(
    {"InvalidStepWindowError", "TransformerShapeError", "StepDecision", "check_step_window"}
)
_RESIDUAL_ATTRS = frozenset({"cached_residual", "cached_residual_neg"})
# (file, form) pairs that are not a forward's skip policy. apply_teacache warns that a threshold of exactly 0.0
# disables caching; it decides nothing per step.
_ALLOWED = frozenset({("api.py", "rel_l1_thresh ==")})
_FORWARDS = (
    "integrations/mflux/forward_flux1.py",
    "integrations/mflux/forward_flux2.py",
    "integrations/mflux/forward_qwen.py",
    "variants/z_image_base/integration.py",
)


def _names_threshold(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "rel_l1_thresh") or (
        isinstance(node, ast.Attribute) and node.attr == "rel_l1_thresh"
    )


def _callee_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _fast_path_compares(node: ast.Compare) -> list[str]:
    """The fast-path test `thresh <= 0` in each spelling: `<=` or `==` with the threshold on the left, `>=` or `==`
    with it on the right, at any link of a chained comparison."""
    operands = [node.left, *node.comparators]
    forms: list[str] = []
    for left, op, right in zip(operands, node.ops, operands[1:], strict=False):
        if _names_threshold(left) and isinstance(op, ast.LtE | ast.Eq):
            forms.append(f"rel_l1_thresh {'<=' if isinstance(op, ast.LtE) else '=='}")
        elif _names_threshold(right) and isinstance(op, ast.GtE | ast.Eq):
            forms.append(f"{'>=' if isinstance(op, ast.GtE) else '=='} rel_l1_thresh")
    return forms


def _residual_writes(node: ast.Assign | ast.AnnAssign | ast.AugAssign) -> list[str]:
    """Residual attributes this statement writes, unless it only clears them (`= None`, or no value at all)."""
    if isinstance(node, ast.AnnAssign) and node.value is None:
        return []
    if (
        not isinstance(node, ast.AugAssign)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    ):
        return []
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [
        sub.attr
        for target in targets
        for sub in ast.walk(target)
        if isinstance(sub, ast.Attribute) and sub.attr in _RESIDUAL_ATTRS and isinstance(sub.ctx, ast.Store)
    ]


def skip_policy_violations(source: str) -> list[str]:
    """One `"<line>: <form>"` entry per skip-policy expression found in `source`."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Compare):
            found.extend(f"{node.lineno}: {form}" for form in _fast_path_compares(node))
        elif isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
            found.extend(f"{node.lineno}: write .{attr}" for attr in _residual_writes(node))
        elif isinstance(node, ast.Call):
            name = _callee_name(node)
            if name in _POLICY_CALLS or name == "store_residuals":
                found.append(f"{node.lineno}: call {name}")
        elif isinstance(node, ast.Attribute):
            if node.attr in _RESIDUAL_ATTRS and isinstance(node.ctx, ast.Load):
                found.append(f"{node.lineno}: read .{node.attr}")
        elif (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == "_step_decision_from_gate"
        ):
            found.append(f"{node.lineno}: def _step_decision_from_gate")
    return found


def _scanned_files() -> list[Path]:
    return sorted(p for p in _SRC.rglob("*.py") if p.relative_to(_SRC).parts[0] != "_kernel")


def test_skip_policy_lives_only_in_the_kernel() -> None:
    """Breaks: a forward re-inlines the fast-path compare, the shape raise, a StepDecision, the window check,
    a residual write (a call or an assignment) or read, or keeps its own `_step_decision_from_gate`."""
    violations: dict[str, list[str]] = {}
    for p in _scanned_files():
        rel = p.relative_to(_SRC).as_posix()
        found = [
            entry
            for entry in skip_policy_violations(p.read_text(encoding="utf-8"))
            if (rel, entry.split(": ", 1)[1]) not in _ALLOWED
        ]
        if found:
            violations[rel] = found
    assert violations == {}


def test_each_allowed_form_is_still_used() -> None:
    """Breaks: an allowed (file, form) pair outlives the line it was for, so a later fast-path copy of that form
    in that file slips through."""
    for rel, form in sorted(_ALLOWED):
        forms = [
            entry.split(": ", 1)[1]
            for entry in skip_policy_violations((_SRC / rel).read_text(encoding="utf-8"))
        ]
        assert forms.count(form) == 1, (rel, form)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("if handle.rel_l1_thresh <= 0.0:\n    pass\n", ["1: rel_l1_thresh <="]),
        ("ok = rel_l1_thresh <= 0.0\n", ["1: rel_l1_thresh <="]),
        ("raise InvalidStepWindowError(1, 2)\n", ["1: call InvalidStepWindowError"]),
        (
            "raise errors.TransformerShapeError(step_idx=0, expected=(1,), actual=(2,))\n",
            ["1: call TransformerShapeError"],
        ),
        ("stats.record(StepDecision(step_idx=0))\n", ["1: call StepDecision"]),
        ("check_step_window(3, skip_first_n_steps=1)\n", ["1: call check_step_window"]),
        ("mlx_teacache.check_step_window(3)\n", ["1: call check_step_window"]),
        ("state.store_residuals(pos=a - b)\n", ["1: call store_residuals"]),
        ("out = x + state.cached_residual\n", ["1: read .cached_residual"]),
        ("out = x + state.cached_residual_neg\n", ["1: read .cached_residual_neg"]),
        ("def _step_decision_from_gate(d):\n    return d\n", ["1: def _step_decision_from_gate"]),
    ],
)
def test_the_guard_flags_each_form(source: str, expected: list[str]) -> None:
    """Breaks: the scan stops recognising one form, so a re-inlined copy of it passes the guard."""
    assert skip_policy_violations(source) == expected


def test_the_guard_passes_the_helper_forms() -> None:
    """Breaks: the scan flags the helper calls the forwards are meant to make, or a write that clears a
    residual, so the guard can never be green."""
    source = (
        "if is_disabled(handle.rel_l1_thresh):\n"
        "    stats.record(fast_path_decision(state, timestep=t))\n"
        "if handle.rel_l1_thresh > 0.0:\n"
        "    pass\n"
        "check_signal_shape(state, mod_in)\n"
        "validate_window_once(state, active_num_steps=3, skip_first_n_steps=1, skip_last_n_steps=1)\n"
        "cache_residuals(state, decision, pos=(out, inp))\n"
        "body_out = reconstruct(inp, state, decision, branch='neg')\n"
        "state.cached_residual = None\n"
        "# state.store_residuals(pos=x) in a comment\n"
        "'''check_step_window(3) in a docstring'''\n"
    )
    assert skip_policy_violations(source) == []


def test_the_scan_covers_the_forwards() -> None:
    """Breaks: the scan root or the `_kernel` exclusion is wrong, so the forwards are never read and the guard
    passes vacuously."""
    scanned = {p.relative_to(_SRC).as_posix() for p in _scanned_files()}
    assert set(_FORWARDS) <= scanned
    assert "_kernel/step.py" not in scanned


def _handle() -> InternalHandle:
    return InternalHandle(
        rel_l1_thresh=0.2,
        coefficients=(0.0, 0.0, 0.0, 1.0, 0.0),
        skip_first_n_steps=1,
        skip_last_n_steps=1,
    )


def test_resolve_active_num_steps_prefers_the_generation_context() -> None:
    """Breaks: the forward re-derives the count from the config when the lifecycle already set it."""
    handle = _handle()
    handle._gen_ctx.active_num_steps = 7
    config = SimpleNamespace(num_inference_steps=20, init_time_step=5)
    assert lifecycle.resolve_active_num_steps(handle, config) == 7


def test_resolve_active_num_steps_falls_back_to_the_active_window() -> None:
    """Breaks: with no context the fallback uses the nominal schedule (20) instead of the img2img active window
    (20 - 5 = 15), so skip_last is measured from the wrong end."""
    handle = _handle()
    assert handle._gen_ctx.active_num_steps is None
    config = SimpleNamespace(num_inference_steps=20, init_time_step=5)
    assert lifecycle.resolve_active_num_steps(handle, config) == 15


def test_resolve_active_num_steps_keeps_a_zero_count() -> None:
    """Breaks: a context count of 0 (image_strength=1.0) is read as missing and replaced by the config."""
    handle = _handle()
    handle._gen_ctx.active_num_steps = 0
    config = SimpleNamespace(num_inference_steps=20, init_time_step=0)
    assert lifecycle.resolve_active_num_steps(handle, config) == 0


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("state.cached_residual = body_out - body_in\n", ["1: write .cached_residual"]),
        ("cache.cached_residual_neg = out\n", ["1: write .cached_residual_neg"]),
        ("state.cached_residual += delta\n", ["1: write .cached_residual"]),
        ("state.cached_residual: Any = out - inp\n", ["1: write .cached_residual"]),
        (
            "state.cached_residual, state.cached_residual_neg = pos, neg\n",
            ["1: write .cached_residual", "1: write .cached_residual_neg"],
        ),
        ("if handle.rel_l1_thresh == 0.0:\n    pass\n", ["1: rel_l1_thresh =="]),
        ("if 0.0 >= handle.rel_l1_thresh:\n    pass\n", ["1: >= rel_l1_thresh"]),
        ("off = 0 == rel_l1_thresh\n", ["1: == rel_l1_thresh"]),
        ("ok = 0.0 < rel_l1_thresh <= 1.0\n", ["1: rel_l1_thresh <="]),
    ],
)
def test_the_guard_flags_residual_writes_and_every_fast_path_compare(
    source: str, expected: list[str]
) -> None:
    """Breaks: a forward writes its own residual (`state.cached_residual = out - inp`), or spells the fast-path
    test as `== 0` or `0 >= thresh`, and the scan, which looked only at reads and at `thresh <=`, passes it."""
    assert skip_policy_violations(source) == expected


def test_the_guard_passes_clearing_writes_and_positive_compares() -> None:
    """Breaks: the write and compare checks also flag clearing a residual to None, an annotation without a value,
    or the gate's own `0 < thresh` form, so the guard can never be green."""
    source = (
        "state.cached_residual = None\n"
        "state.cached_residual_neg: Any = None\n"
        "state.cached_residual: Any\n"
        "if 0.0 < handle.rel_l1_thresh:\n"
        "    pass\n"
        "if rel_l1_thresh is None:\n"
        "    pass\n"
    )
    assert skip_policy_violations(source) == []
