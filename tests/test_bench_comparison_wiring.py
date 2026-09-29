"""Critical call sites in the comparison harness's generation worker must appear exactly once, so a
hand-edit or a refactor can't silently duplicate or drop the wiring these tests exist to guard.

The checks read the parsed AST: a call that survives only in a comment or a string is not wiring
(style of tests/test_calibrated_range_wiring.py)."""

import ast
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bench_comparison.py"


def _expr(source: str) -> str:
    """Normalised form of an expression, so quote style and spacing do not matter."""
    return ast.dump(ast.parse(source, mode="eval").body)


def _calls(source: str, callee: str) -> list[ast.Call]:
    """Every real call to ``callee`` (bare name or attribute), wherever it sits."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Name) and func.id == callee) or (
                isinstance(func, ast.Attribute) and func.attr == callee
            ):
                found.append(node)
    return found


def _keyword_values(call: ast.Call, name: str | None) -> list[ast.expr]:
    """Values passed as ``name=...``; ``name=None`` selects the ``**splat`` entries."""
    return [kw.value for kw in call.keywords if kw.arg == name]


def _count_calls_with_keyword(source: str, callee: str, keyword: str | None, value_expr: str) -> int:
    """How many calls to ``callee`` pass ``keyword=<value_expr>`` (``**<value_expr>`` when keyword is None)."""
    want = _expr(value_expr)
    return sum(
        any(ast.dump(v) == want for v in _keyword_values(call, keyword)) for call in _calls(source, callee)
    )


def _apply_teacache_splats(source: str) -> int:
    return _count_calls_with_keyword(source, "apply_teacache", None, "teacache_kwargs(recipe)")


def _generate_image_splats(source: str) -> int:
    return _count_calls_with_keyword(source, "generate_image", None, "generate_kwargs_for(recipe)")


def _probe_path_slugs(source: str) -> int:
    return sum(
        _count_calls_with_keyword(source, callee, "path_slug", 'f"{recipe.slug}/{name}"')
        for callee in ("_run_generation",)
    )


def _gate_trace_records(source: str) -> int:
    return sum(
        any(
            ast.dump(v) == _expr("gate_trace(handle.stats.last_generation.decisions, steps)")
            for v in _keyword_values(call, "gate_trace")
        )
        for call in ast.walk(ast.parse(source))
        if isinstance(call, ast.Call)
    )


def _main_parse_assignments(source: str) -> int:
    """`args = _parse_and_validate()` statements inside main()."""
    want = _expr("_parse_and_validate()")
    count = 0
    for fn in ast.walk(ast.parse(source)):
        if isinstance(fn, ast.FunctionDef) and fn.name == "main":
            for node in ast.walk(fn):
                if (
                    isinstance(node, ast.Assign)
                    and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["args"]
                    and ast.dump(node.value) == want
                ):
                    count += 1
    return count


def test_apply_teacache_passes_teacache_kwargs_exactly_once() -> None:
    # bug caught: a hand-edited call drops **teacache_kwargs(recipe), so a recipe's rel_l1_thresh override
    # (or a future teacache_kwargs field) silently never reaches condition B's apply_teacache.
    assert _apply_teacache_splats(SCRIPT.read_text()) == 1


def test_generate_image_forwards_generate_kwargs_for_exactly_once() -> None:
    # bug caught: generate_image is called without **generate_kwargs_for(recipe), so Z-Image/Qwen's negative
    # prompt never reaches generate_image even though precompute_prompt already encoded it.
    assert _generate_image_splats(SCRIPT.read_text()) == 1


def test_quality_probe_worker_nests_output_by_slug_and_name_exactly_once() -> None:
    # bug caught: the quality-probe worker's path_slug stops matching quality_probe_dir's <slug>/<name>
    # nesting, so record.json and the images it points at end up in different directories.
    assert _probe_path_slugs(SCRIPT.read_text()) == 1


def test_main_parses_through_the_validating_helper_exactly_once() -> None:
    # bug caught: main() goes back to a bare _build_parser().parse_args(), which switches off both probe-flag
    # refusals in real use while their helper tests stay green.
    assert _main_parse_assignments(SCRIPT.read_text()) == 1


def test_run_generation_records_the_gate_trace_exactly_once() -> None:
    # bug caught: the worker stops passing the committed decisions through gate_trace, so every probe record
    # carries gate_trace=None and a threshold-independent skip is untraceable again.
    assert _gate_trace_records(SCRIPT.read_text()) == 1


# ----- the checks themselves, against fixture sources -----

_COMMENT_ONLY = (
    "def worker(recipe, flux, name):\n"
    "    # apply_teacache(flux, **teacache_kwargs(recipe))\n"
    "    # flux.generate_image(**generate_kwargs_for(recipe))\n"
    "    # _run_generation(recipe, path_slug=f'{recipe.slug}/{name}')\n"
    '    """gate_trace=gate_trace(handle.stats.last_generation.decisions, steps)"""\n'
    "    apply_teacache(flux)\n"
    "    flux.generate_image(prompt='x')\n"
    "    _run_generation(recipe, path_slug=recipe.slug)\n"
    "    Result(gate_trace=None)\n"
    "def main():\n"
    "    # args = _parse_and_validate()\n"
    "    args = _build_parser().parse_args()\n"
)

_WIRED = (
    "def worker(recipe, flux, name):\n"
    "    apply_teacache(flux, **teacache_kwargs(recipe))\n"
    "    flux.generate_image(prompt='x', **generate_kwargs_for(recipe))\n"
    "    _run_generation(recipe, path_slug=f'{recipe.slug}/{name}')\n"
    "    Result(gate_trace=gate_trace(handle.stats.last_generation.decisions, steps))\n"
    "def main():\n"
    "    args = _parse_and_validate()\n"
)


def test_the_call_checks_ignore_text_that_is_only_a_comment_or_string() -> None:
    """Bug: a substring count is satisfied by a commented-out call, so dropping the real wiring stays green."""
    for check in (
        _apply_teacache_splats,
        _generate_image_splats,
        _probe_path_slugs,
        _gate_trace_records,
        _main_parse_assignments,
    ):
        assert check(_COMMENT_ONLY) == 0, check.__name__


def test_the_call_checks_count_real_wiring_once() -> None:
    """Bug: a check that never matches (wrong normalisation) would make the red case above meaningless."""
    for check in (
        _apply_teacache_splats,
        _generate_image_splats,
        _probe_path_slugs,
        _gate_trace_records,
        _main_parse_assignments,
    ):
        assert check(_WIRED) == 1, check.__name__


def test_a_duplicated_call_is_counted_twice() -> None:
    """Bug: the 'exactly once' promise is lost because a duplicated call is not seen."""
    doubled = _WIRED + "\napply_teacache(flux, **teacache_kwargs(recipe))\n"
    assert _apply_teacache_splats(doubled) == 2
