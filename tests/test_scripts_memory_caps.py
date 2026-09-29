"""Every heavy script must derive its memory caps through _mlx_caps, never a
literal, and must not touch the deprecated mx.metal namespace (its deprecation
notice goes to stderr, invisible to filterwarnings=error).

The caps and watchdog rules are checked on the parsed AST, not on substrings: a
comment or a docstring that mentions ``arm_mlx_watchdog(`` must not satisfy the
guard, and a raw cap call must be caught however the name was reached
(``mx.set_wired_limit``, ``mlx.core.set_wired_limit`` or a ``from mlx.core import``)."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CAPS_MODULE = "_mlx_caps.py"

RAW_CAP_NAMES = frozenset({"set_wired_limit", "set_memory_limit", "set_cache_limit"})
# mflux pipeline classes a heavy script constructs (directly or through `from_name`).
MODEL_CLASSES = frozenset({"Flux1", "Flux2Klein", "ZImage", "QwenImage"})
# Helper functions that load a model. Constructing inside one of these is a definition, not a load;
# the loading happens where the helper is called.
LOADER_FUNCTIONS = frozenset({"load_model", "_load_model", "_load_flux", "_load_klein"})


def _heavy_files() -> list[Path]:
    files = sorted(p for p in SCRIPTS.glob("*.py") if p.name not in (CAPS_MODULE, "_mlx_watchdog.py"))
    return [*files, ROOT / "tests" / "generate_references.py"]


def _callee_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


class _Scan(ast.NodeVisitor):
    def __init__(self) -> None:
        self.called: set[str] = set()
        self.loads_model = False
        self.raw_cap_uses: list[str] = []
        self._function_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:
        name = _callee_name(node)
        if name is not None:
            self.called.add(name)
            if name in RAW_CAP_NAMES:
                self.raw_cap_uses.append(f"call {name}")
            in_loader = bool(self._function_stack) and self._function_stack[-1] in LOADER_FUNCTIONS
            if name in LOADER_FUNCTIONS:
                self.loads_model = True
            elif not in_loader:
                receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
                if name in MODEL_CLASSES or (isinstance(receiver, ast.Name) and receiver.id in MODEL_CLASSES):
                    self.loads_model = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in ("mlx.core", "mlx"):
            for alias in node.names:
                if alias.name in RAW_CAP_NAMES:
                    self.raw_cap_uses.append(f"import {alias.name}")
        self.generic_visit(node)


def scan(source: str) -> _Scan:
    scanner = _Scan()
    scanner.visit(ast.parse(source))
    return scanner


def violations(name: str, source: str) -> list[str]:
    """Rule breaches in one file: raw cap use outside _mlx_caps, or a model load without both guards."""
    found = scan(source)
    problems: list[str] = []
    if name != CAPS_MODULE:
        problems += [f"{name}: raw cap API ({use})" for use in found.raw_cap_uses]
    if found.loads_model:
        problems += [
            f"{name}: loads a model without a real {guard}(...) call"
            for guard in ("install_caps", "arm_mlx_watchdog")
            if guard not in found.called
        ]
    return problems


# ----- the rule, shown against fixture sources -----

_LOADS_NO_CAPS = "def main():\n    flux = Flux1(quantize=4, model_config=None)\n"
_WATCHDOG_IN_COMMENT = (
    "def main():\n"
    "    # install_caps(wired_gb=1, soft_gb=1); arm_mlx_watchdog(on_abort=None)\n"
    '    """arm_mlx_watchdog( and install_caps( only in prose"""\n'
    "    flux = ZImage(quantize=4, model_config=None)\n"
)
_FROM_IMPORT_RAW = "from mlx.core import set_wired_limit\n\nset_wired_limit(1)\n"
_GUARDED = (
    "def main():\n"
    "    install_caps(wired_gb=1, soft_gb=1)\n"
    "    arm_mlx_watchdog(on_abort=None)\n"
    "    flux = Flux1.from_name('dev', quantize=4)\n"
)


def test_a_model_load_without_caps_or_watchdog_is_flagged() -> None:
    """Bug: a worker constructs an mflux model with no caps and no watchdog and the guard stays green."""
    assert violations("w.py", _LOADS_NO_CAPS) == [
        "w.py: loads a model without a real install_caps(...) call",
        "w.py: loads a model without a real arm_mlx_watchdog(...) call",
    ]


def test_guard_names_in_a_comment_or_docstring_do_not_satisfy_the_rule() -> None:
    """Bug: the substring check accepts `arm_mlx_watchdog(` written in a comment as if it were called."""
    assert len(violations("w.py", _WATCHDOG_IN_COMMENT)) == 2


def test_from_import_of_a_raw_cap_api_is_flagged() -> None:
    """Bug: `from mlx.core import set_wired_limit` slips past a regex that only looks for `mx.`."""
    assert sorted(violations("w.py", _FROM_IMPORT_RAW)) == [
        "w.py: raw cap API (call set_wired_limit)",
        "w.py: raw cap API (import set_wired_limit)",
    ]


def test_dotted_raw_cap_calls_are_flagged_for_every_receiver() -> None:
    """Bug: `mlx.core.set_wired_limit(...)` is missed because only the `mx` alias is matched."""
    assert violations("w.py", "import mlx.core\nmlx.core.set_wired_limit(1)\n") == [
        "w.py: raw cap API (call set_wired_limit)"
    ]
    assert violations("w.py", "import mlx.core as mx\nmx.set_cache_limit(1)\n") == [
        "w.py: raw cap API (call set_cache_limit)"
    ]


def test_the_caps_module_itself_may_call_the_raw_apis() -> None:
    """Bug: the exemption for _mlx_caps.py is dropped and install_caps cannot exist."""
    assert violations(CAPS_MODULE, "import mlx.core as mx\nmx.set_wired_limit(1)\n") == []


def test_a_fully_guarded_loader_passes() -> None:
    """Bug: the rule over-fires on the shape every shipped worker uses."""
    assert violations("w.py", _GUARDED) == []


def test_defining_a_loader_helper_is_not_loading() -> None:
    """Bug: a helper module that only defines load_model is demanded to arm a watchdog it cannot own."""
    helper = "def load_model(r):\n    return Flux1(quantize=4, model_config=None)\n"
    assert violations("_helper.py", helper) == []
    assert len(violations("caller.py", "def main():\n    load_model(1)\n")) == 2


# ----- the real repository -----


def test_no_heavy_file_uses_the_raw_cap_apis() -> None:
    # bug caught: a new script pasting `mx.set_wired_limit(int(20 * 1024**3))`
    offenders = []
    for path in _heavy_files():
        offenders += [v for v in violations(path.name, path.read_text()) if "raw cap API" in v]
    assert offenders == [], offenders


def test_every_file_that_loads_a_model_installs_caps_and_arms_the_watchdog() -> None:
    # bug caught: a model-loading worker with no caps or no ceiling; the wired cap prevents only the
    # wired-exhaustion panic, a paging storm needs the watchdog (tests/generate_references.py had neither)
    loaders = [p for p in _heavy_files() if scan(p.read_text()).loads_model]
    offenders = []
    for path in loaders:
        offenders += [v for v in violations(path.name, path.read_text()) if "loads a model" in v]
    assert len(loaders) >= 10, [p.name for p in loaders]
    assert "generate_references.py" in [p.name for p in loaders]
    assert offenders == [], offenders


def test_every_script_that_installs_caps_also_arms_the_watchdog() -> None:
    # bug caught: caps installed with no ceiling in a file the loader detection does not recognise
    installers = [p for p in _heavy_files() if "install_caps" in scan(p.read_text()).called]
    unguarded = [p.name for p in installers if "arm_mlx_watchdog" not in scan(p.read_text()).called]
    assert installers, "no file installs caps; the file list is wrong"
    assert unguarded == [], unguarded


def test_no_script_uses_the_deprecated_metal_namespace() -> None:
    # bug caught: mx.metal.get_peak_memory (stderr deprecation, invisible to filterwarnings)
    offenders = sorted(p.name for p in SCRIPTS.glob("*.py") if "mx.metal." in p.read_text())
    assert offenders == [], offenders


# ----- exit-code contract: 3 is PARTIAL only, the watchdog owns 4 -----

EXIT_CALLEES = frozenset({"exit", "_exit", "SystemExit"})
# Orchestrators that translate a worker's death; each must compare against the shared constant.
ORCHESTRATORS = (
    "bench_speedup.py",
    "bench_comparison.py",
    "bench_klein_base_vs_distilled.py",
    "sweep_threshold_qwen.py",
    "sweep_threshold_klein_base_4b.py",
    "calibrate_qwen.py",
    "calibrate_flux2.py",
    "calibrate_flux1.py",
)


def _is_int(node: ast.AST | None, value: int) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) is int and node.value == value


def literal_exit_offenders(source: str, value: int) -> list[int]:
    """Line numbers where ``value`` is passed as a literal to an exit call or returned."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and _callee_name(node) in EXIT_CALLEES:
            if any(_is_int(a, value) for a in node.args):
                lines.append(node.lineno)
        elif isinstance(node, ast.Return) and _is_int(node.value, value):
            lines.append(node.lineno)
    return lines


def assigned_int_names(source: str, value: int) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and _is_int(node.value, value):
            names += [t.id for t in node.targets if isinstance(t, ast.Name)]
    return names


def test_exit_scanner_flags_a_literal_three_and_accepts_a_named_constant() -> None:
    # bug caught: a scanner that never fires (the repository test below would then pass on anything)
    assert literal_exit_offenders("import sys\nsys.exit(3)\n", 3) == [2]
    assert literal_exit_offenders("def f():\n    return 3\n", 3) == [2]
    assert literal_exit_offenders("raise SystemExit(3)\n", 3) == [1]
    assert literal_exit_offenders("import os\nos._exit(3)\n", 3) == [2]
    assert literal_exit_offenders("import sys\nsys.exit(PARTIAL_EXIT_CODE)\n", 3) == []
    assert assigned_int_names("PARTIAL_EXIT_CODE = 3\n", 3) == ["PARTIAL_EXIT_CODE"]


def test_no_script_exits_with_a_literal_three() -> None:
    """Bug: a script hard-coding exit 3 for something other than PARTIAL, colliding with the resume loop."""
    offenders = {}
    for path in [*sorted(SCRIPTS.glob("*.py")), ROOT / "tests" / "generate_references.py"]:
        lines = literal_exit_offenders(path.read_text(), 3)
        if lines:
            offenders[path.name] = lines
    assert offenders == {}, offenders


def test_partial_is_three_and_never_four() -> None:
    """Bug: a script defining PARTIAL = 4, or a PARTIAL constant that is not 3."""
    for path in sorted(SCRIPTS.glob("*.py")):
        source = path.read_text()
        assert [n for n in assigned_int_names(source, 4) if "PARTIAL" in n.upper()] == [], path.name
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and "PARTIAL" in t.id.upper() for t in node.targets
            ):
                assert _is_int(node.value, 3), (path.name, node.lineno)


def test_orchestrators_compare_against_the_watchdog_constant() -> None:
    """Bug: an orchestrator that recognises a worker abort by a literal, so a changed code goes unnoticed."""
    missing = []
    for name in ORCHESTRATORS:
        names = {n.id for n in ast.walk(ast.parse((SCRIPTS / name).read_text())) if isinstance(n, ast.Name)}
        if "WATCHDOG_EXIT_CODE" not in names:
            missing.append(name)
    assert missing == [], missing
