import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_RELEASE_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release.yml"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# The single source of truth for "which Pythons do we claim to support".
# Every surface below must agree with it, so a widened floor can never ship
# untested (the CI matrix entry IS the test for a version).
SUPPORTED_PYTHONS = ("3.11", "3.12", "3.13")


def _pyproject() -> dict:
    return tomllib.loads(_PYPROJECT.read_text())


def _matrix_pythons(workflow: str) -> set[str]:
    """Versions in the workflow's single bracketed python-version matrix line."""
    live_lines = [line.split("#", 1)[0] for line in workflow.splitlines()]
    matrix_lines = [line for line in live_lines if "python-version:" in line and "[" in line]

    assert len(matrix_lines) == 1, matrix_lines
    return {part.strip().strip('"') for part in matrix_lines[0].split("[")[1].rstrip("]").split(",")}


def test_requires_python_floor_matches_lowest_supported_version() -> None:
    assert _pyproject()["project"]["requires-python"] == f">={SUPPORTED_PYTHONS[0]}"


def test_classifiers_list_every_supported_version() -> None:
    classifiers = _pyproject()["project"]["classifiers"]
    declared = {
        c.rsplit(" :: ", 1)[1]
        for c in classifiers
        if c.startswith("Programming Language :: Python :: ") and c[-1].isdigit() and "." in c
    }

    assert declared == set(SUPPORTED_PYTHONS)


def test_ci_test_matrix_covers_every_supported_version() -> None:
    """A version we advertise but never run is an untested claim."""
    assert _matrix_pythons(_CI_WORKFLOW.read_text()) == set(SUPPORTED_PYTHONS)


def test_matrix_parser_ignores_commented_out_matrix_lines() -> None:
    """A stale comment holding the old bracketed list must not satisfy the matrix
    check while the live matrix has been removed or squashed to one version."""
    gutted_workflow = (
        "jobs:\n"
        "  test-mflux:\n"
        "    strategy:\n"
        "      matrix:\n"
        '        # python-version: ["3.10", "3.11", "3.12", "3.13"]\n'
        '        python-version: ["3.13"]\n'
    )

    assert _matrix_pythons(gutted_workflow) == {"3.13"}


def test_mypy_targets_the_lowest_supported_version() -> None:
    """Type-checking the floor is what catches typing that only works on newer Pythons."""
    assert _pyproject()["tool"]["mypy"]["python_version"] == SUPPORTED_PYTHONS[0]


def test_ruff_targets_the_lowest_supported_version() -> None:
    """Linting against the floor is what flags syntax/stdlib use newer than we support."""
    floor = SUPPORTED_PYTHONS[0]
    assert _pyproject()["tool"]["ruff"]["target-version"] == f"py{floor.replace('.', '')}"


def test_ruff_format_leaves_markdown_alone() -> None:
    """Bug: someone drops the *.md format exclude and the next ruff bump (0.16+ formats Markdown code blocks)
    rewrites the Python snippets in docs/papers, which is edited by its own process."""
    assert "*.md" in _pyproject()["tool"]["ruff"].get("format", {}).get("exclude", [])


def test_readme_python_badge_states_the_floor_as_a_range() -> None:
    """The badge is hand-written (shields' pyversions enumerates every version), so it
    must be pinned to the declared floor or it will quietly advertise the wrong one."""
    readme = (_REPO_ROOT / "README.md").read_text()
    floor = SUPPORTED_PYTHONS[0]

    assert f"badge/python-{floor}%2B-" in readme, (
        f"README needs a python-{floor}%2B badge matching requires-python"
    )
    assert "pypi/pyversions" not in readme, "the enumerating pyversions badge was replaced by the floor badge"


def test_readme_requires_line_states_the_floor() -> None:
    readme = (_REPO_ROOT / "README.md").read_text()

    assert f"Requires Python ≥ {SUPPORTED_PYTHONS[0]}" in readme


# The mflux requirement lives in five places. pyproject's [mflux] extra is the source of truth;
# the test-mflux group, the CI job that installs the newest mflux, and the README must quote the
# same spec. The extra has a floor and no upper bound: a newer mflux installs, runs and warns once.
# The spec pattern also matches a `,<` bound so a leftover ceiling on any surface shows up as a
# mismatch instead of being skipped. CHANGELOG / ROADMAP ranges are history and exempt.
_MFLUX_SPEC = r">=\d+(?:\.\d+)*(?:,<\d+(?:\.\d+)*)?"
_FLOOR_ONLY = re.compile(r">=(\d+(?:\.\d+)*)")


def _mflux_extra_range() -> str:
    (spec,) = _pyproject()["project"]["optional-dependencies"]["mflux"]
    assert spec.startswith("mflux"), spec
    return spec.removeprefix("mflux")


def _group_mflux_specs() -> set[str]:
    group = _pyproject()["dependency-groups"]["test-mflux"]
    return {s.removeprefix("mflux") for s in group if isinstance(s, str) and s.startswith("mflux")}


def _ci_mflux_specs() -> set[str]:
    live = "\n".join(line.split("#", 1)[0] for line in _CI_WORKFLOW.read_text().splitlines())
    return {
        m.group(0)
        for m in re.finditer(_MFLUX_SPEC, live)
        if "mflux" in live[max(0, m.start() - 6) : m.start()]
    }


def _readme_mflux_specs() -> set[str]:
    readme = (_REPO_ROOT / "README.md").read_text()
    return set(re.findall(rf"`(?:mflux)?({_MFLUX_SPEC})`", readme))


def test_test_mflux_group_pins_the_same_mflux_range_as_the_extra() -> None:
    assert _group_mflux_specs() == {_mflux_extra_range()}


def test_ci_newest_mflux_job_installs_the_extra_range() -> None:
    assert _ci_mflux_specs() == {_mflux_extra_range()}


def test_ci_runs_daily_on_a_schedule() -> None:
    """Bug: the schedule trigger is dropped, so the newest-mflux and mflux-main jobs only run on a push or a
    pull request and a new mflux release goes unnoticed until someone happens to push."""
    live = [line.split("#", 1)[0].rstrip() for line in _CI_WORKFLOW.read_text().splitlines()]
    on_block = live[live.index("on:") + 1 : live.index("permissions:")]
    assert "  schedule:" in on_block
    assert any(re.fullmatch(r'\s+- cron: "\d+ \d+ \* \* \*"', line) for line in on_block)


def test_readme_quotes_only_the_extra_mflux_range() -> None:
    assert _readme_mflux_specs() == {_mflux_extra_range()}


def test_mflux_requirement_is_a_verified_floor_with_no_upper_bound() -> None:
    """Bug: a `,<0.26` ceiling is left on (or comes back to) one surface, the [mflux] extra, the test-mflux group,
    the CI newest-mflux install or the README, so pip refuses the next mflux release and the daily newest-mflux job
    never tests it. Also catches a floor raised past NEWEST_VERIFIED_MFLUX (`mflux>=0.25` while 0.24.0 is the newest
    verified), which would admit only releases this version was never verified on."""
    from mlx_teacache._mflux_versions import NEWEST_VERIFIED_MFLUX, release_tuple

    surfaces = {
        "[mflux] extra": {_mflux_extra_range()},
        "test-mflux group": _group_mflux_specs(),
        "ci.yml": _ci_mflux_specs(),
        "README": _readme_mflux_specs(),
    }
    for surface, specs in surfaces.items():
        assert specs, f"{surface} quotes no mflux requirement"
        for spec in specs:
            assert _FLOOR_ONLY.fullmatch(spec), f"{surface}: {spec!r} is not a bare >= floor"
    match = _FLOOR_ONLY.fullmatch(_mflux_extra_range())
    assert match is not None
    floor, verified = release_tuple(match.group(1)), release_tuple(NEWEST_VERIFIED_MFLUX)
    assert floor is not None and verified is not None
    assert floor <= verified


# `[mflux]` is the extra's name on another requirement (`mlx-teacache[mflux]<0.15`), so its bound is not on mflux;
# `mflux-teacache` and `mflux_x` are other names. A ceiling is `<` or `<=`, with or without a space before the digit.
_MFLUX_CEILING = re.compile(r"(?<!\[)mflux(?![\]\w-])[^\n]{0,20}<=?\s*\d")


def _mflux_ceiling_lines(lines: list[str]) -> list[str]:
    """Lines that put an upper bound on mflux, skipping `[mflux]` used as an extra's name."""
    return [line for line in lines if _MFLUX_CEILING.search(line)]


@pytest.mark.parametrize(
    ("line", "flagged"),
    [
        ('uv pip install "mflux<0.26"', True),
        ('uv pip install "mflux>=0.24" "mflux<0.26"', True),
        ("mflux 0.24 up to <0.26", True),
        ("mflux" + "." * 20 + "<0.26", True),
        ("mflux" + "." * 21 + "<0.26", False),
        ('stay on the 0.14 line: `pip install "mlx-teacache[mflux]<0.15"`', False),
        ('uv pip install "mflux>=0.24"', False),
        ("mflux >= 0.24, < 0.26", True),
        ("mflux < 0.26", True),
        ("mflux<=0.25", True),
        ("mflux-teacache<0.2", False),
    ],
)
def test_ceiling_scan_flags_mflux_bounds_not_the_extra_name(line: str, flagged: bool) -> None:
    """Bug: the ceiling scan skips a separate `mflux<X` constraint (so a ceiling hides beside the bare floor), or it
    flags `mlx-teacache[mflux]<0.15`, where the bound is on mlx-teacache and `mflux` is only the extra's name."""
    assert _mflux_ceiling_lines([line]) == ([line] if flagged else [])


def test_no_live_ci_line_or_readme_line_states_an_mflux_ceiling() -> None:
    """Bug: a ceiling comes back as its own constraint (`"mflux<0.26"` next to `"mflux>=0.24"` in ci.yml) or as
    README prose (`mflux 0.24 up to <0.26`), which the spec-shaped surface check above does not read."""
    ci_live = [line.split("#", 1)[0] for line in _CI_WORKFLOW.read_text().splitlines()]
    readme = (_REPO_ROOT / "README.md").read_text().splitlines()
    assert _mflux_ceiling_lines(ci_live) == []
    assert _mflux_ceiling_lines(readme) == []


def test_ci_floor_job_installs_and_asserts_the_extra_floor() -> None:
    """Bug: the extra's floor moves while test-mflux-floor still pins and asserts the old one (`mflux==0.17.5`
    under `mflux>=0.24`), so the advertised floor is never tested."""
    from mlx_teacache._mflux_versions import release_tuple

    block = [ln.strip() for ln in _job_block(_CI_WORKFLOW.read_text(), "test-mflux-floor")]
    pins = [m.group(1) for ln in block for m in re.finditer(r'"mflux==(\d+(?:\.\d+)*)"', ln)]
    asserted = [m.group(1) for ln in block for m in re.finditer(r"v\.startswith\('(\d+(?:\.\d+)*)'\)", ln)]
    floor = _FLOOR_ONLY.fullmatch(_mflux_extra_range())
    assert floor is not None
    assert len(pins) == 1 and asserted == pins
    assert release_tuple(pins[0]) == release_tuple(floor.group(1))


def _job_block(workflow: str, job: str) -> list[str]:
    """Comment-stripped lines of one top-level job (two-space indent) in the workflow."""
    lines = [line.split("#", 1)[0].rstrip() for line in workflow.splitlines()]
    start = lines.index(f"  {job}:")
    block: list[str] = []
    for line in lines[start + 1 :]:
        if re.match(r"^  \S", line) or re.match(r"^\S", line):
            break
        block.append(line)
    return block


def test_parity_job_raises_the_wall_backstop_to_twelve_hours() -> None:
    """Bug: the dispatch-only parity job running under the 3 h default wall backstop; the full
    parity lane takes ~9 h, so the backstop would abort it partway with exit code 4."""
    block = _job_block(_CI_WORKFLOW.read_text(), "test-parity")
    env_values = [line.strip() for line in block if line.strip().startswith("PYTEST_PARITY_WALL_S:")]
    assert env_values == ['PYTEST_PARITY_WALL_S: "43200"']


def _jobs_installing_outside_the_lock(workflow: str) -> list[str]:
    """Top-level job ids whose comment-stripped block runs `uv pip install`."""
    lines = [line.split("#", 1)[0].rstrip() for line in workflow.splitlines()]
    jobs = lines[lines.index("jobs:") + 1 :]
    ids = [m.group(1) for line in jobs if (m := re.fullmatch(r"  ([\w-]+):", line))]
    return [job for job in ids if any("uv pip install" in line for line in _job_block(workflow, job))]


def test_unlocked_mflux_jobs_carry_a_timeout_below_the_six_hour_default() -> None:
    """Bug: a job that installs mflux outside the lock (`uv pip install`) has no job-level timeout-minutes, so a
    broken or hostile wheel that hangs keeps the runner busy for GitHub's 360-minute default. The job list is read
    from the workflow, so a new unlocked job (or test-mflux-floor losing its timeout) cannot slip past a fixed list;
    the non-empty check catches the parser silently finding no job at all."""
    workflow = _CI_WORKFLOW.read_text()
    unlocked = _jobs_installing_outside_the_lock(workflow)
    assert unlocked, "no job runs `uv pip install`; the job parser found nothing"
    for job in unlocked:
        block = _job_block(workflow, job)
        values = [
            int(m.group(1)) for line in block if (m := re.fullmatch(r"    timeout-minutes: (\d+)", line))
        ]
        assert len(values) == 1, f"{job} job-level timeout-minutes: {values}"
        assert 0 < values[0] < 360, f"{job} timeout-minutes: {values[0]}"


_QWEN_PARITY_FILE = "tests/test_parity_qwen.py"


def _parity_pytest_commands() -> list[str]:
    block = [ln.strip() for ln in _job_block(_CI_WORKFLOW.read_text(), "test-parity")]
    return [ln.removeprefix("run:").strip() for ln in block if ln.startswith("run:") and "pytest" in ln]


def test_parity_job_never_selects_both_qwen_checkpoints_in_one_pytest_process() -> None:
    """Bug: the dispatch-only parity job runs `pytest -m parity` over the whole tree, which selects both Qwen
    checkpoints; the collection guard (one model per process) then exits 2 before a single test runs, or, without
    it, two 20B models share one process. Every command must either leave the Qwen file out or pick one id."""
    commands = _parity_pytest_commands()
    qwen_ids: list[str] = []
    for command in commands:
        if f"--ignore={_QWEN_PARITY_FILE}" in command:
            continue
        ids = re.findall(r"-k (original|2512)(?:\s|$)", command)
        assert _QWEN_PARITY_FILE in command and len(ids) == 1, command
        qwen_ids.extend(ids)
    assert sorted(qwen_ids) == ["2512", "original"], commands
    assert any(f"--ignore={_QWEN_PARITY_FILE}" in command for command in commands), commands


def test_parity_job_runs_every_split_step_and_still_fails_on_any_failure() -> None:
    """Bug: splitting the parity run into steps lets a red first step skip the Qwen checkpoints (default step
    gating), or `continue-on-error` turns a red step green, so the job hides a parity failure."""
    block = [ln.strip() for ln in _job_block(_CI_WORKFLOW.read_text(), "test-parity")]
    commands = _parity_pytest_commands()
    assert len(commands) == 3
    assert not any(ln.startswith("continue-on-error") for ln in block)
    assert block.count("if: ${{ !cancelled() }}") == len(commands) - 1


def test_coverage_job_gates_both_lanes_at_a_floor_of_at_least_ninety() -> None:
    """Bug: the combined-coverage FLOOR set back to 58 (or the exit-on-floor line removed, or one
    test job dropped from `needs`), so a real loss of stub-testable coverage passes CI."""
    block = [ln.strip() for ln in _job_block(_CI_WORKFLOW.read_text(), "coverage")]
    floors = [float(m.group(1)) for ln in block if (m := re.fullmatch(r"FLOOR = (\d+(?:\.\d+)?)", ln))]
    (needs_line,) = [ln for ln in block if ln.startswith("needs:")]
    needs = {part.strip() for part in needs_line.removeprefix("needs:").strip().strip("[]").split(",")}

    assert len(floors) == 1, floors
    assert floors[0] >= 90
    assert "sys.exit(0 if pct >= FLOOR else 1)" in block
    assert {"test-pure-core", "test-mflux"} <= needs


# ---- Workflow hardening: pinned actions, least privilege, locked installs ----

_WORKFLOWS = (_CI_WORKFLOW, _RELEASE_WORKFLOW)
_SHA_REF = re.compile(r"@[0-9a-f]{40}$")


def _live_lines(path: Path) -> list[str]:
    """Comment-stripped, right-trimmed lines of a workflow file."""
    return [line.split("#", 1)[0].rstrip() for line in path.read_text().splitlines()]


def _step_block(lines: list[str], index: int) -> list[str]:
    """The whole workflow step containing ``lines[index]`` (a ``uses:`` line, with or without the dash)."""
    start = index
    while not lines[start].lstrip().startswith("- "):
        start -= 1
    indent = len(lines[start]) - len(lines[start].lstrip())
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        block.append(line)
    return block


@pytest.mark.parametrize("workflow", _WORKFLOWS, ids=lambda p: p.name)
def test_every_action_is_pinned_to_a_full_commit_sha(workflow: Path) -> None:
    """Bug: a `uses:` left on a movable tag or branch, so a retagged upstream action runs in our jobs
    (and, in the release workflow, next to the PyPI publish token)."""
    refs = [m.group(1) for line in _live_lines(workflow) if (m := re.match(r"^\s*-?\s*uses:\s*(\S+)$", line))]
    assert refs, workflow.name
    unpinned = [ref for ref in refs if not _SHA_REF.search(ref)]
    assert unpinned == []


@pytest.mark.parametrize("workflow", _WORKFLOWS, ids=lambda p: p.name)
def test_workflow_default_token_is_read_only(workflow: Path) -> None:
    """Bug: no top-level `permissions:` block, so every job gets the repository's default token scope."""
    lines = _live_lines(workflow)
    start = lines.index("permissions:")
    body = []
    for line in lines[start + 1 :]:
        if line.strip() and not line.startswith(" "):
            break
        if line.strip():
            body.append(line.strip())
    assert body == ["contents: read"]


@pytest.mark.parametrize("workflow", _WORKFLOWS, ids=lambda p: p.name)
def test_every_checkout_drops_the_persisted_token(workflow: Path) -> None:
    """Bug: `actions/checkout` leaving the token in .git/config, readable by every later step."""
    lines = _live_lines(workflow)
    checkouts = [i for i, line in enumerate(lines) if "uses: actions/checkout@" in line]
    assert checkouts, workflow.name
    for i in checkouts:
        assert "persist-credentials: false" in [ln.strip() for ln in _step_block(lines, i)], i


@pytest.mark.parametrize("workflow", _WORKFLOWS, ids=lambda p: p.name)
def test_no_third_party_release_action(workflow: Path) -> None:
    """Bug: the GitHub release created by a third-party action that holds `contents: write`."""
    assert not [line for line in _live_lines(workflow) if "softprops/" in line]


def test_every_ci_sync_is_locked() -> None:
    """Bug: `uv sync` without `--locked` re-resolving dependencies in CI, so a passing job no longer
    tests the versions uv.lock records. The two jobs that float mflux sync from the lock too, and only
    the explicit `uv pip install` of mflux afterwards moves off it."""
    lines = _live_lines(_CI_WORKFLOW)
    current_job = ""
    unlocked: list[str] = []
    for line in lines:
        job = re.match(r"^  (\S+):$", line)
        if job:
            current_job = job.group(1)
        if "uv sync" in line and "--locked" not in line:
            unlocked.append(f"{current_job}: {line.strip()}")
    assert unlocked == []
    assert any("uv sync" in ln for ln in lines)


# ---- Release workflow: verified tag, pinned toolchain, gh-created release ----


def _release_job(job: str) -> list[str]:
    return _job_block(_RELEASE_WORKFLOW.read_text(), job)


def _needs(job: str) -> set[str]:
    (line,) = [ln.strip() for ln in _release_job(job) if ln.strip().startswith("needs:")]
    return {part.strip() for part in line.removeprefix("needs:").strip().strip("[]").split(",")}


def test_release_installs_uv_through_the_pinned_setup_action() -> None:
    """Bug: the release build running whatever `pip install uv` resolves that day, so the sdist and
    wheel are built by an unpinned toolchain."""
    lines = _live_lines(_RELEASE_WORKFLOW)
    (idx,) = [i for i, ln in enumerate(lines) if "uses: astral-sh/setup-uv@" in ln]
    step = [ln.strip() for ln in _step_block(lines, idx)]
    assert 'version: "0.11.2"' in step
    assert not [ln for ln in lines if "pip install uv" in ln]


def test_build_backend_requirements_are_exactly_pinned() -> None:
    """Bug: `[build-system].requires` floating, so the release artefacts depend on the newest
    hatchling / hatch-vcs at tag time."""
    requires = _pyproject()["build-system"]["requires"]
    assert sorted(requires) == ["hatch-vcs==0.5.0", "hatchling==1.32.4"]


def test_publish_and_github_release_wait_for_verify_and_build() -> None:
    """Bug: a tag that is not on main, or whose CI is red, still reaching PyPI."""
    assert {"verify", "build"} <= _needs("publish")
    assert {"verify", "build"} <= _needs("github-release")


def test_verify_job_requires_main_ancestry_and_the_ci_workflow_run_with_read_only_scopes() -> None:
    """Bug: a tag pushed from an unmerged branch, or on a commit whose CI run is red, publishing anyway;
    or the gate keyed on check-run names, so an unrelated bot check fails a good release."""
    block = [ln.strip() for ln in _release_job("verify")]
    assert "fetch-depth: 0" in block
    assert any('git merge-base --is-ancestor "$GITHUB_SHA" origin/main' in ln for ln in block)
    assert "run: bash .github/scripts/verify_ci_green.sh" in block
    assert "GH_TOKEN: ${{ github.token }}" in block
    assert "actions: read" in block
    assert "contents: read" in block
    assert "contents: write" not in block
    assert "check-runs" not in _RELEASE_WORKFLOW.read_text()


_VERIFY_SCRIPT = _REPO_ROOT / ".github" / "scripts" / "verify_ci_green.sh"


def _run_gate(
    tmp_path: Path, responses: list[dict | str], timeout_s: int = 5
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run the gate script against a fake `gh` that replays ``responses`` (the last one repeats).

    A dict is printed as the JSON body of a successful call; a str makes that call fail, printing the
    str on stderr and exiting 1, the way `gh api` reports an HTTP or network error."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for i, body in enumerate(responses, start=1):
        if isinstance(body, str):
            (tmp_path / f"resp{i}.fail").write_text(body)
        else:
            (tmp_path / f"resp{i}.json").write_text(json.dumps(body))
    fake = bin_dir / "gh"
    fake.write_text(
        "#!/bin/bash\n"
        f'D="{tmp_path}"\n'
        'n=$(cat "$D/count" 2>/dev/null || echo 0); n=$((n+1)); echo $n > "$D/count"\n'
        'echo "$@" >> "$D/args"\n'
        f'[ "$n" -gt {len(responses)} ] && n={len(responses)}\n'
        'if [ -f "$D/resp$n.fail" ]; then cat "$D/resp$n.fail" >&2; exit 1; fi\n'
        'cat "$D/resp$n.json"\n'
    )
    fake.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_REPOSITORY": "octo/repo",
        "GITHUB_SHA": "abc123",
        "VERIFY_POLL_SECONDS": "0",
        "VERIFY_TIMEOUT_SECONDS": str(timeout_s),
    }
    result = subprocess.run(
        ["bash", str(_VERIFY_SCRIPT)], env=env, capture_output=True, text=True, check=False, timeout=30
    )
    calls = (tmp_path / "args").read_text().splitlines() if (tmp_path / "args").exists() else []
    return result, calls


def _run(status: str, conclusion: str | None, created: str = "2026-09-29T10:00:00Z") -> dict:
    return {"status": status, "conclusion": conclusion, "created_at": created}


def test_gate_passes_on_a_completed_successful_ci_push_run(tmp_path: Path) -> None:
    """Bug: the gate rejecting (or never reading) a green CI run; also pins the endpoint it reads."""
    result, calls = _run_gate(tmp_path, [{"workflow_runs": [_run("completed", "success")]}])
    assert result.returncode == 0, result.stderr + result.stdout
    assert calls == ["api repos/octo/repo/actions/workflows/ci.yml/runs?head_sha=abc123&event=push"]


def test_gate_fails_at_once_when_the_run_completes_red(tmp_path: Path) -> None:
    """Bug: waiting out the timeout (or passing) after the CI run already failed."""
    result, calls = _run_gate(tmp_path, [{"workflow_runs": [_run("completed", "failure")]}])
    assert result.returncode != 0
    assert "failure" in result.stdout + result.stderr
    assert len(calls) == 1


def test_gate_waits_while_the_run_is_in_progress_then_passes(tmp_path: Path) -> None:
    """Bug: failing a release because CI was still running when the tag was pushed."""
    result, calls = _run_gate(
        tmp_path,
        [
            {"workflow_runs": []},
            {"workflow_runs": [_run("in_progress", None)]},
            {"workflow_runs": [_run("completed", "success")]},
        ],
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert len(calls) == 3


def test_gate_times_out_with_a_clear_message_when_no_run_appears(tmp_path: Path) -> None:
    """Bug: an unbounded wait, or a silent pass, when CI never ran for the tagged commit."""
    result, calls = _run_gate(tmp_path, [{"workflow_runs": []}], timeout_s=0)
    assert result.returncode != 0
    assert "timed out" in (result.stdout + result.stderr).lower()
    assert len(calls) == 1


def test_gate_retries_after_a_failed_api_call_then_passes(tmp_path: Path) -> None:
    """Bug: one transient `gh api` error (a 502, a dropped connection) killing the release under
    `set -e` although CI is green."""
    result, calls = _run_gate(
        tmp_path, ["HTTP 502: Bad Gateway", {"workflow_runs": [_run("completed", "success")]}]
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert len(calls) == 2
    assert "HTTP 502: Bad Gateway" in result.stdout + result.stderr


def test_gate_retries_after_an_unreadable_api_response_then_passes(tmp_path: Path) -> None:
    """Bug: a 200 response without `workflow_runs` (an API error body) making jq fail and `set -e`
    end the release instead of retrying."""
    result, calls = _run_gate(
        tmp_path, [{"message": "Server Error"}, {"workflow_runs": [_run("completed", "success")]}]
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert len(calls) == 2


def test_gate_times_out_naming_the_last_error_when_every_api_call_fails(tmp_path: Path) -> None:
    """Bug: a persistent API failure retried forever, or ending on a timeout message that hides the
    error that caused it."""
    result, calls = _run_gate(tmp_path, ["HTTP 401: Bad credentials"], timeout_s=0)
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "timed out" in output.lower()
    assert "last error: HTTP 401: Bad credentials" in output
    assert len(calls) == 1


def test_gate_says_plainly_when_no_ci_run_exists_yet(tmp_path: Path) -> None:
    """Bug: the wait loop printing an internal token ("absent") or a blank state instead of saying
    that CI has not started a push run for the commit."""
    result, _ = _run_gate(
        tmp_path, [{"workflow_runs": []}, {"workflow_runs": [_run("completed", "success")]}]
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "no CI push run for abc123 yet" in result.stdout


def test_gate_judges_only_the_latest_run_for_the_sha(tmp_path: Path) -> None:
    """Bug: an old red attempt failing a commit whose re-run went green (or the reverse)."""
    old_green = _run("completed", "success", "2026-09-29T10:00:00Z")
    old_red = _run("completed", "failure", "2026-09-29T10:00:00Z")
    new_green = _run("completed", "success", "2026-09-29T11:00:00Z")
    new_red = _run("completed", "failure", "2026-09-29T11:00:00Z")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    rerun_green, _ = _run_gate(tmp_path / "a", [{"workflow_runs": [old_red, new_green]}])
    rerun_red, _ = _run_gate(tmp_path / "b", [{"workflow_runs": [new_red, old_green]}])
    assert rerun_green.returncode == 0, rerun_green.stderr + rerun_green.stdout
    assert rerun_red.returncode != 0


def test_gate_poll_interval_and_deadline_defaults_are_thirty_seconds_and_thirty_minutes() -> None:
    """Bug: the shipped defaults drifting from the agreed 30 s poll / 30 min bound."""
    text = _VERIFY_SCRIPT.read_text()
    assert "VERIFY_POLL_SECONDS:-30" in text
    assert "VERIFY_TIMEOUT_SECONDS:-1800" in text


def test_github_release_uses_gh_cli_with_the_only_write_scope() -> None:
    """Bug: the release created by a third-party action, or `contents: write` granted beyond the one
    job that creates the release."""
    live = "\n".join(_live_lines(_RELEASE_WORKFLOW))
    assert 'gh release create "$GITHUB_REF_NAME" dist/* --generate-notes --verify-tag' in live
    assert live.count("contents: write") == 1
    assert "contents: write" in [ln.strip() for ln in _release_job("github-release")]
    assert "GH_TOKEN: ${{ github.token }}" in [ln.strip() for ln in _release_job("github-release")]


@pytest.mark.parametrize(
    ("tag", "prerelease"),
    [
        ("v1.0.0rc1", True),
        ("v1.0.0rc12", True),
        ("v1.0.0a1", True),
        ("v1.0.0b2", True),
        ("v1.0.0.dev3", True),
        ("v1.0.0-rc1", True),
        ("v1.0.0", False),
        ("v0.18.0", False),
        ("v0.16.0", False),
        ("v1.0.0.post1", False),
        ("v10.0.0", False),
    ],
)
def test_github_release_marks_pep440_prerelease_tags(tag: str, prerelease: bool) -> None:
    """Bug: only hyphenated tags are flagged, so v1.0.0rc1 becomes the "Latest" GitHub Release."""
    (case_line,) = [
        ln.strip()
        for ln in _release_job("github-release")
        if ln.strip().startswith('case "$GITHUB_REF_NAME" in')
    ]
    assert case_line.endswith("esac")
    result = subprocess.run(
        ["bash", "-c", f'flags=""; {case_line}; printf %s "$flags"'],
        env={"PATH": os.environ["PATH"], "GITHUB_REF_NAME": tag},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ("--prerelease" if prerelease else "")


def test_dependabot_updates_actions_and_uv_weekly() -> None:
    """Bug: pinned action SHAs and uv.lock going stale because nothing proposes updates."""
    text = (_REPO_ROOT / ".github" / "dependabot.yml").read_text()
    blocks = [b for b in text.split("  - package-ecosystem:")[1:]]
    assert sorted(b.split()[0].strip('"') for b in blocks) == ["github-actions", "uv"]
    for block in blocks:
        assert 'interval: "weekly"' in block


# ---- Dependency audit: informational, never a merge gate ----


def test_ci_has_a_non_blocking_least_privilege_audit_job() -> None:
    """Bug: no advisory audit in CI, or one wired as a required job that turns every new upstream
    advisory into a red merge gate."""
    block = [ln.strip() for ln in _job_block(_CI_WORKFLOW.read_text(), "audit")]
    assert "continue-on-error: true" in block
    assert "contents: read" in block
    assert any("pip-audit" in ln and "--no-deps" in ln for ln in block)
    assert any("uv export --frozen" in ln for ln in block)
    assert not [ln for ln in block if "write" in ln]


def test_ci_runs_a_non_blocking_fast_lane_against_mflux_main() -> None:
    """Bug: no job tests mflux's main branch (a breaking upstream change shows up only after a release), the job
    blocks merges whenever upstream main is red, it re-resolves the lock instead of moving only mflux, its log does
    not say which upstream commit was tested, or a red run passes unnoticed because the job is non-blocking."""
    block = [ln.strip() for ln in _job_block(_CI_WORKFLOW.read_text(), "test-mflux-main")]
    assert "continue-on-error: true" in block
    sync = block.index("uv sync --locked --group test-mflux")
    install = block.index('uv pip install "mflux @ git+https://github.com/mflux-community/mflux@main"')
    assert sync < install
    pytest_at = next(
        i
        for i, ln in enumerate(block)
        if ln.startswith('uv run --no-sync pytest -m "not parity and not slow')
    )
    assert install < pytest_at
    assert any("direct_url.json" in ln and "commit_id" in ln for ln in block[install:pytest_at])
    failure_at = block.index("if: failure()")
    assert pytest_at < failure_at
    assert block[failure_at + 1] == 'run: echo "::warning::mflux main broke the fast lane"'


def test_audit_checks_every_locked_package() -> None:
    """Bug: a package exclusion stays in the audit step after its reason is gone (the pillow 12 skip,
    `grep -v '^pillow=='` on the exported requirements), so that package's advisories are never reported."""
    block = [ln.strip() for ln in _job_block(_CI_WORKFLOW.read_text(), "audit")]
    touching_req = [ln for ln in block if "req" in ln]
    assert touching_req == [
        'uv export --frozen --all-groups --all-extras --no-hashes > "$RUNNER_TEMP/req.txt"',
        'uvx pip-audit -r "$RUNNER_TEMP/req.txt" --no-deps --disable-pip --desc off',
    ]
