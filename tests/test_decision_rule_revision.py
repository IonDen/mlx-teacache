"""The decision-rule revision number and its history.

The revision names the rule that turns a step's inputs into a compute or skip decision. It
moves only in a minor or major release, and every value it has held is recorded with the
release that introduced it."""

import re
import subprocess
import sys

from mlx_teacache._kernel.gate import DECISION_RULE_HISTORY, DECISION_RULE_REVISION


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def test_revision_one_was_introduced_in_0_16_0() -> None:
    """Bug: the starting revision or the release it is dated to drifts from 1 / 0.16.0."""
    assert DECISION_RULE_REVISION == 1
    assert DECISION_RULE_HISTORY[1] == "0.16.0"


def test_newest_history_key_is_the_current_revision() -> None:
    """Bug: the revision is bumped without a history entry for the new value."""
    assert max(DECISION_RULE_HISTORY) == DECISION_RULE_REVISION


def test_history_keys_run_from_one_without_gaps() -> None:
    """Bug: a revision number is skipped or renumbered."""
    keys = sorted(DECISION_RULE_HISTORY)
    assert keys == list(range(1, len(keys) + 1))


def test_every_revision_was_introduced_in_a_minor_or_major() -> None:
    """Bug: a revision bumped in a patch release such as "0.16.1"."""
    patch_or_odd = [v for v in DECISION_RULE_HISTORY.values() if not re.fullmatch(r"\d+\.\d+\.0", v)]
    assert patch_or_odd == []


def test_history_versions_increase_with_the_revision() -> None:
    """Bug: a new revision dated to an older release than the one before it."""
    versions = [_version_tuple(DECISION_RULE_HISTORY[key]) for key in sorted(DECISION_RULE_HISTORY)]
    assert all(earlier < later for earlier, later in zip(versions, versions[1:], strict=False))


def test_package_root_import_does_not_load_mlx_core() -> None:
    """Bug: the handle reads the revision from a module that imports mlx.core, so `import mlx_teacache` loads MLX
    again and a launcher that only asks about support pays for it. Runs in a fresh isolated interpreter."""
    code = (
        "import sys\n"
        "import mlx_teacache\n"
        "loaded = sorted(m for m in sys.modules if m == 'mlx.core' or m.startswith('mlx.core.'))\n"
        "print(mlx_teacache.TeaCacheHandle.__name__, loaded)\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "TeaCacheHandle []"
