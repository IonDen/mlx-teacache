"""The parity-lane guard: arm the active+cache watchdog and a wall backstop only
when parity items are selected. Memory reader, exit function and clock are fakes;
the poll threads are real."""

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _mlx_watchdog as wd  # noqa: E402

from tests import _memory_guard as mg  # noqa: E402

GIB = 1024**3


@pytest.fixture
def restore_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """arm_parity_guard sets a module flag; put the session's value back after the test."""
    monkeypatch.setattr(mg, "PARITY_GUARD_ARMED", mg.PARITY_GUARD_ARMED)


def _item(*marks: str) -> SimpleNamespace:
    return SimpleNamespace(get_closest_marker=lambda name: object() if name in marks else None)


def test_unarmed_when_no_parity_item_selected(tmp_path: Path) -> None:
    """Bug: arming on every session adds a poll thread to the fast lane."""
    calls: list[str] = []
    armed = mg.arm_parity_guard(
        [_item("slow"), _item()],
        start_memory=lambda on_abort: calls.append("mem"),
        start_wall=lambda **_: calls.append("wall"),
        artifact_dir=tmp_path,
    )
    assert armed is False
    assert calls == []


def test_armed_when_a_parity_item_is_selected(tmp_path: Path, restore_flag: None) -> None:
    """Bug: parity selection is not detected, so the heavy lane runs unguarded."""
    calls: list[str] = []
    armed = mg.arm_parity_guard(
        [_item("slow"), _item("parity")],
        start_memory=lambda on_abort: calls.append("mem"),
        start_wall=lambda **_: calls.append("wall"),
        artifact_dir=tmp_path,
    )
    assert armed is True
    assert calls == ["mem", "wall"]


def test_memory_trip_writes_artifact_and_exits_with_watchdog_code(tmp_path: Path, restore_flag: None) -> None:
    """Bug: a trip exits with the wrong code or leaves no pytest-parity artifact."""
    exits: list[int] = []
    done = threading.Event()

    def start_memory(on_abort):
        def _exit(code: int) -> None:
            exits.append(code)
            done.set()

        return wd.start_watchdog(
            ceiling=28 * GIB,
            sample=lambda: (27 * GIB, 2 * GIB),
            on_abort=on_abort,
            exit_fn=_exit,
            poll_s=0.01,
        )

    assert mg.arm_parity_guard(
        [_item("parity")], start_memory=start_memory, start_wall=lambda **_: None, artifact_dir=tmp_path
    )
    assert done.wait(5)
    assert exits == [4]
    record = json.loads((tmp_path / "pytest-parity.aborted.json").read_text())
    assert record["label"] == "pytest-parity"
    assert record["reason"] == "memory"
    assert record["resident_bytes"] == 29 * GIB
    assert record["ceiling_bytes"] == 28 * GIB


def test_wall_expiry_writes_wall_artifact_and_exits(tmp_path: Path, restore_flag: None) -> None:
    """Bug: the wall backstop never fires, or exits without an artifact."""
    exits: list[int] = []
    done = threading.Event()

    def _exit(code: int) -> None:
        exits.append(code)
        done.set()

    mg.arm_parity_guard(
        [_item("parity")],
        start_memory=lambda on_abort: None,
        environ={"PYTEST_PARITY_WALL_S": "0.05"},
        exit_fn=_exit,
        artifact_dir=tmp_path,
    )
    assert done.wait(5)
    assert exits == [4]
    record = json.loads((tmp_path / "pytest-parity.aborted.json").read_text())
    assert record["reason"] == "wall"
    assert record["wall_s"] == 0.05


def test_wall_backstop_stops_without_firing_when_halted(tmp_path: Path) -> None:
    """Bug: the backstop ignores its stop event and kills a finished session."""
    exits: list[int] = []
    stop = threading.Event()
    thread = mg.start_wall_backstop(
        wall_s=60.0, on_abort=lambda payload: None, exit_fn=exits.append, stop=stop
    )
    stop.set()
    thread.join(5)
    assert not thread.is_alive()
    assert exits == []


def test_wall_seconds_default_and_override() -> None:
    """Bug: the default is not 3 h, or the env override is ignored."""
    assert mg.wall_seconds({}) == 10800.0
    assert mg.wall_seconds({"PYTEST_PARITY_WALL_S": "90"}) == 90.0
    assert mg.wall_seconds({"PYTEST_PARITY_WALL_S": "junk"}) == 10800.0


def test_fast_lane_session_did_not_arm() -> None:
    """Bug: conftest arms the guard in a session with no parity item."""
    assert mg.PARITY_GUARD_ARMED is False


def test_exit_code_matches_the_watchdog_module() -> None:
    """Bug: the wall backstop exits with a code the resume loops read differently from a memory abort."""
    assert mg.WATCHDOG_EXIT_CODE == wd.WATCHDOG_EXIT_CODE == 4


def test_conftest_hook_hands_the_final_item_list_to_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug: the session hook is dropped or reads the wrong list, so a parity run is never guarded."""
    from tests import conftest

    seen: list[object] = []
    monkeypatch.setattr(conftest, "arm_parity_guard", lambda items: seen.append(items))
    items = [_item("parity")]
    conftest.pytest_collection_finish(SimpleNamespace(items=items))
    assert seen == [items]
