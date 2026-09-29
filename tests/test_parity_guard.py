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
        start_memory=lambda on_abort, **_: calls.append("mem"),
        start_wall=lambda **_: calls.append("wall"),
        artifact_dir=tmp_path,
        install_caps=lambda cap: None,
    )
    assert armed is False
    assert calls == []


def test_armed_when_a_parity_item_is_selected(tmp_path: Path, restore_flag: None) -> None:
    """Bug: parity selection is not detected, so the heavy lane runs unguarded."""
    calls: list[str] = []
    armed = mg.arm_parity_guard(
        [_item("slow"), _item("parity")],
        start_memory=lambda on_abort, **_: calls.append("mem"),
        start_wall=lambda **_: calls.append("wall"),
        artifact_dir=tmp_path,
        install_caps=lambda cap: None,
    )
    assert armed is True
    assert calls == ["mem", "wall"]


def test_memory_trip_writes_artifact_and_exits_with_watchdog_code(tmp_path: Path, restore_flag: None) -> None:
    """Bug: a trip exits with the wrong code or leaves no pytest-parity artifact."""
    exits: list[int] = []
    done = threading.Event()

    def start_memory(on_abort, **_):
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
        [_item("parity")],
        start_memory=start_memory,
        start_wall=lambda **_: None,
        artifact_dir=tmp_path,
        install_caps=lambda cap: None,
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
        start_memory=lambda on_abort, **_: None,
        environ={"PYTEST_PARITY_WALL_S": "0.05"},
        exit_fn=_exit,
        artifact_dir=tmp_path,
        install_caps=lambda cap: None,
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
    capture = object()
    monkeypatch.setattr(conftest, "arm_parity_guard", lambda items, *, capture: seen.append((items, capture)))
    items = [_item("parity")]
    plugins = SimpleNamespace(getplugin=lambda name: capture if name == "capturemanager" else None)
    conftest.pytest_collection_finish(
        SimpleNamespace(items=items, config=SimpleNamespace(pluginmanager=plugins))
    )
    assert seen == [(items, capture)]


# --- stale artifact ---------------------------------------------------------


def test_arming_removes_a_stale_abort_artifact(tmp_path: Path, restore_flag: None) -> None:
    """Bug: a pytest-parity.aborted.json left by an earlier run survives a later clean run, so the
    resume loop (or a person) reads an old abort as this session's result."""
    stale = tmp_path / "pytest-parity.aborted.json"
    stale.write_text('{"reason": "memory"}')
    mg.arm_parity_guard(
        [_item("parity")],
        start_memory=lambda on_abort, **_: None,
        start_wall=lambda **_: None,
        artifact_dir=tmp_path,
        install_caps=lambda cap: None,
    )
    assert not stale.exists()


def test_fast_lane_session_keeps_the_last_abort_artifact(tmp_path: Path) -> None:
    """Bug: the stale-artifact cleanup running on every session, so a quick fast-lane run
    wipes the only record of the parity abort that just happened."""
    stale = tmp_path / "pytest-parity.aborted.json"
    stale.write_text('{"reason": "memory"}')
    mg.arm_parity_guard(
        [_item("slow")],
        start_memory=lambda on_abort, **_: None,
        start_wall=lambda **_: None,
        artifact_dir=tmp_path,
        install_caps=lambda cap: None,
    )
    assert stale.read_text() == '{"reason": "memory"}'


# --- cache pool -------------------------------------------------------------


def test_arming_lowers_the_cache_pool_to_one_gib(tmp_path: Path, restore_flag: None) -> None:
    """Bug: the parity lane keeping the fast lane's 2 GiB pool; Qwen parity (~26 GiB active)
    plus 2 GiB sits on the 28 GiB active+cache ceiling."""
    caps: list[int] = []
    mg.arm_parity_guard(
        [_item("parity")],
        start_memory=lambda on_abort, **_: None,
        start_wall=lambda **_: None,
        artifact_dir=tmp_path,
        install_caps=caps.append,
    )
    assert caps == [GIB]


def test_unarmed_session_leaves_the_cache_pool_alone(tmp_path: Path) -> None:
    """Bug: the 1 GiB pool applied to the fast lane too (it keeps 2 GiB)."""
    caps: list[int] = []
    mg.arm_parity_guard(
        [_item()],
        start_memory=lambda on_abort, **_: None,
        start_wall=lambda **_: None,
        artifact_dir=tmp_path,
        install_caps=caps.append,
    )
    assert caps == []


# --- headroom override ------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 4.0),
        ("6", 6.0),
        ("0.5", 0.5),
        ("0", 4.0),
        ("-2", 4.0),
        ("junk", 4.0),
        ("nan", 4.0),
        ("inf", 4.0),
    ],
)
def test_headroom_default_and_override(raw: str | None, expected: float) -> None:
    """Bug: the override ignored, or a zero / negative / non-finite value accepted (an infinite
    headroom makes the ceiling negative and aborts the session at once; zero leaves the OS nothing)."""
    environ = {} if raw is None else {"PYTEST_PARITY_HEADROOM_GIB": raw}
    assert mg.headroom_gib(environ) == expected


def test_arming_hands_the_headroom_override_to_the_memory_watchdog(
    tmp_path: Path, restore_flag: None
) -> None:
    """Bug: the env override parsed but never reaching the watchdog, which keeps its 4 GiB default."""
    seen: list[float] = []
    mg.arm_parity_guard(
        [_item("parity")],
        start_memory=lambda on_abort, *, headroom_gib: seen.append(headroom_gib),
        start_wall=lambda **_: None,
        environ={"PYTEST_PARITY_HEADROOM_GIB": "6"},
        artifact_dir=tmp_path,
        install_caps=lambda cap: None,
    )
    assert seen == [6.0]


# --- abort line under pytest capture ---------------------------------------


class _FakeCapture:
    """Stand-in for pytest's capture manager: records the suspend call into a shared event log."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def suspend_global_capture(self, in_: bool = False) -> None:
        self.events.append(f"suspend(in_={in_})")


class _RecordingStderr:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def write(self, text: str) -> int:
        if text.strip():
            self.events.append(f"write:{text.strip()}")
        return len(text)

    def flush(self) -> None:
        pass


def _arm_and_capture_handlers(tmp_path: Path, capture: object, **kw: object) -> dict[str, object]:
    handlers: dict[str, object] = {}
    mg.arm_parity_guard(
        [_item("parity")],
        start_memory=lambda on_abort, **_: handlers.__setitem__("memory", on_abort),
        start_wall=lambda **k: handlers.__setitem__("wall", k["on_abort"]),
        artifact_dir=tmp_path,
        install_caps=lambda cap: None,
        capture=capture,
        **kw,
    )
    return handlers


def test_memory_abort_line_suspends_pytest_capture_before_printing(
    tmp_path: Path, restore_flag: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: the abort line printed while pytest's fd capture holds stderr; the process then
    os._exit()s, the capture buffer is never replayed, and the terminal shows nothing."""
    events: list[str] = []
    monkeypatch.setattr(sys, "stderr", _RecordingStderr(events))
    handlers = _arm_and_capture_handlers(tmp_path, _FakeCapture(events))
    handlers["memory"]({"resident_bytes": 29 * GIB, "ceiling_bytes": 28 * GIB})  # type: ignore[operator]
    assert events[0] == "suspend(in_=True)"
    assert events[1] == "write:[watchdog] ABORTED pytest-parity: 29.00 GiB resident > 28.00 GiB ceiling"


def test_wall_abort_line_suspends_pytest_capture_before_printing(
    tmp_path: Path, restore_flag: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug: the wall handler printing under capture (same silent exit as the memory handler)."""
    events: list[str] = []
    monkeypatch.setattr(sys, "stderr", _RecordingStderr(events))
    handlers = _arm_and_capture_handlers(tmp_path, _FakeCapture(events))
    handlers["wall"]({"wall_s": 43200.0})  # type: ignore[operator]
    assert events == ["suspend(in_=True)", "write:[watchdog] ABORTED pytest-parity: wall backstop 43200.0 s"]


def test_abort_line_goes_to_the_saved_stderr_fd_without_a_capture_plugin(
    tmp_path: Path, restore_flag: None
) -> None:
    """Bug: with the capture plugin absent (-p no:capture) the handler crashing on a None
    capture manager, or writing nowhere."""
    import os

    read_fd, write_fd = os.pipe()
    try:
        handlers = _arm_and_capture_handlers(tmp_path, None, stderr_fd=write_fd)
        handlers["wall"]({"wall_s": 5.0})  # type: ignore[operator]
        os.close(write_fd)
        write_fd = -1
        assert os.read(read_fd, 4096).decode() == "[watchdog] ABORTED pytest-parity: wall backstop 5.0 s\n"
    finally:
        os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
