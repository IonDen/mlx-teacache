"""The memory watchdog's decision logic and thread behaviour, with a fake
sampler. The thread is real; only the memory source and the exit are injected."""

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _mlx_watchdog as wd  # noqa: E402

GIB = 1024**3


def test_ceiling_is_memory_size_minus_headroom() -> None:
    # bug caught: forgetting to subtract the headroom (watchdog fires only at physical RAM)
    assert wd.ceiling_bytes(32 * GIB) == 28 * GIB
    assert wd.ceiling_bytes(32 * GIB, headroom_gib=2) == 30 * GIB


def test_ceiling_rejects_non_positive_result() -> None:
    with pytest.raises(ValueError):
        wd.ceiling_bytes(3 * GIB)  # 3 - 4 < 0


def test_over_ceiling_counts_active_plus_cache() -> None:
    # bug caught: comparing active alone (dropped buffers sit in the cache pool)
    assert not wd.over_ceiling(20 * GIB, 7 * GIB, 28 * GIB)
    assert not wd.over_ceiling(20 * GIB, 8 * GIB, 28 * GIB)  # exactly at the ceiling is still inside it
    assert wd.over_ceiling(20 * GIB, 9 * GIB, 28 * GIB)


def test_watchdog_aborts_once_resident_exceeds_ceiling() -> None:
    # bug caught: the thread never calling exit_fn, or aborting on active alone
    samples = iter([(10 * GIB, 1 * GIB), (26 * GIB, 3 * GIB)])
    payloads: list[dict[str, int]] = []
    exited = threading.Event()
    codes: list[int] = []

    def _exit(code: int) -> None:
        codes.append(code)
        exited.set()

    wd.start_watchdog(
        ceiling=28 * GIB,
        sample=lambda: next(samples),
        on_abort=payloads.append,
        exit_fn=_exit,
        poll_s=0.001,
        stop=exited,
    )
    assert exited.wait(2.0), "watchdog never aborted"
    assert codes == [3]
    assert payloads[0]["resident_bytes"] == 29 * GIB
    assert payloads[0]["ceiling_bytes"] == 28 * GIB


def test_watchdog_stays_quiet_under_ceiling() -> None:
    stop = threading.Event()
    calls: list[int] = []
    t = wd.start_watchdog(
        ceiling=28 * GIB,
        sample=lambda: (10 * GIB, 1 * GIB),
        on_abort=lambda p: calls.append(1),
        exit_fn=lambda c: calls.append(c),
        poll_s=0.001,
        stop=stop,
    )
    threading.Event().wait(0.05)
    stop.set()
    t.join(1.0)
    assert calls == []


def test_watchdog_still_exits_when_on_abort_raises() -> None:
    # bug caught: an exception in the abort handler (a write under memory distress)
    # killing the thread before exit_fn, leaving the process unguarded and silent
    exited = threading.Event()
    codes: list[int] = []

    def _exit(code: int) -> None:
        codes.append(code)
        exited.set()

    def _bad_handler(payload: dict[str, int]) -> None:
        raise OSError("disk full")

    wd.start_watchdog(
        ceiling=28 * GIB,
        sample=lambda: (27 * GIB, 2 * GIB),
        on_abort=_bad_handler,
        exit_fn=_exit,
        poll_s=0.001,
        stop=exited,
    )
    assert exited.wait(2.0), "watchdog did not exit after a raising abort handler"
    assert codes == [3]


def test_watchdog_keeps_polling_when_sample_raises_once() -> None:
    # bug caught: a transient sampler failure ending the poll loop for good
    calls = {"n": 0}

    def _sample() -> tuple[int, int]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return (27 * GIB, 2 * GIB)

    exited = threading.Event()
    codes: list[int] = []

    def _exit(code: int) -> None:
        codes.append(code)
        exited.set()

    wd.start_watchdog(
        ceiling=28 * GIB, sample=_sample, on_abort=lambda p: None, exit_fn=_exit, poll_s=0.001, stop=exited
    )
    assert exited.wait(2.0), "watchdog stopped polling after one sampler error"
    assert codes == [3]
    assert calls["n"] >= 2
