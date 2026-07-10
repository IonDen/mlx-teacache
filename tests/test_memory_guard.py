"""Unit tests for the test-session MLX memory guard."""

import pytest

from tests._memory_guard import apply_mlx_memory_caps, wired_limit_target

GIB = 1024**3


@pytest.mark.parametrize("total_gb", [7, 16, 32, 64, 128])
def test_target_is_positive_and_strictly_below_ceiling(total_gb):
    total = total_gb * GIB
    max_working_set = int(total_gb * 0.78 * GIB)
    target = wired_limit_target(total, max_working_set)
    assert target is not None
    assert 0 < target < max_working_set


def test_clamps_below_small_working_set_ceiling():
    target = wired_limit_target(32 * GIB, 18 * GIB)
    assert target is not None
    assert 0 < target < 18 * GIB


@pytest.mark.parametrize(
    "total,max_working_set",
    [(0, 25 * GIB), (32 * GIB, 0), (-1, 25 * GIB), (32 * GIB, -1)],
)
def test_invalid_device_sizes_return_none(total, max_working_set):
    assert wired_limit_target(total, max_working_set) is None


class _FakeMx:
    def __init__(self, info, *, wired_raises=False, memory_raises=False):
        self._info = info
        self._wired_raises = wired_raises
        self._memory_raises = memory_raises
        self.wired_calls: list[int] = []
        self.memory_calls: list[int] = []

    def device_info(self):
        return self._info

    def set_wired_limit(self, value):
        if self._wired_raises:
            raise RuntimeError("set_wired_limit unavailable")
        self.wired_calls.append(value)

    def set_memory_limit(self, value):
        if self._memory_raises:
            raise RuntimeError("set_memory_limit unavailable")
        self.memory_calls.append(value)


def test_apply_caps_sets_both_on_a_healthy_device():
    max_working_set = int(0.78 * 32 * GIB)
    mx = _FakeMx({"memory_size": 32 * GIB, "max_recommended_working_set_size": max_working_set})
    messages: list[str] = []
    apply_mlx_memory_caps(mx, messages.append)
    assert mx.wired_calls and 0 < mx.wired_calls[0] < max_working_set
    assert mx.memory_calls
    assert messages == []


def test_wired_failure_still_attempts_memory_cap_and_emits():
    max_working_set = int(0.78 * 7 * GIB)
    mx = _FakeMx(
        {"memory_size": 7 * GIB, "max_recommended_working_set_size": max_working_set},
        wired_raises=True,
    )
    messages: list[str] = []
    apply_mlx_memory_caps(mx, messages.append)
    assert mx.wired_calls == []
    assert mx.memory_calls
    assert any("set_wired_limit" in message for message in messages)


def test_incomplete_device_info_skips_wired_and_emits():
    mx = _FakeMx({"memory_size": 32 * GIB})
    messages: list[str] = []
    apply_mlx_memory_caps(mx, messages.append)
    assert mx.wired_calls == []
    assert mx.memory_calls
    assert any("no safe wired cap" in message for message in messages)


def test_memory_cap_failure_is_emitted_and_never_raised():
    max_working_set = int(0.78 * 32 * GIB)
    mx = _FakeMx(
        {"memory_size": 32 * GIB, "max_recommended_working_set_size": max_working_set},
        memory_raises=True,
    )
    messages: list[str] = []
    apply_mlx_memory_caps(mx, messages.append)
    assert any("set_memory_limit" in message for message in messages)


def test_device_info_failure_is_emitted_and_never_raised():
    class _Boom:
        def device_info(self):
            raise RuntimeError("device info unavailable")

    messages: list[str] = []
    apply_mlx_memory_caps(_Boom(), messages.append)
    assert any("device_info" in message for message in messages)
