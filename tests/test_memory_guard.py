"""Unit tests for the test-session MLX memory guard."""

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tests import _memory_guard
from tests._memory_guard import apply_mlx_memory_caps, cache_limit_target, wired_limit_target

GIB = 1024**3
_REPO = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("total_gb", [7, 16, 32, 64, 128])
def test_target_matches_policy_across_machine_sizes(total_gb):
    total = total_gb * GIB
    max_working_set = int(total_gb * 0.78 * GIB)
    target = wired_limit_target(total, max_working_set)
    assert target == min(int(total * 0.60), int(max_working_set * 0.90))


def test_clamps_below_small_working_set_ceiling():
    target = wired_limit_target(32 * GIB, 18 * GIB)
    assert target == int(18 * GIB * 0.90)


@pytest.mark.parametrize(
    "total,max_working_set",
    [(0, 25 * GIB), (32 * GIB, 0), (-1, 25 * GIB), (32 * GIB, -1)],
)
def test_invalid_device_sizes_return_none(total, max_working_set):
    assert wired_limit_target(total, max_working_set) is None


class _FakeMx:
    def __init__(self, info, *, wired_raises=False, memory_raises=False, cache_raises=False):
        self._info = info
        self._wired_raises = wired_raises
        self._memory_raises = memory_raises
        self._cache_raises = cache_raises
        self.wired_calls: list[int] = []
        self.memory_calls: list[int] = []
        self.cache_calls: list[int] = []

    def set_cache_limit(self, value):
        if self._cache_raises:
            raise RuntimeError("set_cache_limit unavailable")
        self.cache_calls.append(value)

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
    assert mx.wired_calls == [min(int(32 * GIB * 0.60), int(max_working_set * 0.90))]
    assert mx.memory_calls == [int(32 * GIB * 0.70)]
    assert messages == []


def test_conftest_installs_memory_caps_at_import(monkeypatch):
    fake_mlx = ModuleType("mlx")
    fake_mlx.__path__ = []
    fake_core = ModuleType("mlx.core")
    fake_mlx.core = fake_core
    calls = []

    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setattr(_memory_guard, "apply_mlx_memory_caps", lambda mx, emit: calls.append(mx))

    runpy.run_path(str(_REPO / "tests/conftest.py"), run_name="_memory_guard_conftest_probe")

    assert calls == [fake_core]


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


# --- cache pool -----------------------------------------------------------------
# MLX parks freed buffers in a retained cache pool whose default limit is near
# device memory, so a parity module that generates dozens of images grows its
# footprint far past one generation's peak (2026-09-06: a FLUX.1 parity file
# reached a 26 GB footprint on a 32 GB Mac and paged the machine for 45 min).
# The scripts bound the pool to a quarter of the wired cap, at most 2 GiB; the
# test session follows the same policy.


def test_cache_target_is_a_quarter_of_the_wired_cap_at_most_two_gib():
    """bug caught: dropping the quarter clamp (a 2 GiB pool on a 4 GiB wired cap)."""
    assert cache_limit_target(4 * GIB) == GIB
    assert cache_limit_target(19 * GIB) == 2 * GIB


def test_apply_caps_bounds_the_cache_pool_from_the_wired_target():
    """bug caught: never calling set_cache_limit, or sizing it off the wrong cap."""
    max_working_set = int(0.78 * 32 * GIB)
    mx = _FakeMx({"memory_size": 32 * GIB, "max_recommended_working_set_size": max_working_set})
    apply_mlx_memory_caps(mx, lambda _m: None)
    assert mx.cache_calls == [cache_limit_target(mx.wired_calls[0])]


def test_no_wired_cap_means_no_cache_cap():
    """bug caught: sizing the pool off a missing wired target (TypeError or a 0 cap)."""
    mx = _FakeMx({"memory_size": 32 * GIB})
    messages: list[str] = []
    apply_mlx_memory_caps(mx, messages.append)
    assert mx.cache_calls == []


def test_cache_cap_failure_is_emitted_and_never_raised():
    max_working_set = int(0.78 * 32 * GIB)
    mx = _FakeMx(
        {"memory_size": 32 * GIB, "max_recommended_working_set_size": max_working_set},
        cache_raises=True,
    )
    messages: list[str] = []
    apply_mlx_memory_caps(mx, messages.append)
    assert any("set_cache_limit" in message for message in messages)
    assert mx.memory_calls, "the soft cap must still be applied after a cache-cap failure"
