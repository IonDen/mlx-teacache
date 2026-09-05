"""Pure-helper tests for scripts/_mlx_caps.py. No real mlx state is touched:
install_caps is exercised against a fake mlx.core so the three set_* calls
and their derived values can be asserted without a device."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _mlx_caps as caps  # noqa: E402

GIB = 1024**3


def test_clamped_wired_bytes_takes_the_smaller_of_request_and_fraction() -> None:
    # bug caught: dropping the min() and returning the raw request
    assert caps.clamped_wired_bytes(22, 24 * GIB) == int(24 * GIB * 0.85)
    assert caps.clamped_wired_bytes(10, 24 * GIB) == 10 * GIB


def test_clamped_cache_bytes_never_exceeds_a_quarter_of_the_wired_cap() -> None:
    # bug caught: returning cache_gb * GIB unclamped (the 30.4 GiB default footgun)
    assert caps.clamped_cache_bytes(2.0, 20 * GIB) == 2 * GIB
    assert caps.clamped_cache_bytes(8.0, 20 * GIB) == 5 * GIB


def test_clamped_cache_bytes_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        caps.clamped_cache_bytes(0.0, 20 * GIB)


def test_install_caps_sets_all_three_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    # bug caught: install_caps never calling set_cache_limit
    import mlx.core as mx

    calls: dict[str, int] = {}
    monkeypatch.setattr(mx, "device_info", lambda: {"max_recommended_working_set_size": 24 * GIB})
    monkeypatch.setattr(mx, "set_wired_limit", lambda n: calls.__setitem__("wired", n))
    monkeypatch.setattr(mx, "set_memory_limit", lambda n: calls.__setitem__("soft", n))
    monkeypatch.setattr(mx, "set_cache_limit", lambda n: calls.__setitem__("cache", n))
    wired, soft, cache = caps.install_caps(wired_gb=22, soft_gb=22, cache_gb=2.0)
    assert calls == {"wired": wired, "soft": soft, "cache": cache}
    assert cache == 2 * GIB
    assert wired == int(24 * GIB * 0.85)
    assert soft == 22 * GIB
