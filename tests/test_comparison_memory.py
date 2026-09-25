"""Memory sampling for the comparison harness (pure-core lane; libproc/sysctl tests are macOS-only)."""

import ctypes
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _comparison_memory as cm  # noqa: E402

darwin = pytest.mark.skipif(sys.platform != "darwin", reason="libproc / sysctl are macOS-only")


def _sampler(resident, footprint, host) -> cm.PeakSampler:  # noqa: ANN001
    return cm.PeakSampler(sample_resident=resident, sample_footprint=footprint, sample_host_free=host)


def test_phases_keep_their_own_peaks_and_the_totals_span_all_phases() -> None:
    """Bug: a phase boundary resets the overall peak, or the load phase is not recorded separately."""
    r, f, h = iter([5, 9, 4, 3]), iter([10, 12, 11, 7]), iter([50.0, 30.0, 45.0, 60.0])
    s = _sampler(lambda: next(r), lambda: next(f), lambda: next(h))
    s.sample_once()
    s.sample_once()
    s.end_phase("load")
    s.sample_once()
    s.sample_once()
    s.end_phase("generation")
    out = s.stop_without_thread()
    assert out["phases"]["load"] == {
        "peak_resident_bytes": 9,
        "peak_footprint_bytes": 12,
        "min_host_free_pct": 30.0,
    }
    assert out["phases"]["generation"]["peak_resident_bytes"] == 4
    assert (out["peak_resident_bytes"], out["peak_footprint_bytes"], out["min_host_free_pct"]) == (
        9,
        12,
        30.0,
    )


def test_a_failing_sample_in_the_thread_makes_stop_raise() -> None:
    """Bug: the daemon thread dies silently and the probe passes on frozen, optimistic numbers."""

    def host() -> float:
        raise OSError("sysctl failed")

    s = cm.PeakSampler(
        sample_resident=lambda: 1, sample_footprint=lambda: 1, sample_host_free=host, poll_s=0.001
    )
    s.start()
    time.sleep(0.05)
    with pytest.raises(cm.SamplerError, match="sysctl failed"):
        s.stop()


def test_stop_raises_when_the_host_was_never_sampled() -> None:
    """Bug: min_host_free starts at 100 and 'passes' when nothing was measured."""
    s = _sampler(lambda: 1, lambda: 1, lambda: 50.0)
    with pytest.raises(cm.SamplerError, match="no host"):
        s.stop_without_thread()


@darwin
def test_rusage_struct_matches_the_v0_layout() -> None:
    """Bug: a field added/dropped shifts ri_phys_footprint onto ri_resident_size."""
    assert ctypes.sizeof(cm._RusageInfoV0) == 96


@darwin
def test_phys_footprint_grows_when_this_process_touches_memory() -> None:
    """Bug: the wrong field or pid is read (the number would not follow our own allocation)."""
    before = cm.phys_footprint_bytes(os.getpid())
    block = bytearray(256 * 1024 * 1024)
    for i in range(0, len(block), 4096):
        block[i] = 1
    after = cm.phys_footprint_bytes(os.getpid())
    assert after - before > 200 * 1024 * 1024
    del block


@darwin
def test_host_free_pct_is_a_percentage() -> None:
    """Bug: sysctl read as the wrong type (bytes or a pointer value)."""
    assert 0.0 <= cm.host_free_pct() <= 100.0
