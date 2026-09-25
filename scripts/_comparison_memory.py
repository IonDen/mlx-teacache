"""Resident / OS-footprint / host-free sampling for a comparison worker, per phase.

``mx.get_peak_memory`` counts active memory only; the watchdog counts active + cache; macOS counts the process
footprint (FLUX.1-dev: 14.8 GiB OS vs 10.4 GiB MLX); the harness kills background tasks on host free memory. The
probe decision needs all of them, over every phase (load, encode, generation). A sampler failure is an error, never a
silent optimistic number.
"""

import ctypes
import os
import threading
from collections.abc import Callable
from typing import Any

_LIBPROC: Any = None
_LIBC: Any = None


class _RusageInfoV0(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time",
            "ri_system_time",
            "ri_pkg_idle_wkups",
            "ri_interrupt_wkups",
            "ri_pageins",
            "ri_wired_size",
            "ri_resident_size",
            "ri_phys_footprint",
            "ri_proc_start_abstime",
            "ri_proc_exit_abstime",
        )
    ]


def phys_footprint_bytes(pid: int | None = None) -> int:
    global _LIBPROC
    if _LIBPROC is None:
        _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
    info = _RusageInfoV0()
    rc = _LIBPROC.proc_pid_rusage(ctypes.c_int(pid or os.getpid()), ctypes.c_int(0), ctypes.byref(info))
    if rc != 0:
        raise OSError(f"proc_pid_rusage failed with {rc}")
    return int(info.ri_phys_footprint)


def host_free_pct() -> float:
    global _LIBC
    if _LIBC is None:
        _LIBC = ctypes.CDLL(None)
    value, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
    rc = _LIBC.sysctlbyname(b"kern.memorystatus_level", ctypes.byref(value), ctypes.byref(size), None, 0)
    if rc != 0:
        raise OSError(f"sysctlbyname(kern.memorystatus_level) failed with {rc}")
    return float(value.value)


def mlx_resident_sampler() -> Callable[[], int]:
    import mlx.core as mx

    return lambda: int(mx.get_active_memory()) + int(mx.get_cache_memory())


class SamplerError(RuntimeError):
    pass


class _Window:
    def __init__(self) -> None:
        self.resident = 0
        self.footprint = 0
        self.host: float | None = None

    def add(self, resident: int, footprint: int, host: float) -> None:
        self.resident = max(self.resident, resident)
        self.footprint = max(self.footprint, footprint)
        self.host = host if self.host is None else min(self.host, host)

    def as_dict(self) -> dict[str, Any]:
        return {
            "peak_resident_bytes": self.resident,
            "peak_footprint_bytes": self.footprint,
            "min_host_free_pct": self.host,
        }


class PeakSampler:
    def __init__(
        self,
        *,
        sample_resident: Callable[[], int],
        sample_footprint: Callable[[], int],
        sample_host_free: Callable[[], float],
        poll_s: float = 0.05,
    ) -> None:
        self._resident, self._footprint, self._host = sample_resident, sample_footprint, sample_host_free
        self._poll_s = poll_s
        self._lock = threading.Lock()
        self._total, self._phase = _Window(), _Window()
        self._phases: dict[str, dict[str, Any]] = {}
        self._error: BaseException | None = None
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> None:
        resident, footprint, host = self._resident(), self._footprint(), self._host()
        with self._lock:
            self._total.add(resident, footprint, host)
            self._phase.add(resident, footprint, host)

    def end_phase(self, name: str) -> None:
        """Close the current window. Sampling is the thread's job (or the test's), never the boundary's."""
        with self._lock:
            self._phases[name] = self._phase.as_dict()
            self._phase = _Window()

    def start(self) -> None:
        def _loop() -> None:
            try:
                while not self._halt.is_set():
                    self.sample_once()
                    self._halt.wait(self._poll_s)
            except BaseException as exc:  # noqa: BLE001 — surfaced by stop()
                self._error = exc

        self._thread = threading.Thread(target=_loop, name="comparison-peak-sampler", daemon=True)
        self._thread.start()

    def _result(self) -> dict[str, Any]:
        if self._error is not None:
            raise SamplerError(f"memory sampler failed: {self._error!r}") from self._error
        if self._total.host is None:
            raise SamplerError("no host free-memory sample was taken")
        return {"phases": dict(self._phases), **self._total.as_dict()}

    def stop_without_thread(self) -> dict[str, Any]:
        """For tests and single-threaded use: the result without joining a thread."""
        return self._result()

    def stop(self) -> dict[str, Any]:
        self._halt.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise SamplerError("memory sampler thread did not stop within 5 s")
        return self._result()
