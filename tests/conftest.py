# tests/conftest.py
"""Shared pytest fixtures and marker handling.

The `mflux` marker is auto-applied to every test whose path relative to
`tests/` is in the explicit allowlist in `tests/_lanes.py` (matched exactly, not
by glob or basename) — these all import the integration layer and therefore
require mflux. The
test-pure-core CI job skips them via `-m "not mflux"`.

Memory guardrail: at session start we install a hard cap on MLX wired
(non-pageable Metal) memory via `mx.set_wired_limit`. Without this cap,
running a marker-misfiltered parity test (e.g. `pytest tests/ -m "not
slow"`) loads a real FLUX model whose wired peak crosses the system
wired limit and panics the kernel watchdog — observed on this 32 GB
M1 Max as crashes on 2026-05-17, 2026-05-19, and 2026-05-20. The
wired cap prevents only wired exhaustion: pageable GPU memory can still
grow past the working set into a paging storm, which has also panicked
this machine. So when a session selects parity tests it arms the
active+cache watchdog (scripts/_mlx_watchdog.py, abort at memory_size minus
4 GiB, PYTEST_PARITY_HEADROOM_GIB) and a wall backstop (PYTEST_PARITY_WALL_S,
default 3 h), and lowers the MLX cache pool to 1 GiB; a trip prints one line
past pytest's output capture, writes
tests/_artifacts/watchdog_aborts/pytest-parity.aborted.json and exits with
code 4. The fast lane arms neither and keeps a 2 GiB pool. See CLAUDE.md "Memory
guardrails for heavy generations" and ml-explore/mlx-lm #883 for the
upstream confirmation that wired memory, not the soft `set_memory_limit`,
is the root cause."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from tests._lanes import is_mflux_file
from tests._memory_guard import apply_mlx_memory_caps, arm_parity_guard


def _install_mlx_memory_caps() -> None:
    """Hard-cap Metal wired memory before any test imports MLX models."""
    try:
        import mlx.core as mx
    except ImportError:
        return
    apply_mlx_memory_caps(mx, lambda message: print(f"mlx-teacache tests: {message}", file=sys.stderr))


_install_mlx_memory_caps()


_TESTS_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        rel_path = Path(item.path).resolve().relative_to(_TESTS_DIR).as_posix()
        if is_mflux_file(rel_path):
            item.add_marker(pytest.mark.mflux)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Arm the parity watchdog and wall backstop from the final (post -m) item list."""
    arm_parity_guard(session.items, capture=session.config.pluginmanager.getplugin("capturemanager"))


@pytest.fixture(autouse=True)
def _untested_mflux_warning_already_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts as if apply_teacache's once-per-process TeaCacheUntestedMfluxWarning had already fired.
    Without this, on a newer mflux the warning lands in whichever test applies first, so warning assertions depend
    on test order. The warning's own tests reset the flag (fixture installed_mflux in
    tests/test_untested_mflux_warning.py). The message-based ignore in pyproject.toml stays for subprocesses."""
    import mlx_teacache._mflux_versions as versions

    monkeypatch.setattr(versions, "_warned", True)


@contextlib.contextmanager
def expect_distilled_warning(variant_id: str, coefficients: Sequence[float] | None = None) -> Iterator[None]:
    """Wrap an `apply_teacache(...)` call site that may touch a distilled
    variant (registry `default_thresh is None` — currently flux2-klein-4b
    and flux2-klein-9b): asserts `TeaCacheNoBenefitWarning` under
    `pytest.warns(...)` for those variants, and is a no-op otherwise.

    Pass the same `coefficients` the call site hands to `apply_teacache`:
    api.py suppresses the warning when the caller supplies their own tuple,
    so the warning is expected only when `default_thresh is None and
    coefficients is None`.

    Centralizing this against the live `_REGISTRY` (rather than a hardcoded
    variant-id set) means a parity/slow-lane test parametrized across engaged
    AND distilled variants stays correct if a future variant ships with no
    per-variant default: distilled variants get the warning asserted, and
    every other variant in the same parametrize still fails loudly under the
    repo's `filterwarnings = error` if the warning ever fires there."""
    from mlx_teacache import TeaCacheNoBenefitWarning
    from mlx_teacache.variants import _REGISTRY

    entry = _REGISTRY.get(variant_id)
    if entry is not None and entry["default_thresh"] is None and coefficients is None:
        with pytest.warns(TeaCacheNoBenefitWarning, match="distilled"):
            yield
    else:
        yield


@contextlib.contextmanager
def allow_uncalibrated_checkpoint() -> Iterator[None]:
    """Wrap an `apply_teacache(...)` call site on a real-weights lane that may
    load a checkpoint the variant's coefficients were not calibrated on (today:
    Qwen-Image on mflux 0.19 and 0.20, whose `qwen-image` alias is Qwen/Qwen-Image-2512).
    Under `filterwarnings = error` the apply-time
    `TeaCacheUncalibratedCheckpointWarning` would otherwise raise before the
    test measures anything; the lane exists to measure exactly that checkpoint."""
    import warnings

    from mlx_teacache import TeaCacheUncalibratedCheckpointWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TeaCacheUncalibratedCheckpointWarning)
        yield
