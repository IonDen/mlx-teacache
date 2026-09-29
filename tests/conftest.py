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
session cap means even a misrouted heavy test runs slow (or fails
cleanly with an MLX allocation error) instead of taking the machine
down. See CLAUDE.md "Memory guardrails for heavy generations" and
ml-explore/mlx-lm #883 for the upstream confirmation that wired
memory — not the soft `set_memory_limit` — is the root cause."""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from tests._lanes import is_mflux_file
from tests._memory_guard import apply_mlx_memory_caps


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
