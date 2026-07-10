# tests/conftest.py
"""Shared pytest fixtures and marker handling.

The `mflux` marker is auto-applied to every test whose file name is in the
explicit `_MFLUX_FILES` allowlist below (matched exactly, not by glob) — these
all import the integration layer and therefore require mflux. The
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

import sys

import pytest

from tests._memory_guard import apply_mlx_memory_caps


def _install_mlx_memory_caps() -> None:
    """Hard-cap Metal wired memory before any test imports MLX models."""
    try:
        import mlx.core as mx
    except ImportError:
        return
    apply_mlx_memory_caps(mx, lambda message: print(f"mlx-teacache tests: {message}", file=sys.stderr))


_install_mlx_memory_caps()


_MFLUX_FILES = {
    "test_lifecycle.py",
    "test_forward_flux1.py",
    "test_forward_flux2.py",
    "test_cfg_branch_independence.py",  # calls flux2_cfg_forward_with_gate which lazily imports mflux
    "test_api.py",
    "test_parity_flux1.py",
    "test_parity_flux2.py",
    "test_parity_z_image.py",
    "test_parity_qwen.py",
    "test_image_quality_flux1.py",
    "test_image_quality_flux2.py",
    "test_integration_slow.py",
    "test_perf.py",
    "test_detect.py",  # imports mflux types for variant detection
    "test_mflux_contract_smoke.py",
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        path = str(item.path.name)
        if path in _MFLUX_FILES:
            item.add_marker(pytest.mark.mflux)
