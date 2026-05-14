# tests/conftest.py
"""Shared pytest fixtures and marker handling.

The `mflux` marker is auto-applied to any test that lives in a test file
named test_integration*, test_parity*, test_flux1*, test_flux2*, test_forward_*,
test_lifecycle.py, or test_api.py — these all import the integration layer
and therefore require mflux. The test-pure-core CI job skips them via
`-m "not mflux"`."""

from __future__ import annotations

import pytest

_MFLUX_FILES = {
    "test_lifecycle.py",
    "test_forward_flux1.py",
    "test_forward_flux2.py",
    "test_flux1_proxy.py",
    "test_flux2_predict.py",
    "test_api.py",
    "test_parity_flux1.py",
    "test_parity_flux2.py",
    "test_integration_slow.py",
    "test_perf.py",
    "test_detect.py",  # imports mflux types for variant detection
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        path = str(item.path.name)
        if path in _MFLUX_FILES:
            item.add_marker(pytest.mark.mflux)
