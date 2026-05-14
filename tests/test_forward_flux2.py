# tests/test_forward_flux2.py
"""Unit tests for FLUX.2 forward helper.

Deep bit-exact parity coverage lives in tests/test_parity_flux2.py (Task 26).
This file is a placeholder for any future unit-level tests against synthetic
Flux2Transformer-shaped fakes."""

import pytest

pytestmark = pytest.mark.parity


def test_threshold_zero_matches_vanilla_on_random_inputs():
    pytest.skip("Deep parity coverage lives in tests/test_parity_flux2.py (Task 26). "
                "Synthetic Flux2Transformer testing is deferred — the parity test "
                "against a real Flux2Klein 4b at threshold=0 is the correctness gate.")
