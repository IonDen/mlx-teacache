# tests/test_forward_flux1.py
"""Unit tests for FLUX.1 forward helper. We use a real mflux Transformer
constructed at minimal size (using ModelConfig stubs is hard; use a tiny
custom Transformer if needed) to exercise the prelude/body/tail wiring
without loading real weights. The deep bit-exact parity test against vanilla
mflux lives in tests/test_parity_flux1.py (Task 25)."""

import pytest

pytestmark = pytest.mark.parity  # mflux required; skipped on pure-core CI


def test_threshold_zero_matches_vanilla_on_random_inputs():
    """At rel_l1_thresh=0, every call goes through the body unchanged.
    Output should be bit-exact to vanilla Transformer.__call__."""
    # See Task 25 for the deep parity test using a real Flux1 model + reference latents.
    # This unit-level smoke test uses a synthetic transformer.
    pytest.skip("Deep parity coverage lives in tests/test_parity_flux1.py (Task 25). "
                "This file is a placeholder for any future unit-level forward.py tests.")
