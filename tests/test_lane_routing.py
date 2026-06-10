# tests/test_lane_routing.py
"""Guards the marker-lane routing (no model loads — pure collection introspection)."""

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _collected_node_ids(marker_expr: str) -> set[str]:
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            marker_expr,
            "tests/test_flux1_proxy.py",
            "tests/test_flux2_predict.py",
            "tests/test_api.py",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert out.returncode in (0, 5), (
        f"pytest collection subprocess failed (rc={out.returncode}): {out.stderr}"
    )
    return {ln for ln in out.stdout.splitlines() if "::" in ln}


def test_proxy_and_predict_run_in_pure_core_lane():
    ids = _collected_node_ids("not parity and not mflux")
    assert any("test_flux1_proxy.py" in i for i in ids)
    assert any("test_flux2_predict.py" in i for i in ids)
    # test_api.py stays mflux-gated (real Flux1 instantiation) -> NOT collected here.
    assert not any("test_api.py" in i for i in ids)


def test_api_real_model_tests_stay_parity_gated():
    ids = _collected_node_ids("not parity")
    # All real-Flux2Klein tests must NOT appear when parity is excluded.
    real_model_tests = (
        "test_apply_teacache_accepts_flux2_klein_9b",
        "test_apply_teacache_accepts_flux2_klein_base_4b",
        "test_apply_teacache_uses_per_variant_default_for_klein_base_4b",
        "test_apply_teacache_explicit_thresh_overrides_per_variant_default",
        "test_apply_teacache_user_coefficients_skip_per_variant_default",
        "test_apply_teacache_cfg_records_cfg_was_active_klein_base_4b",
        "test_invalid_skip_window_raises_under_cfg_klein_base_4b",
    )
    for name in real_model_tests:
        assert not any(name in i for i in ids), f"{name} leaked out of the parity gate"
