"""Invariants for the committed coefficient + calibration artifacts.

These guard two bug classes the rest of the suite missed:
  1. The wrong-order-coefficient transcription bug (the FLUX.1-dev error fixed
     2026-05-15: predicted distances ~10x too large, cache never engaged).
  2. config <-> calibration drift: a recalibration that updates
     scripts/_calibration_z_image.json but not the shipped z-image config
     (or a hand-edit to any variant's COEFFICIENTS) must turn a test red.

Pure-core: imports variant configs (mflux-free) and reads committed JSON.
"""

import importlib
import json
import math
import pkgutil
from pathlib import Path

import pytest

import mlx_teacache.variants as _variants_pkg

_REPO_ROOT = Path(__file__).resolve().parent.parent
_Z_IMAGE_CALIB = _REPO_ROOT / "scripts" / "_calibration_z_image.json"

# Published TeaCache4FLUX rescale-poly coefficients, numpy poly1d high-to-low.
# Source: https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4FLUX/teacache_flux.py
_UPSTREAM_FLUX1_DEV = (498.651651, -283.781631, 55.8554382, -3.82021401, 0.264230861)


def _isclose_seq(actual, expected, *, rel_tol: float, abs_tol: float = 0.0) -> bool:
    return len(actual) == len(expected) and all(
        math.isclose(float(a), float(e), rel_tol=rel_tol, abs_tol=abs_tol)
        for a, e in zip(actual, expected, strict=True)
    )


def _all_variant_coefficients() -> list[tuple[str, tuple[float, ...]]]:
    """(variant_id, COEFFICIENTS) for every registered variant, enumerated by
    walking the variants package the same way the registry does — so a new
    variant shipping a malformed coefficient tuple is caught without editing
    this test."""
    out: list[tuple[str, tuple[float, ...]]] = []
    for _, subname, ispkg in pkgutil.iter_modules(_variants_pkg.__path__):
        if not ispkg:
            continue
        config = importlib.import_module(f"mlx_teacache.variants.{subname}.config")
        out.append((config.META["variant_id"], config.COEFFICIENTS))
    return out


_VARIANT_COEFFS = _all_variant_coefficients()


@pytest.mark.parametrize("variant_id,coeffs", _VARIANT_COEFFS, ids=[vid for vid, _ in _VARIANT_COEFFS])
def test_builtin_coefficients_are_length5_and_finite(variant_id: str, coeffs: tuple[float, ...]) -> None:
    assert len(coeffs) == 5, f"{variant_id}: expected 5 coefficients, got {len(coeffs)}"
    assert all(isinstance(c, float) for c in coeffs), f"{variant_id}: coefficients must be floats"
    assert all(math.isfinite(c) for c in coeffs), f"{variant_id}: non-finite coefficient in {coeffs}"


def test_flux1_dev_coefficients_match_upstream_teacache4flux() -> None:
    """Pin the shipped FLUX.1-dev poly to the vendored upstream values. A
    wrong-order transcription (the 2026-05-15 bug) is off by orders of magnitude
    and breaches this; precision-only differences stay within tolerance."""
    from mlx_teacache.variants.flux1_dev.config import COEFFICIENTS

    assert _isclose_seq(COEFFICIENTS, _UPSTREAM_FLUX1_DEV, rel_tol=1e-5, abs_tol=1e-6), (
        f"FLUX.1-dev COEFFICIENTS {COEFFICIENTS} diverge from upstream "
        f"TeaCache4FLUX {_UPSTREAM_FLUX1_DEV} (wrong order?)"
    )


def _load_z_image_calibration() -> dict:
    return json.loads(_Z_IMAGE_CALIB.read_text())


@pytest.mark.parametrize("signal", ["A", "B"])
def test_z_image_calibration_signal_coefficients_length5_finite(signal: str) -> None:
    coeffs = _load_z_image_calibration()["signals"][signal]["coefficients_c4_to_c0"]
    assert len(coeffs) == 5, f"signal {signal}: expected 5 coefficients, got {len(coeffs)}"
    assert all(math.isfinite(float(c)) for c in coeffs), f"signal {signal}: non-finite coefficient"


@pytest.mark.parametrize("signal", ["A", "B"])
def test_z_image_calibration_r_squared_in_unit_range(signal: str) -> None:
    sig = _load_z_image_calibration()["signals"][signal]
    for key in ("fit_r_squared", "heldout_r_squared"):
        r2 = float(sig[key])
        assert math.isfinite(r2), f"signal {signal}: {key} is not finite"
        assert 0.0 <= r2 <= 1.0, f"signal {signal}: {key}={r2} outside [0, 1] (broken fit?)"


def test_z_image_signal_b_fit_beats_signal_a() -> None:
    """Signal B is the selected fit; the documented selection reason is that B's
    R^2 beats A's. A recalibration that inverts this ordering would mean the
    wrong signal feeds the shipped poly."""
    signals = _load_z_image_calibration()["signals"]
    assert float(signals["B"]["fit_r_squared"]) > float(signals["A"]["fit_r_squared"])


def test_z_image_config_coefficients_match_calibration_signal_b() -> None:
    """The shipped z-image config is read verbatim from signal B of the committed
    calibration. A recalibration that updates the JSON but not the config (or
    vice versa) drifts and must red."""
    from mlx_teacache.variants.z_image_base.config import COEFFICIENTS

    json_b = _load_z_image_calibration()["signals"]["B"]["coefficients_c4_to_c0"]
    assert _isclose_seq(COEFFICIENTS, json_b, rel_tol=1e-9, abs_tol=1e-12), (
        f"z-image config COEFFICIENTS {COEFFICIENTS} drifted from "
        f"scripts/_calibration_z_image.json signal B {json_b}"
    )


_QWEN_CALIB = _REPO_ROOT / "scripts" / "_calibration_qwen.json"


def _load_qwen_calibration() -> dict:
    return json.loads(_QWEN_CALIB.read_text())


@pytest.mark.parametrize("signal", ["A", "B"])
def test_qwen_calibration_signal_coefficients_length5_finite(signal: str) -> None:
    coeffs = _load_qwen_calibration()["signals"][signal]["coefficients_c4_to_c0"]
    assert len(coeffs) == 5, f"signal {signal}: expected 5 coefficients, got {len(coeffs)}"
    assert all(math.isfinite(float(c)) for c in coeffs), f"signal {signal}: non-finite coefficient"


@pytest.mark.parametrize("signal", ["A", "B"])
def test_qwen_calibration_r_squared_in_unit_range(signal: str) -> None:
    sig = _load_qwen_calibration()["signals"][signal]
    for key in ("fit_r_squared", "heldout_r_squared"):
        r2 = float(sig[key])
        assert math.isfinite(r2), f"signal {signal}: {key} is not finite"
        assert 0.0 <= r2 <= 1.0, f"signal {signal}: {key}={r2} outside [0, 1] (broken fit?)"


def test_qwen_config_coefficients_match_calibration_signal_a() -> None:
    """The shipped qwen-image config is read verbatim from signal A of the committed
    calibration. Signal A is the SELECTED gate signal (caption-independent + cheaper
    skips) even though signal B's R^2 is marginally higher — see the config docstring.
    A recalibration that updates the JSON but not the config (or vice versa) drifts
    and must red."""
    from mlx_teacache.variants.qwen_image.config import COEFFICIENTS

    json_a = _load_qwen_calibration()["signals"]["A"]["coefficients_c4_to_c0"]
    assert _isclose_seq(COEFFICIENTS, json_a, rel_tol=1e-9, abs_tol=1e-12), (
        f"qwen-image config COEFFICIENTS {COEFFICIENTS} drifted from "
        f"scripts/_calibration_qwen.json signal A {json_a}"
    )


# ---------------------------------------------------------------------------
# flux1-krea-dev: config <-> committed calibration JSON (scripts/calibrate_flux1.py)
# ---------------------------------------------------------------------------

_KREA_CALIB = _REPO_ROOT / "scripts" / "_calibration_flux1_krea_dev.json"


def _load_krea_calibration() -> dict:
    return json.loads(_KREA_CALIB.read_text())


def test_krea_calibration_is_a_full_ten_prompt_capture_at_the_model_card_recipe() -> None:
    # bug caught: shipping a fit from a partial capture, or at dev's 25-step / 3.5 recipe
    d = _load_krea_calibration()
    assert d["model"] == "krea-dev"
    assert d["recipe"] == {"num_inference_steps": 28, "guidance": 4.5}
    assert d["n_prompts"] == 10 and d["n_pairs"] == 270
    assert len(d["x_values"]) == len(d["y_values"]) == 270


def test_krea_calibration_r_squared_in_unit_range_and_dev_tuple_scored() -> None:
    # bug caught: dropping the scored-dev record, which is the evidence reuse was rejected
    d = _load_krea_calibration()
    assert 0.0 <= float(d["r2"]) <= 1.0
    assert math.isfinite(float(d["scored_r2"]))
    assert float(d["scored_r2"]) < 0.0, (
        "FLUX.1-dev's tuple does not fit Krea's pairs; if it does now, re-decide"
    )


def test_krea_config_coefficients_match_the_committed_fit() -> None:
    from mlx_teacache.variants.flux1_krea_dev.config import COEFFICIENTS

    fit = _load_krea_calibration()["coefficients_c4_to_c0"]
    assert _isclose_seq(COEFFICIENTS, fit, rel_tol=1e-9, abs_tol=1e-12), (
        f"config {COEFFICIENTS} != scripts/_calibration_flux1_krea_dev.json {fit}"
    )
