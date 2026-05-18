import math

import pytest

from mlx_teacache.coefficients import (
    Provenance,
    load_builtin,
    validate_custom,
)
from mlx_teacache.errors import CalibrationError


def test_load_builtin_flux1_dev_returns_5_coeffs_and_provenance():
    coeffs, prov = load_builtin("flux1-dev")
    assert isinstance(coeffs, tuple)
    assert len(coeffs) == 5
    assert all(math.isfinite(c) for c in coeffs)
    assert prov.source == "builtin"
    assert prov.revision == "upstream-flux-v1"
    assert "ali-vilab" in (prov.reference_url or "")


def test_load_builtin_flux1_schnell_reuses_dev_coeffs_with_shared_revision():
    dev_coeffs, _ = load_builtin("flux1-dev")
    schnell_coeffs, schnell_prov = load_builtin("flux1-schnell")
    assert schnell_coeffs == dev_coeffs
    assert schnell_prov.revision == "upstream-flux-v1-shared"


def test_load_builtin_flux2_klein_4b_has_dataset_and_metric():
    coeffs, prov = load_builtin("flux2-klein-4b")
    assert len(coeffs) == 5
    assert prov.source == "builtin"
    assert prov.revision == "in-repo-2026-05-15"
    assert prov.calibration_dataset is not None
    assert prov.fit_metric is not None
    assert prov.fit_metric_value is not None
    assert 0.0 < prov.fit_metric_value <= 1.0


def test_load_builtin_flux2_klein_9b_has_dataset_and_metric():
    coeffs, prov = load_builtin("flux2-klein-9b")
    assert len(coeffs) == 5
    assert all(math.isfinite(c) for c in coeffs)
    assert prov.source == "builtin"
    assert prov.revision == "in-repo-2026-05-16-origin"
    assert prov.calibration_dataset is not None
    assert prov.fit_metric is not None
    assert prov.fit_metric_value is not None
    assert 0.0 < prov.fit_metric_value <= 1.0
    assert (prov.reference_url or "").endswith("calibrate_flux2.py")


def test_load_builtin_flux2_klein_base_4b_has_dataset_and_metric():
    coeffs, prov = load_builtin("flux2-klein-base-4b")
    assert len(coeffs) == 5
    assert all(math.isfinite(c) for c in coeffs)
    assert coeffs[-1] == 0.0  # origin-constrained: poly(0) = 0
    assert prov.source == "builtin"
    assert prov.revision is not None and prov.revision.startswith("in-repo-2026-05-")
    assert prov.calibration_dataset is not None
    assert "25 steps" in prov.calibration_dataset
    assert "guidance=1.0" in prov.calibration_dataset
    assert prov.fit_metric is not None
    assert prov.fit_metric_value is not None
    assert 0.0 < prov.fit_metric_value <= 1.0
    assert (prov.reference_url or "").endswith("calibrate_flux2.py")
    assert prov.default_thresh == 0.17


def test_klein_4b_9b_have_no_per_variant_default():
    """Distilled Klein variants intentionally keep the package-wide 0.20 default."""
    _, prov_4b = load_builtin("flux2-klein-4b")
    _, prov_9b = load_builtin("flux2-klein-9b")
    assert prov_4b.default_thresh is None
    assert prov_9b.default_thresh is None


def test_klein_base_9b_reuses_base_4b_polynomial():
    """v0.5.0 ships klein-base-9b with base-4b's coefficients verbatim.

    The reuse is intentional. If this test fails, either the 9B entry was
    edited (and the change should be deliberate, not silent) or the 4B
    entry drifted.
    """
    coeffs_4b, _ = load_builtin("flux2-klein-base-4b")
    coeffs_9b, _ = load_builtin("flux2-klein-base-9b")
    assert coeffs_9b == coeffs_4b


def test_klein_base_9b_default_thresh_017():
    """klein-base-9b ships with the same default threshold as base-4b."""
    _, prov = load_builtin("flux2-klein-base-9b")
    assert prov.default_thresh == 0.17


def test_flux1_variants_have_no_per_variant_default():
    """FLUX.1 variants intentionally keep the package-wide 0.20 default."""
    _, prov_dev = load_builtin("flux1-dev")
    _, prov_schnell = load_builtin("flux1-schnell")
    assert prov_dev.default_thresh is None
    assert prov_schnell.default_thresh is None


@pytest.mark.parametrize(
    "variant_id",
    [
        "flux1-dev",
        "flux1-schnell",
        "flux2-klein-4b",
        "flux2-klein-9b",
        "flux2-klein-base-4b",
        "flux2-klein-base-9b",
    ],
)
def test_every_supported_variant_has_builtin_coefficients(variant_id):
    coeffs, prov = load_builtin(variant_id)
    assert isinstance(coeffs, tuple)
    assert len(coeffs) == 5
    assert all(math.isfinite(c) for c in coeffs)
    assert prov.source == "builtin"


def test_load_builtin_unknown_variant_raises_calibration_error():
    with pytest.raises(CalibrationError) as excinfo:
        load_builtin("flux42-mythical")
    assert "flux42-mythical" in str(excinfo.value)


def test_validate_custom_accepts_length_5_finite():
    out = validate_custom([1.0, -0.5, 0.1, 0.0, 0.3])
    assert out == (1.0, -0.5, 0.1, 0.0, 0.3)


def test_validate_custom_rejects_wrong_length():
    with pytest.raises(ValueError, match="length 5"):
        validate_custom([1.0, 2.0, 3.0])


def test_validate_custom_rejects_nan():
    with pytest.raises(ValueError, match="finite"):
        validate_custom([1.0, 2.0, float("nan"), 4.0, 5.0])


def test_validate_custom_rejects_inf():
    with pytest.raises(ValueError, match="finite"):
        validate_custom([1.0, float("inf"), 3.0, 4.0, 5.0])


def test_provenance_for_user_supplied():
    prov = Provenance.for_user_supplied()
    assert prov.source == "user"
    assert prov.revision is None
    assert prov.calibration_dataset is None
