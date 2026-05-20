import pytest

from mlx_teacache.coefficients import (
    Provenance,
    validate_custom,
)


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
