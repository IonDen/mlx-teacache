"""T5 discipline: Provenance lives in _kernel.coefficients now; the legacy
mlx_teacache.coefficients path re-exports it for compat."""

from __future__ import annotations


def test_provenance_accessible_from_kernel():
    from mlx_teacache._kernel.coefficients import Provenance

    p = Provenance.for_user_supplied()
    assert p.source == "user"


def test_provenance_field_set_matches_v05():
    import dataclasses

    from mlx_teacache._kernel.coefficients import Provenance

    actual = {f.name for f in dataclasses.fields(Provenance)}
    assert actual == {
        "source",
        "revision",
        "calibration_dataset",
        "fit_metric",
        "fit_metric_value",
        "reference_url",
        "default_thresh",
    }


def test_legacy_coefficients_re_exports_provenance_identity():
    from mlx_teacache._kernel.coefficients import Provenance as KP
    from mlx_teacache.coefficients import Provenance as LP

    assert LP is KP


def test_validate_custom_rejects_non_sequence():
    import pytest

    from mlx_teacache._kernel.coefficients import validate_custom

    with pytest.raises(ValueError, match="must be a sequence"):
        validate_custom(42)  # type: ignore[arg-type]


def test_validate_custom_rejects_wrong_length():
    import pytest

    from mlx_teacache._kernel.coefficients import validate_custom

    with pytest.raises(ValueError, match="length 5"):
        validate_custom([1.0, 2.0, 3.0])


def test_validate_custom_rejects_non_float_convertible():
    import pytest

    from mlx_teacache._kernel.coefficients import validate_custom

    with pytest.raises(ValueError, match="convertible to float"):
        validate_custom([1.0, 2.0, 3.0, 4.0, object()])


def test_validate_custom_rejects_non_finite():
    import math

    import pytest

    from mlx_teacache._kernel.coefficients import validate_custom

    with pytest.raises(ValueError, match="finite"):
        validate_custom([1.0, 2.0, 3.0, 4.0, math.inf])
