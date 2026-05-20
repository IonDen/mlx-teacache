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
        "source", "revision", "calibration_dataset",
        "fit_metric", "fit_metric_value", "reference_url", "default_thresh",
    }


def test_legacy_coefficients_re_exports_provenance_identity():
    from mlx_teacache._kernel.coefficients import Provenance as KP
    from mlx_teacache.coefficients import Provenance as LP
    assert LP is KP


