"""The variant registry must isolate each variant's import + metadata so a single
malformed variant fails `import mlx_teacache` with a CalibrationError that NAMES
the offending subpackage — not an opaque ImportError / AttributeError / KeyError
(backlog 0031 #4). v0.8.0 already validated COEFFICIENTS; these pin the import
isolation + META-key validation that were still missing.
"""

import importlib

import pytest

from mlx_teacache.errors import CalibrationError
from mlx_teacache.variants import _build_one, _validate_meta


def test_validate_meta_rejects_non_dict() -> None:
    with pytest.raises(CalibrationError) as ei:
        _validate_meta(None, subname="ghost")
    assert ei.value.variant_id == "ghost"


def test_validate_meta_rejects_missing_required_key() -> None:
    with pytest.raises(CalibrationError) as ei:
        _validate_meta({"display_name": "X", "license": "Y"}, subname="ghost")
    assert "variant_id" in ei.value.reason


def test_validate_meta_accepts_complete_meta() -> None:
    meta = {"variant_id": "v", "display_name": "D", "license": "L"}
    assert _validate_meta(meta, subname="v") is meta


def test_build_one_wraps_import_failure_in_named_calibration_error(monkeypatch) -> None:
    real_import = importlib.import_module

    def fake_import(name, *args, **kwargs):
        if name == "mlx_teacache.variants.phantom.config":
            raise RuntimeError("boom in config")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(CalibrationError) as ei:
        _build_one("mlx_teacache.variants.phantom", "phantom")
    assert ei.value.variant_id == "phantom"
