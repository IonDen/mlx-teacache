"""The variant registry must isolate each variant's import + metadata so a single
malformed variant fails `import mlx_teacache` with a CalibrationError that NAMES
the offending subpackage — not an opaque ImportError / AttributeError / KeyError
(backlog 0031 #4). v0.8.0 already validated COEFFICIENTS; these pin the import
isolation + META-key validation that were still missing.
"""

import importlib
import types

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


_VALID_CONFIG = types.SimpleNamespace(
    META={"variant_id": "phantom", "display_name": "Phantom", "license": "L"},
    COEFFICIENTS=(0.0, 0.0, 0.0, 1.0, 0.0),
    DEFAULT_THRESH=0.2,
)


@pytest.mark.parametrize(
    ("detect", "fragment"),
    [
        (types.SimpleNamespace(matches=lambda f: False, MODEL_NAMES=("p",)), "matches_config"),
        (
            types.SimpleNamespace(matches=lambda f: False, matches_config=lambda c, k: False, MODEL_NAMES=()),
            "MODEL_NAMES",
        ),
        (
            types.SimpleNamespace(
                matches=lambda f: False, matches_config=lambda c, k: False, MODEL_NAMES=["p"]
            ),
            "MODEL_NAMES",
        ),
        (
            types.SimpleNamespace(
                matches=lambda f: False, matches_config=lambda c, k: False, MODEL_NAMES=("",)
            ),
            "MODEL_NAMES",
        ),
        (
            types.SimpleNamespace(
                matches=lambda f: False,
                matches_config=lambda c, k: False,
                MODEL_NAMES=("p",),
                is_calibrated_checkpoint=lambda c, name: True,
            ),
            "hf_model_id",
        ),
    ],
)
def test_build_one_requires_the_config_level_detector(monkeypatch, detect, fragment) -> None:
    """Bug: a variant without matches_config or usable MODEL_NAMES (missing, a list, an empty name a user could never
    type), or with a checkpoint check but no META["hf_model_id"] to check against, registers, and match_variant then
    fails with a KeyError or AttributeError for every caller instead of the import naming the broken variant."""
    modules = {
        "mlx_teacache.variants.phantom.config": _VALID_CONFIG,
        "mlx_teacache.variants.phantom.detect": detect,
    }
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib, "import_module", lambda name, *a, **k: modules.get(name) or real_import(name, *a, **k)
    )
    with pytest.raises(CalibrationError, match=fragment) as caught:
        _build_one("mlx_teacache.variants.phantom", "phantom")
    assert caught.value.variant_id == "phantom"
