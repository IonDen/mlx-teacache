"""Pin reference fixtures by SHA-256. Catches accidental fixture drift."""

import hashlib
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.parity

_FIXTURES_TOML = Path(__file__).parent / "fixtures.toml"
_REFERENCE_ROOT = Path(__file__).parent / "reference"


def _load_pins() -> dict[str, str]:
    data = tomllib.loads(_FIXTURES_TOML.read_text())
    return {k: v["sha256"] for k, v in data.get("fixtures", {}).items()}


def test_all_pinned_fixtures_match_sha256():
    pins = _load_pins()
    for relpath, expected in pins.items():
        path = _REFERENCE_ROOT / relpath
        assert path.exists(), f"Pinned fixture missing: {path}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, (
            f"Fixture SHA mismatch for {relpath}:\n  expected={expected}\n  actual  ={actual}"
        )


def test_every_reference_file_is_pinned():
    pins = _load_pins()
    pinned_paths = set(pins)
    actual_files = {
        str(p.relative_to(_REFERENCE_ROOT))
        for p in _REFERENCE_ROOT.rglob("*.safetensors")
    }
    assert actual_files == pinned_paths, (
        f"fixtures.toml out of sync. "
        f"Files not pinned: {actual_files - pinned_paths}. "
        f"Pins without files: {pinned_paths - actual_files}."
    )
