"""Drift detection for the FLUX HuggingFace revisions used to generate the
committed reference fixtures. Runs before parity tests so an upstream model
update produces a clear remediation message instead of silent bit-exact
failures."""

import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.parity

_REVISIONS_TOML = Path(__file__).parent / "hf_revisions.toml"
_HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"


def _load_pins() -> dict[str, str]:
    data = tomllib.loads(_REVISIONS_TOML.read_text())
    return {k: v["sha"] for k, v in data.get("revisions", {}).items()}


def _repo_to_cache_dir(repo_id: str) -> Path:
    return _HF_CACHE_ROOT / f"models--{repo_id.replace('/', '--')}"


@pytest.mark.parametrize("repo_id", list(_load_pins().keys()))
def test_hf_revision_matches_pin(repo_id: str) -> None:
    pinned = _load_pins()[repo_id]
    cache_dir = _repo_to_cache_dir(repo_id)
    main_ref = cache_dir / "refs" / "main"
    if not main_ref.exists():
        pytest.skip(
            f"HF cache for {repo_id} not present. Run `hf download {repo_id} "
            f"--revision {pinned}` to populate it before parity tests."
        )
    actual = main_ref.read_text().strip()
    assert actual == pinned, (
        f"HF revision drift for {repo_id}:\n"
        f"  pinned (used to generate fixtures): {pinned}\n"
        f"  local cache refs/main:             {actual}\n"
        f"To restore: `hf download {repo_id} --revision {pinned}`. "
        f"To re-pin: update tests/hf_revisions.toml and regenerate the "
        f"affected fixtures via tests/generate_references.py."
    )
