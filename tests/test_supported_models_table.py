"""README "Supported models" table: popularity order, registry coverage, README in sync (pure-core lane)."""

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "docs"))
import _generate_supported_models as gen  # noqa: E402


def _row_ids(table: str) -> list[str]:
    return [line.split("`")[1] for line in table.splitlines()[2:]]


def test_table_lists_variants_most_popular_first_flux1_last() -> None:
    """Bug: the generator sorts alphabetically, so FLUX.1 (legacy) leads and Qwen-Image sits low."""
    assert _row_ids(gen._build_table()) == [
        "qwen-image-2512",
        "qwen-image",
        "flux2-klein-base-9b",
        "flux2-klein-base-4b",
        "flux2-klein-4b",
        "flux2-klein-9b",
        "z-image-base",
        "flux1-dev",
        "flux1-krea-dev",
        "flux1-schnell",
    ]


def test_order_missing_a_registered_variant_raises() -> None:
    """Bug: a new variant not added to the order list drops out of the README table silently."""
    with pytest.raises(RuntimeError, match=r"missing=\['new-variant'\]"):
        gen._ordered_ids({"a", "b", "new-variant"}, ("a", "b"))


def test_order_naming_an_unregistered_variant_raises() -> None:
    """Bug: a removed or misspelled id in the order list is skipped instead of reported."""
    with pytest.raises(RuntimeError, match=r"unknown=\['gone'\]"):
        gen._ordered_ids({"a", "b"}, ("a", "gone", "b"))


def test_order_with_a_duplicate_raises() -> None:
    """Bug: a duplicated id prints the same row twice."""
    with pytest.raises(RuntimeError, match=r"duplicates=\['a'\]"):
        gen._ordered_ids({"a", "b"}, ("a", "b", "a"))


def test_order_matching_the_registry_keeps_the_given_order() -> None:
    """Bug: the helper re-sorts instead of keeping the hand-picked order."""
    assert gen._ordered_ids({"a", "b", "c"}, ("c", "a", "b")) == ["c", "a", "b"]


def test_readme_block_matches_the_generator() -> None:
    """Bug: the README table was hand-edited or not regenerated after a variant or order change."""
    readme = (_REPO / "README.md").read_text()
    start = readme.index(gen.START_MARKER) + len(gen.START_MARKER)
    end = readme.index(gen.END_MARKER)
    assert readme[start:end].strip() == gen._build_table()
