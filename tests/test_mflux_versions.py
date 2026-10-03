"""Release-number parsing and the "newer than verified" comparison behind the mflux version policy."""

import pytest

from mlx_teacache._mflux_versions import is_newer_than_verified, release_tuple


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.21.0", (0, 21, 0)),
        ("0.21", (0, 21, 0)),
        ("0.22.0rc1", (0, 22, 0)),
        ("0.21.0.dev3+gabc1234", (0, 21, 0)),
        ("1", (1, 0, 0)),
        ("0.100.0", (0, 100, 0)),
        (" 0.21.1 ", (0, 21, 1)),
    ],
)
def test_release_tuple_reads_the_numeric_release(text: str, expected: tuple[int, ...]) -> None:
    """Bug: a pre-release or dev suffix breaks parsing, or "0.21" and "0.21.0" compare as different releases."""
    assert release_tuple(text) == expected


@pytest.mark.parametrize("text", ["unknown", "", "v0.21.0"])
def test_release_tuple_is_none_without_a_leading_number(text: str) -> None:
    """Bug: an unparseable version string raises instead of being treated as unknown."""
    assert release_tuple(text) is None


@pytest.mark.parametrize(
    ("installed", "verified", "newer"),
    [
        ("0.21.1", "0.21.0", True),
        ("0.22.0rc1", "0.21.0", True),
        ("0.100.0", "0.21.0", True),
        ("0.21.0", "0.21.0", False),
        ("0.21.0.dev3+gabc", "0.21.0", False),
        ("0.20.9", "0.21.0", False),
        (None, "0.21.0", False),
        ("unknown", "0.21.0", False),
    ],
)
def test_is_newer_than_verified_compares_numerically(
    installed: str | None, verified: str, newer: bool
) -> None:
    """Bug: versions compared as strings ("0.100.0" < "0.21.0"), or the verified release itself counted as newer."""
    assert is_newer_than_verified(installed, verified) is newer
