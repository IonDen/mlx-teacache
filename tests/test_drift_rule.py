"""The drift guard's rule, decided without an mflux install: an installed release with its own row is held to it;
a release newer than every row is held to the newest row and passes while no fingerprinted function moved; anything
else fails."""

import ast
import textwrap

from mlx_teacache._mflux_versions import NEWEST_VERIFIED_MFLUX, release_tuple
from tests._mflux_surface import drift_failures, drift_reference, fingerprint_function_node
from tests.test_mflux_forward_drift import _TARGETS, KNOWN

# Newest row inserted FIRST on purpose: the reference must be chosen by release number, not dict order.
_TABLE = {"0.21.0": {"f": "b"}, "0.20.0": {"f": "a"}}


def test_a_recorded_release_is_held_to_its_own_row() -> None:
    """Bug: a recorded release is compared against the newest row, so 0.20.0 fails on a digest that moved in 0.21."""
    assert drift_failures("0.20.0", {"f": "a"}, _TABLE) == []


def test_a_newer_release_passes_while_nothing_moved() -> None:
    """Bug: every unrecorded release fails (the pre-0.13 rule), so a harmless new mflux turns CI red."""
    assert drift_failures("0.22.0", {"f": "b"}, _TABLE) == []


def test_a_newer_release_fails_when_a_function_moved() -> None:
    """Bug: an unrecorded newer release is waved through without comparing its fingerprints."""
    failures = drift_failures("0.22.0", {"f": "c"}, _TABLE)
    assert len(failures) == 1
    assert "f changed in mflux 0.22.0" in failures[0] and "0.21.0" in failures[0]


def test_a_newer_release_is_compared_with_the_newest_row_not_an_older_one() -> None:
    """Bug: the reference row is the last-inserted one (_TABLE lists 0.21.0 first, so dict order ends at 0.20.0) or
    the lowest release, so a digest that matches 0.20.0 but not 0.21.0 passes."""
    assert drift_reference("0.22.0", _TABLE) == "0.21.0"
    assert len(drift_failures("0.22.0", {"f": "a"}, _TABLE)) == 1


def test_newest_row_is_chosen_numerically() -> None:
    """Bug: max() over version strings picks "0.9.0" over "0.21.0", so 0.10.0 counts as newer than every row."""
    assert drift_reference("0.10.0", ["0.9.0", "0.21.0"]) is None
    assert drift_reference("0.22.0", ["0.9.0", "0.21.0"]) == "0.21.0"


def test_an_unrecorded_release_inside_the_verified_span_fails() -> None:
    """Bug: an unverified 0.20.5 is compared against the newest row and passes, though its own code was never
    checked."""
    failures = drift_failures("0.20.5", {"f": "b"}, _TABLE)
    assert len(failures) == 1 and "0.20.5 has no verified row" in failures[0]


def test_a_missing_function_counts_as_moved() -> None:
    """Bug: a fingerprint absent from the fresh set (renamed upstream) is skipped instead of failing."""
    assert len(drift_failures("0.22.0", {}, _TABLE)) == 1


def _digest(source: str) -> str:
    node = ast.parse(textwrap.dedent(source)).body[0]
    assert isinstance(node, ast.FunctionDef)
    return fingerprint_function_node(node)


def test_a_one_statement_change_in_real_source_fails_a_newer_release() -> None:
    """Bug: the rule compares something other than the function fingerprint (a version string, a file hash that
    ignores the body), so a real code change upstream passes."""
    original = "def forward(x):\n    y = x + 1\n    return y\n"
    mutated = "def forward(x):\n    y = x + 2\n    return y\n"
    known = {"0.21.0": {"forward": _digest(original)}}
    assert drift_failures("0.22.0", {"forward": _digest(original)}, known) == []
    assert len(drift_failures("0.22.0", {"forward": _digest(mutated)}, known)) == 1


def test_the_newest_row_is_the_runtime_verified_release() -> None:
    """Bug: a row is recorded for a new mflux without moving NEWEST_VERIFIED_MFLUX (or the constant moves with no
    row), so the runtime warning and the drift guard disagree about what was verified."""
    assert max(KNOWN, key=release_tuple) == NEWEST_VERIFIED_MFLUX


def test_every_row_pins_every_target() -> None:
    """Bug: a new row leaves out a target, so that function is silently never compared on that release."""
    labels = {label for label, *_ in _TARGETS}
    assert {version: set(row) for version, row in KNOWN.items()} == {version: labels for version in KNOWN}
