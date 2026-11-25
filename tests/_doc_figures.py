"""Shared helpers for tests that pin a figure printed in a doc to the committed record it comes from.

A figure's place is a table cell (``Row``: the cell under one header, in the one row whose first cell is given) or a
hand-typed regex (``Near``) in which ``{f}`` marks the figure and ``{n}`` any other number. ``doc_prints`` is true
when the doc prints the expected figure at that place and the place occurs exactly once. Records are parsed with
every float kept as its decimal text, and ``half_up`` rounds the way the docs print. Pure: no MLX, no mflux.
"""

import json
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Row:
    """The cell under the one header starting with ``column``, in the one table row whose first cell is the row key.

    With ``pattern`` unset the whole cell is the figure. With it set (``{f}`` for the figure, ``{n}`` for any other
    number), the figure is what the pattern captures inside the cell, for cells such as ``2.68×¹`` or ``33 of 50``."""

    column: str
    pattern: str | None = None


@dataclass(frozen=True, slots=True)
class Near:
    """A hand-typed regex for the figure's sentence: ``{f}`` marks the figure, ``{n}`` any other number. It must
    match exactly once in the doc, read with line breaks, indents and ``#`` comment marks collapsed to one space."""

    pattern: str


Place = Row | Near


def load_record(record_path: str) -> dict[str, Any]:
    """Parse a record with every float kept as its exact decimal text (no binary rounding before half-up)."""
    text = (REPO_ROOT / record_path).read_text(encoding="utf-8")
    parsed: dict[str, Any] = json.loads(text, parse_float=Decimal)
    return parsed


def half_up(value: Decimal, digits: int) -> str:
    """``value`` rounded half-up to ``digits`` decimals, as the docs print it."""
    return str(value.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


# ``{f}`` captures the whole number (never 0.97 out of 0.976); ``{n}`` stands for any other number.
FIGURE = r"(?<![\d.])(\d+(?:\.\d+)?)(?!\d)"
NUMBER = r"\d+(?:\.\d+)?"


def compile_place(pattern: str) -> re.Pattern[str]:
    """``pattern`` with ``{f}`` and ``{n}`` expanded; it must hold exactly one ``{f}``."""
    compiled = re.compile(pattern.replace("{f}", FIGURE).replace("{n}", NUMBER))
    assert compiled.groups == 1, f"anchor {pattern!r} needs exactly one {{f}}"
    return compiled


def table_cells(text: str, row_key: str, column: str) -> list[str]:
    """Every cell under a header starting with ``column`` in a table row whose first cell is ``row_key``."""
    cells: list[str] = []
    header: list[str] | None = None
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            header = None
            continue
        row = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if header is None:
            header = row
            continue
        columns = [i for i, name in enumerate(header) if name.startswith(column)]
        if row[0] == row_key and len(columns) == 1 and columns[0] < len(row):
            cells.append(row[columns[0]])
    return cells


def flatten(text: str) -> str:
    """``text`` as one line: line breaks, indents and leading ``#`` comment marks become one space."""
    lines = (re.sub(r"^\s*#*\s*", "", line) for line in text.splitlines())
    return re.sub(r"\s+", " ", " ".join(lines))


def figures_at(text: str, where: Place, row_key: str = "") -> list[str]:
    """Every figure ``text`` prints at the place: one item when the place is unambiguous."""
    if isinstance(where, Row):
        cells = table_cells(text, row_key, where.column)
        if where.pattern is None:
            return cells
        inner = compile_place(where.pattern)
        found: list[str] = []
        for cell in cells:
            found.extend(inner.findall(cell) or [cell])
        return found
    return compile_place(where.pattern).findall(flatten(text))


def doc_prints(text: str, where: Place, published: str, row_key: str = "") -> bool:
    """True when ``text`` prints ``published`` at the place, and that place occurs exactly once."""
    return figures_at(text, where, row_key) == [published]


# --- What a record leaves out, and how a doc says so --------------------------------------------------------

PROVENANCE_FIELDS = ("commit", "dirty", "date", "mlx", "mflux", "hardware")


def _provenance_blocks(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Where a record keeps its provenance: the top level, its ``hardware`` block, and (in the comparison report)
    each variant's ``provenance``."""
    blocks = [record, record.get("hardware") or {}]
    variants = record.get("variants")
    if isinstance(variants, dict):
        blocks += [v.get("provenance") or {} for v in variants.values() if isinstance(v, dict)]
    return blocks


_FIELD_KEYS: dict[str, tuple[str, ...]] = {
    "commit": ("git_commit", "git_sha_a", "git_sha_b"),
    "dirty": ("git_dirty",),
    "date": ("generated_at",),
    "mlx": ("mlx_version", "version_mlx"),
    "mflux": ("mflux_version", "version_mflux"),
    "hardware": ("chip",),
}


def missing_provenance(record: dict[str, Any]) -> set[str]:
    """The provenance fields the record does not have, by key presence (``git_dirty: false`` is present). In the
    comparison report a field held only per variant is present only when every variant holds it."""
    shared = [record, record.get("hardware") or {}]
    variants = record.get("variants")
    each = (
        [v.get("provenance") or {} for v in variants.values() if isinstance(v, dict)]
        if isinstance(variants, dict)
        else []
    )

    def has(block: dict[str, Any], keys: tuple[str, ...]) -> bool:
        return any(k in block for k in keys)

    return {
        field
        for field, keys in _FIELD_KEYS.items()
        if not any(has(b, keys) for b in shared) and not (each and all(has(b, keys) for b in each))
    }


def dirty_at_run_time(record: dict[str, Any]) -> bool:
    """True when the record says its working tree had uncommitted changes when the run started."""
    return any(b.get("git_dirty") is True for b in _provenance_blocks(record))


# What a doc must say for each gap: a negation ("no", "neither", "none", "nor", or "leaves out:") and the field's
# word in the same clause (no ".", ";" or ":" between them), or one of the field's fixed wordings. A bare field word ("names commit
# 040056a", "names its dirty flag") states nothing.
UNCOMMITTED = (
    "uncommitted"  # the record's own git_dirty is true: the doc must say the tree had uncommitted changes
)

# The negation covers the words after it up to the clause end, or up to a new claim inside a list: ", names …",
# "and records …", "and its …" ("has no date field, names commit …, and records MLX 0.31.2" states no MLX gap).
_NEW_CLAIM = (
    r"(?:,|\band|\bbut)(?: and)? (?:names?|records?|notes?|has|have|holds?|gives?|states?|lists?|says?|is|are|was"
    r"|were)\b|\band (?:its|it)\b"
)
_NEGATION = r"(?:(?i:\b(?:no|neither|none|nor)\b)(?!-)|\bleaves? out:)(?:(?!" + _NEW_CLAIM + r")[^.;:])*?"
_FIELD_WORDS: dict[str, str] = {
    "commit": r"\bcommit\b(?! [0-9a-f]{7})",
    "dirty": r"\bdirty\b|\buncommitted-changes flag|\bflag (?:for|saying whether)[^.;:]*uncommitted changes",
    "date": r"\b(?:date|timestamp)\b",
    "mlx": r"\bMLX\b|\bversions\b",
    "mflux": r"\bmflux\b|\bversions\b",
    "hardware": r"\bhardware\b",
}
_FIXED_WORDINGS: dict[str, str] = {
    "dirty": r"not (?:say )?whether each working tree was clean|whether the working tree matched [^.;]*"
    r"(?:not recorded|unknown)",
    # Said of the run, not denied ("no uncommitted changes") and not asked ("whether … had uncommitted changes").
    UNCOMMITTED: r"(?:^|[.;:] )(?:(?!\bwhether\b)(?:[^.;:]|[.:](?=\S)))*?(?<!\b[Nn]o )(?<!\b[Nn]ot )(?<!\b[Ww]ithout )"
    r"(?<!\bfor )uncommitted changes|working tree was not clean|`git_dirty: true`",
}


def unstated_gaps(text: str, fields: frozenset[str]) -> set[str]:
    """The fields in ``fields`` for which ``text`` gives no negated wording."""
    stated: set[str] = set()
    for field in fields:
        forms = [_NEGATION + f"(?:{_FIELD_WORDS[field]})"] if field in _FIELD_WORDS else []
        forms += [_FIXED_WORDINGS[field]] if field in _FIXED_WORDINGS else []
        if any(re.search(form, text) for form in forms):
            stated.add(field)
    return set(fields) - stated


_SUPERSCRIPTS = "¹²³⁴⁵⁶⁷⁸⁹"


def footnote_markers(text: str) -> list[str]:
    """The footnote markers (superscript digits) in ``text``, in order, each once."""
    return list(dict.fromkeys(ch for ch in text if ch in _SUPERSCRIPTS))


@dataclass(frozen=True, slots=True)
class Stamp:
    """A record's install-time ``mlx_teacache_version``: the commit its ``+g<sha>`` names, and whether its
    ``.d<date>`` suffix marks uncommitted changes at install time."""

    version: str
    sha: str | None
    uncommitted: bool


def install_stamp(record: dict[str, Any]) -> Stamp:
    version = next(
        (str(b["mlx_teacache_version"]) for b in _provenance_blocks(record) if "mlx_teacache_version" in b),
        "",
    )
    sha = re.search(r"\+g([0-9a-f]+)", version)
    return Stamp(version, sha.group(1) if sha else None, bool(re.search(r"\.d\d{8}$", version)))


_STAMP_CLAIM = re.compile(
    r"version string names (?:no commit|commit ([0-9a-f]{7,}))( and marks uncommitted changes at install time)?"
)


def stamp_claims(text: str) -> list[tuple[str | None, bool]]:
    """Each install-time version stamp ``text`` describes: the commit it names (None for "names no commit"), and
    whether it marks uncommitted changes at install time."""
    return [(m.group(1), m.group(2) is not None) for m in _STAMP_CLAIM.finditer(text)]


# A limit is stated when its "N.NN×" sits in one clause with a limit word, within a few words of it. The clause ends
# at ";" or at a "." followed by a space (the dots inside "1.35×" or "0.31.2" do not end it).
_LIMIT_WORD = r"\b(?:allows?|allowed|limits?|at most)\b"
_IN_CLAUSE = r"(?:[^.;]|\.(?! |$)){0,50}?"


def states_limit(text: str, limit: str) -> bool:
    """True when ``text`` prints ``limit`` (such as "1.35×") beside "allow", "limit" or "at most" in one clause."""
    number = r"(?<![\d.])" + re.escape(limit)
    return re.search(f"{_LIMIT_WORD}{_IN_CLAUSE}{number}|{number}{_IN_CLAUSE}{_LIMIT_WORD}", text) is not None
