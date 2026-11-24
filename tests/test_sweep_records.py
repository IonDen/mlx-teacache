"""The committed threshold-sweep records back the skip counts and SSIM figures the docs quote from them.

Three checks, each named for the bug it catches:

- every record under ``_artifacts/sweeps/`` exists and parses with a non-empty ``thresholds`` list
  (a record not committed, renamed or truncated);
- every skip count and SSIM figure listed in ``FIGURES`` is printed at its own place in its doc (the table row for
  its threshold, or the sentence a hand-typed anchor names) and equals its record's value at that threshold, rounded
  half-up to the published digits (a doc or docstring figure that no longer equals its record, even when the same
  number is printed elsewhere in the file);
- every doc with a ``FIGURES`` case for a record names that record's path (a figure left without its record).

Streaks and single-run speedups printed beside these figures are not checked. Pure: reads the committed JSON and
text files, imports nothing from MLX or mflux.
"""

import json
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

R1 = "_artifacts/sweeps/qwen-image_plain-q4_2026-09-06.json"
R3 = "_artifacts/sweeps/qwen-image-2512_plain-q4_2026-10-20.json"
R4 = "_artifacts/sweeps/z-image-base_2026-05-31.json"
R5 = "_artifacts/sweeps/flux1-krea-dev_2026-09-05.json"
RECORDS = (R1, R3, R4, R5)

README = "README.md"
CALIBRATION = "docs/calibration.md"
QWEN = "docs/variants/qwen-image.md"
QWEN_2512 = "docs/variants/qwen-image-2512.md"
Z_IMAGE = "docs/variants/z-image-base.md"
KREA = "docs/variants/flux1-krea-dev.md"
QWEN_CFG = "src/mlx_teacache/variants/qwen_image/config.py"
QWEN_2512_CFG = "src/mlx_teacache/variants/qwen_image_2512/config.py"
Z_IMAGE_CFG = "src/mlx_teacache/variants/z_image_base/config.py"
KREA_CFG = "src/mlx_teacache/variants/flux1_krea_dev/config.py"

SSIM = "ssim_vs_vanilla"
SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class Row:
    """The cell under the one header starting with ``column``, in the one table row whose first cell is the threshold."""

    column: str


@dataclass(frozen=True, slots=True)
class Near:
    """A hand-typed regex for the figure's sentence: ``{f}`` marks the figure, ``{n}`` any other number. It must
    match exactly once in the doc, read with line breaks, indents and ``#`` comment marks collapsed to one space."""

    pattern: str


@dataclass(frozen=True, slots=True)
class Case:
    doc: str
    published: str
    record: str
    threshold: str
    key: str
    digits: int
    where: Row | Near


def _pairs(pairs: str) -> list[tuple[str, str]]:
    """``"threshold=published threshold=published ..."`` as (threshold, published) pairs."""
    return [(threshold, published) for threshold, published in (pair.split("=") for pair in pairs.split())]


def _rows(doc: str, record: str, key: str, digits: int, column: str, pairs: str) -> tuple[Case, ...]:
    """Table cases for one doc, record and column."""
    return tuple(
        Case(doc, published, record, threshold, key, digits, Row(column))
        for threshold, published in _pairs(pairs)
    )


def _at(doc: str, record: str, key: str, digits: int, threshold: str, published: str, pattern: str) -> Case:
    """One prose or docstring case, found by its hand-typed anchor."""
    return Case(doc, published, record, threshold, key, digits, Near(pattern))


def _slots(template: str, index: int) -> str:
    """``template`` with ``{f}`` in its ``index``-th ``{}`` slot and ``{n}`` in the others, for a list such as
    ``24 / 26 / 30 / 33 skips at 0.15 / 0.20 / 0.25 / 0.30`` where the figure's position names its threshold."""
    head, *tail = template.split("{}")
    return head + "".join(("{f}" if i == index else "{n}") + part for i, part in enumerate(tail))


def _listed(doc: str, record: str, key: str, digits: int, template: str, pairs: str) -> tuple[Case, ...]:
    """Cases for a list of figures in one sentence, each at its position in ``template``."""
    return tuple(
        _at(doc, record, key, digits, threshold, figure, _slots(template, i))
        for i, (threshold, figure) in enumerate(_pairs(pairs))
    )


def _triples(doc: str, record: str, digits: int, skips: str, ssims: str) -> tuple[Case, ...]:
    """Skip and SSIM cases for docstring lines of the form ``0.20 -> 26 / 0.980 / 2``."""
    cases: list[Case] = []
    for (threshold, skip), (ssim_threshold, ssim) in zip(_pairs(skips), _pairs(ssims), strict=True):
        assert threshold == ssim_threshold, f"{doc}: skip and SSIM lists out of step at {threshold}"
        line = re.escape(threshold) + r" -> {} / {} / {}"
        cases.append(_at(doc, record, SKIPPED, 0, threshold, skip, _slots(line, 0)))
        cases.append(_at(doc, record, SSIM, digits, threshold, ssim, _slots(line, 1)))
    return tuple(cases)


# The published figures, as each doc prints them. Only figures the text credits to the sweep are listed; bench skip
# counts quoted beside them come from the bench records, not from these.
R1_SSIM = "0.15=0.980 0.20=0.980 0.25=0.976 0.30=0.967"
R1_SKIPS = "0.15=24 0.20=26 0.25=30 0.30=33"
R3_SSIM = "0.15=0.991 0.20=0.978 0.25=0.969 0.30=0.947 0.40=0.928"
R3_SKIPS = "0.15=22 0.20=27 0.25=29 0.30=32 0.40=35"
R5_SSIM = "0.30=0.990 0.35=0.890 0.40=0.888 0.60=0.866 0.80=0.863"
R5_SKIPS = "0.30=10 0.35=12 0.40=13 0.60=16 0.80=18"

R3_SSIM_4DP = "0.15=0.9914 0.20=0.9777 0.25=0.9694 0.30=0.9470 0.40=0.9283"

# Hand-typed anchors for lists, one ``{}`` slot per figure in threshold order.
CALIBRATION_R1_SKIPS = r"gave {} / {} / {} / {} skips at 0\.15 / 0\.20 / 0\.25 / 0\.30,"
CALIBRATION_R1_SSIM = r"at 0\.15 / 0\.20 / 0\.25 / 0\.30, SSIM {} / {} / {} / {}, longest streaks"
QWEN_CFG_R1_FIRST = r"0\.15 -> {} of 48 active steps skipped, SSIM {}, longest streak {n};"
QWEN_CFG_R1_COMMENT = r"\({} at 0\.20, {} at 0\.25, {} at 0\.30\); {}/48 active steps skipped at 0\.30"
README_R5 = r"\(SSIM {} at 0\.30, {} at 0\.35; record:"
KREA_CFG_R5_REST = r"; {} at 0\.35, {} at 0\.40, {} at 0\.60, {} at 0\.80\. The knee is sharp"


FIGURES: tuple[Case, ...] = (
    # R1: Qwen-Image, stock q4, 2026-09-06.
    *_rows(QWEN, R1, SSIM, 3, "SSIM", R1_SSIM),
    *_rows(QWEN, R1, SKIPPED, 0, "Skipped", R1_SKIPS),
    _at(QWEN, R1, SSIM, 3, "0.30", "0.967", r"holding SSIM {f} against vanilla\."),
    _at(QWEN, R1, SKIPPED, 0, "0.20", "26", r"`rel_l1_thresh=0\.20`, which now gives {f} skips at {n}\."),
    _at(QWEN, R1, SSIM, 3, "0.20", "0.980", r"`rel_l1_thresh=0\.20`, which now gives {n} skips at {f}\."),
    *_listed(CALIBRATION, R1, SKIPPED, 0, CALIBRATION_R1_SKIPS, R1_SKIPS),
    *_listed(CALIBRATION, R1, SSIM, 3, CALIBRATION_R1_SSIM, R1_SSIM),
    _at(README, R1, SSIM, 3, "0.30", "0.967", r"SSIM {f} vs vanilla — the largest gain of any variant"),
    _at(README, R1, SKIPPED, 0, "0.30", "33", r"it skips {f} of 48 active steps for \*\*2\.68× combined\*\*"),
    _at(
        README,
        R1,
        SKIPPED,
        0,
        "0.20",
        "26",
        r"Lowering the threshold to 0\.20 gives back a higher-fidelity "
        r"operating point \({f} skips at SSIM {n} in the same record\)",
    ),
    _at(
        README,
        R1,
        SSIM,
        3,
        "0.20",
        "0.980",
        r"Lowering the threshold to 0\.20 gives back a higher-fidelity "
        r"operating point \({n} skips at SSIM {f} in the same record\)",
    ),
    _at(QWEN_CFG, R1, SKIPPED, 0, "0.15", "24", _slots(QWEN_CFG_R1_FIRST, 0)),
    _at(QWEN_CFG, R1, SSIM, 3, "0.15", "0.980", _slots(QWEN_CFG_R1_FIRST, 1)),
    *_triples(QWEN_CFG, R1, 3, "0.20=26 0.25=30 0.30=33", "0.20=0.980 0.25=0.976 0.30=0.967"),
    _at(QWEN_CFG, R1, SSIM, 3, "0.20", "0.980", _slots(QWEN_CFG_R1_COMMENT, 0)),
    _at(QWEN_CFG, R1, SSIM, 3, "0.25", "0.976", _slots(QWEN_CFG_R1_COMMENT, 1)),
    _at(QWEN_CFG, R1, SSIM, 3, "0.30", "0.967", _slots(QWEN_CFG_R1_COMMENT, 2)),
    _at(QWEN_CFG, R1, SKIPPED, 0, "0.30", "33", _slots(QWEN_CFG_R1_COMMENT, 3)),
    # R3: Qwen-Image-2512, stock q4, 2026-10-20.
    *_rows(QWEN_2512, R3, SSIM, 3, "SSIM", R3_SSIM),
    *_rows(QWEN_2512, R3, SKIPPED, 0, "Skipped", R3_SKIPS),
    _at(QWEN_2512, R3, SKIPPED, 0, "0.25", "29", r"\(the sweep counted {f} at this recipe\)"),
    _at(
        README,
        R3,
        SKIPPED,
        0,
        "0.25",
        "29",
        r"`rel_l1_thresh=0\.25`, set from its threshold sweep on the red-apple "
        r"prompt at that recipe: {f} of 48 active steps skipped at SSIM {n} against plain mflux",
    ),
    _at(
        README,
        R3,
        SSIM,
        3,
        "0.25",
        "0.969",
        r"`rel_l1_thresh=0\.25`, set from its threshold sweep on the red-apple "
        r"prompt at that recipe: {n} of 48 active steps skipped at SSIM {f} against plain mflux",
    ),
    _at(README, R3, SSIM, 3, "0.30", "0.947", r"The original checkpoint's 0\.30 falls to SSIM {f} on 2512\."),
    _at(CALIBRATION, R3, SKIPPED, 0, "0.25", "29", r"skipped {f} of 48 active steps at the 0\.25 default"),
    *_triples(QWEN_2512_CFG, R3, 4, R3_SKIPS, R3_SSIM_4DP),
    _at(QWEN_2512_CFG, R3, SSIM, 4, "0.30", "0.9470", r"falls below the floor on this checkpoint \({f}\)"),
    _at(
        QWEN_2512_CFG,
        R3,
        SKIPPED,
        0,
        "0.25",
        "29",
        r"Sweep table in the module docstring: {f} of 48 active steps skipped, SSIM {n}, longest skip run",
    ),
    _at(
        QWEN_2512_CFG,
        R3,
        SSIM,
        4,
        "0.25",
        "0.9694",
        r"Sweep table in the module docstring: {n} of 48 active steps skipped, SSIM {f}, longest skip run",
    ),
    # R4: Z-Image base, q8, 2026-05-31.
    _at(Z_IMAGE, R4, SKIPPED, 0, "0.12", "15", r"through 0\.12 \({f}/48 skips, SSIM {n}\)"),
    _at(Z_IMAGE, R4, SSIM, 3, "0.12", "0.991", r"through 0\.12 \({n}/48 skips, SSIM {f}\)"),
    _at(Z_IMAGE, R4, SSIM, 3, "0.15", "0.974", r"then drops to ~{f} at 0\.15"),
    _at(
        Z_IMAGE,
        R4,
        SSIM,
        3,
        "0.12",
        "0.991",
        r"SSIM {f} vs vanilla on this prompt; that SSIM comes from the threshold",
    ),
    _at(
        README,
        R4,
        SSIM,
        3,
        "0.12",
        "0.991",
        r"SSIM {f} vs vanilla, from the threshold sweep recorded in `_artifacts/sweeps/z-image-base_2026-05-31\.json`",
    ),
    _at(Z_IMAGE_CFG, R4, SKIPPED, 0, "0.12", "15", r"through 0\.12 \({f}/48 steps skipped, SSIM {n}\)"),
    _at(Z_IMAGE_CFG, R4, SSIM, 4, "0.12", "0.9913", r"through 0\.12 \({n}/48 steps skipped, SSIM {f}\)"),
    _at(Z_IMAGE_CFG, R4, SSIM, 3, "0.15", "0.974", r"then cliffs to ~{f} at 0\.15"),
    # R5: FLUX.1 Krea [dev], q4, 2026-09-05.
    *_rows(KREA, R5, SSIM, 3, "SSIM", R5_SSIM),
    *_rows(KREA, R5, SKIPPED, 0, "Skipped", R5_SKIPS),
    _at(KREA, R5, SSIM, 2, "0.30", "0.99", r"0\.30 holds SSIM {f} while skipping {n} of the 26 active steps"),
    _at(
        KREA, R5, SKIPPED, 0, "0.30", "10", r"0\.30 holds SSIM {n} while skipping {f} of the 26 active steps"
    ),
    _at(KREA, R5, SKIPPED, 0, "0.80", "18", r"at 0\.80, where {f} skipped steps out of 28 give at most"),
    *_listed(README, R5, SSIM, 3, README_R5, "0.30=0.990 0.35=0.890"),
    _at(
        KREA_CFG,
        R5,
        SSIM,
        3,
        "0.30",
        "0.990",
        r"SSIM vs vanilla {f} at 0\.30 with {n} of 26 active steps skipped",
    ),
    _at(
        KREA_CFG,
        R5,
        SKIPPED,
        0,
        "0.30",
        "10",
        r"SSIM vs vanilla {n} at 0\.30 with {f} of 26 active steps skipped",
    ),
    *_listed(KREA_CFG, R5, SSIM, 3, KREA_CFG_R5_REST, "0.35=0.890 0.40=0.888 0.60=0.866 0.80=0.863"),
)

# Docs that name a record without quoting its figures: the calibration page's table of what each record leaves out.
NAMES_WITHOUT_FIGURES: tuple[tuple[str, str], ...] = ((CALIBRATION, R4), (CALIBRATION, R5))

# Each (doc, record) pair whose doc must name the record's path: every pair FIGURES quotes, plus the list above.
DOC_NAMES_RECORD: tuple[tuple[str, str], ...] = tuple(
    sorted({(case.doc, case.record) for case in FIGURES} | set(NAMES_WITHOUT_FIGURES))
)


def load_record(record_path: str) -> dict[str, Any]:
    """Parse a record with every float kept as its exact decimal text (no binary rounding before half-up)."""
    text = (_REPO_ROOT / record_path).read_text(encoding="utf-8")
    parsed: dict[str, Any] = json.loads(text, parse_float=Decimal)
    return parsed


def record_value(record: dict[str, Any], threshold: str, key: str) -> Decimal:
    """The value of ``key`` in the one row whose ``threshold`` equals ``threshold``."""
    rows = [row for row in record["thresholds"] if Decimal(str(row["threshold"])) == Decimal(threshold)]
    assert len(rows) == 1, f"expected one row at threshold {threshold}, found {len(rows)}"
    return Decimal(str(rows[0][key]))


def half_up(value: Decimal, digits: int) -> str:
    """``value`` rounded half-up to ``digits`` decimals, as the docs print it."""
    return str(value.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


# ``{f}`` captures the whole number (never 0.97 out of 0.976); ``{n}`` stands for any other number.
_FIGURE = r"(?<![\d.])(\d+(?:\.\d+)?)(?!\d)"
_NUMBER = r"\d+(?:\.\d+)?"


def _table_cells(text: str, threshold: str, column: str) -> list[str]:
    """Every cell under a header starting with ``column`` in a table row whose first cell is ``threshold``."""
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
        if row[0] == threshold and len(columns) == 1 and columns[0] < len(row):
            cells.append(row[columns[0]])
    return cells


def _flatten(text: str) -> str:
    """``text`` as one line: line breaks, indents and leading ``#`` comment marks become one space."""
    lines = (re.sub(r"^\s*#*\s*", "", line) for line in text.splitlines())
    return re.sub(r"\s+", " ", " ".join(lines))


def figures_at(text: str, case: Case) -> list[str]:
    """Every figure ``text`` prints at the case's place: one item when the place is unambiguous."""
    if isinstance(case.where, Row):
        return _table_cells(text, case.threshold, case.where.column)
    pattern = case.where.pattern.replace("{f}", _FIGURE).replace("{n}", _NUMBER)
    assert re.compile(pattern).groups == 1, (
        f"{case.doc}: anchor {case.where.pattern!r} needs exactly one {{f}}"
    )
    return re.findall(pattern, _flatten(text))


def doc_prints(text: str, case: Case) -> bool:
    """True when ``text`` prints ``case.published`` at the case's place, and that place occurs exactly once."""
    return figures_at(text, case) == [case.published]


def _case(
    doc: str, record: str, threshold: str, key: str, kind: type[Row] | type[Near], marker: str = ""
) -> Case:
    """The one FIGURES case for this doc, record, threshold and key whose locator is of ``kind`` and mentions
    ``marker``."""
    found = [
        case
        for case in FIGURES
        if (case.doc, case.record, case.threshold, case.key) == (doc, record, threshold, key)
        and isinstance(case.where, kind)
        and marker in str(case.where)
    ]
    assert len(found) == 1, (
        f"expected one {kind.__name__} case for {doc} {threshold} {key}, found {len(found)}"
    )
    return found[0]


QWEN_TABLE = """\
| `rel_l1_thresh` | Skipped (of 48 active) | Longest streak | SSIM vs vanilla | Single-rep speedup |
|---|---|---|---|---|
| 0.15 | 24 | 2 | 0.980 | 1.71× |
| 0.20 | 26 | 2 | {ssim_020} | 1.93× |
"""


def test_a_wrong_table_figure_is_caught_when_another_row_prints_the_right_one() -> None:
    """Bug caught: a matcher that looks for the figure anywhere in the doc, so row 0.20's SSIM can drift from 0.980
    to 0.975 while row 0.15 still prints 0.980 (review survivor M1)."""
    case = _case(QWEN, R1, "0.20", SSIM, Row)
    assert doc_prints(QWEN_TABLE.format(ssim_020="0.980"), case)
    assert not doc_prints(QWEN_TABLE.format(ssim_020="0.975"), case)


CALIBRATION_SENTENCE = (
    "A threshold sweep under the current gate (stock q4, same recipe)\n"
    "gave 24 / 26 / {skips} skips at 0.15 / 0.20 / 0.25 / 0.30, SSIM 0.980 / 0.980 /\n"
    "0.976 / 0.967, longest streaks 2 / 2 / 3 / 4, and 0.30 stays the default."
)


def test_a_figure_moved_to_another_threshold_in_a_list_is_caught() -> None:
    """Bug caught: a matcher blind to position, so swapping the 0.25 and 0.30 skip counts in a slash list (30 / 33
    printed as 33 / 30) passes because both numbers are still in the text (review survivor M10)."""
    case = _case(CALIBRATION, R1, "0.25", SKIPPED, Near)
    assert doc_prints(CALIBRATION_SENTENCE.format(skips="30 / 33"), case)
    assert not doc_prints(CALIBRATION_SENTENCE.format(skips="33 / 30"), case)


QWEN_DOCSTRING = (
    "stock q4 under the v0.10.0 consecutive-delta gate: 0.15 -> 24 of 48 active steps\n"
    "skipped, SSIM 0.980, longest streak 2; 0.20 -> 26 / {ssim_020} / 2; 0.25 -> 30 / 0.976 / 3;\n"
    "0.30 -> 33 / 0.967 / 4."
)


def test_a_wrong_docstring_figure_is_caught_when_the_line_before_prints_the_right_one() -> None:
    """Bug caught: a matcher that finds 0.980 in the 0.15 clause and so passes a docstring whose 0.20 clause reads
    ``26 / 0.975 / 2`` (review survivor M14)."""
    case = _case(QWEN_CFG, R1, "0.20", SSIM, Near, marker="->")
    assert doc_prints(QWEN_DOCSTRING.format(ssim_020="0.980"), case)
    assert not doc_prints(QWEN_DOCSTRING.format(ssim_020="0.975"), case)


def test_a_figure_whose_place_appears_twice_does_not_count_as_printed() -> None:
    """Bug caught: a locator that takes the first row or sentence it finds, so a second 0.20 row or a second copy
    of the docstring line with 0.975 passes because the first one prints 0.980."""
    row = _case(QWEN, R1, "0.20", SSIM, Row)
    assert not doc_prints(
        QWEN_TABLE.format(ssim_020="0.980") + "\n" + QWEN_TABLE.format(ssim_020="0.975"), row
    )
    near = _case(QWEN_CFG, R1, "0.20", SSIM, Near, marker="->")
    assert not doc_prints(
        QWEN_DOCSTRING.format(ssim_020="0.980") + "\n" + QWEN_DOCSTRING.format(ssim_020="0.975"), near
    )


def test_half_up_rounds_a_tie_up_unlike_round() -> None:
    """Bug caught: half-to-even rounding (what round() does), which turns 0.9465 into 0.946 where the docs'
    half-up rounding prints 0.947."""
    assert half_up(Decimal("0.9465"), 3) == "0.947"
    assert half_up(Decimal("0.94649"), 3) == "0.946"
    assert half_up(Decimal("0.946995"), 4) == "0.9470"
    assert half_up(Decimal("24"), 0) == "24"


@pytest.mark.parametrize("record_path", RECORDS)
def test_sweep_record_is_committed_and_parses(record_path: str) -> None:
    """Bug caught: a record the docs cite is not committed, is renamed, or has lost its threshold rows."""
    path = _REPO_ROOT / record_path
    assert path.is_file(), f"{record_path} is missing"
    thresholds = load_record(record_path).get("thresholds")
    assert isinstance(thresholds, list) and thresholds, f"{record_path} has no thresholds list"


@pytest.mark.parametrize(
    "case",
    FIGURES,
    ids=[f"{c.doc}:{c.published}@{c.threshold}:{c.key}:{type(c.where).__name__}" for c in FIGURES],
)
def test_published_figure_equals_its_record(case: Case) -> None:
    """Bug caught: a doc or docstring figure that no longer equals the record it comes from, or that the doc no
    longer prints at its threshold."""
    text = (_REPO_ROOT / case.doc).read_text(encoding="utf-8")
    assert doc_prints(text, case), (
        f"{case.doc} at {case.threshold} ({case.where}) prints {figures_at(text, case)}, "
        f"expected [{case.published!r}]"
    )
    value = record_value(load_record(case.record), case.threshold, case.key)
    assert half_up(value, case.digits) == case.published, (
        f"{case.record} {case.key} at {case.threshold} is {value}, which rounds to {half_up(value, case.digits)}, "
        f"but {case.doc} prints {case.published}"
    )


@pytest.mark.parametrize(
    ("doc_path", "record_path"),
    DOC_NAMES_RECORD,
    ids=[f"{doc}->{Path(rec).name}" for doc, rec in DOC_NAMES_RECORD],
)
def test_doc_names_the_record_it_quotes(doc_path: str, record_path: str) -> None:
    """Bug caught: a doc quotes a sweep figure but does not say which committed record it comes from."""
    text = (_REPO_ROOT / doc_path).read_text(encoding="utf-8")
    assert record_path in text, f"{doc_path} quotes figures from {record_path} but does not name it"
