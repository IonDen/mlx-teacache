"""The committed bench records back the speedups, timings, skip counts and memory figures the public docs quote.

Each check is named for the bug it catches:

- every figure in ``FIGURES`` is printed at its own place in its doc (a table cell, or the sentence a hand-typed
  anchor names) and equals the value computed from its record's arrays, rounded half-up to the printed digits (a doc
  figure that drifted from its record, even when the same number is printed elsewhere in the file);
- every doc with a ``FIGURES`` entry for a record names that record's ``_artifacts/`` path (a figure left without its
  record);
- for each (doc, record) pair, one paragraph or footnote names the record and gives a negated wording for each gap
  pinned in ``GAPS`` (a deleted gap sentence, or one reversed into a completeness claim). This is per pair, not per
  figure: a figure whose own paragraph drops the gap passes while another paragraph of the doc states it. Where one
  paragraph names several records with gaps by their ``_artifacts/`` paths, each record's gaps count only in its own
  span, from its ``GAP_SPAN`` anchor to the next clause about another record (one record's "no MLX version" standing
  in for another's). Two limits remain: the pairs in ``PARAGRAPH_LEVEL_GAPS`` (six pairs, eight fields) still read
  those fields in the whole paragraph, and a paragraph that names one record by path and describes another only in
  words is read whole for the named record, so the described record's gap wording still counts for it;
- ``GAPS`` itself matches each record's JSON keys (a record that gains or loses a field the docs describe);
- each README table row's footnote marker leads to a footnote whose citation names that row's record (a wrong marker,
  or a footnote citing another record);
- each install-time version stamp a doc quotes matches the stamp of a record that doc quotes;
- every combined median speedup, and every base/base-teacache ratio on the distilled page, that is above the limit
  its skipped steps allow, N / (N - median skipped), is quoted with that limit as "N.NN×" beside "allow", "limit" or
  "at most" in the same paragraph, table row or marked footnote, or sits beside its own record's gating and compile
  figures (a doc that drops the limit beside an above-the-limit figure). Gating ratios and single-run speedups are not
  checked against a limit;
- every "N.NN×" a doc prints outside generated blocks is a FIGURES place, or one of the counted restatements in
  ``UNPINNED_TIMES`` (a new speedup claim, even one that reuses a value printed elsewhere in the doc).

Pure: reads the committed JSON and text files, imports nothing from MLX or mflux.
"""

import re
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from tests._doc_figures import (
    REPO_ROOT,
    UNCOMMITTED,
    Near,
    Place,
    Row,
    clauses,
    compile_place,
    dirty_at_run_time,
    figures_at,
    flatten,
    footnote_markers,
    half_up,
    install_stamp,
    load_record,
    missing_provenance,
    span_from,
    stamp_claims,
    states_limit,
    table_cells,
    unstated_gaps,
)

# Docs are read from DOC_ROOT and records from `_doc_figures.REPO_ROOT`; both are the repo root unless a mutation
# harness repoints them (setting `_doc_figures.REPO_ROOT` before this module is imported moves both).
DOC_ROOT = REPO_ROOT

Q11 = "_artifacts/v0.11.0_bench_qwen_image.json"
Q10 = "_artifacts/v0.10.0_bench_qwen_image.json"
K11 = "_artifacts/v0.11.0_bench_krea_dev.json"
B4 = "_artifacts/v0.12.0_bench_klein_base_4b.json"
B9 = "_artifacts/v0.12.0_bench_klein_base_9b.json"
B4_10 = "_artifacts/v0.10.0_bench_klein_base_4b.json"
B9_10 = "_artifacts/v0.10.0_bench_klein_base_9b.json"
B4_6 = "_artifacts/v0.6.0_bench_klein_base_4b.json"
B9_6 = "_artifacts/v0.6.0_bench_klein_base_9b.json"
B4_41 = "_artifacts/_bench_report_klein_base_4b_v0_4_1.json"
Z10 = "_artifacts/v0.10.0_bench_z_image.json"
Z7 = "_artifacts/_bench_z_image_v0_7_0.json"
F18 = "_artifacts/v0.18.0_bench_flux1_dev.json"
F10 = "_artifacts/v0.10.0_bench_flux1_dev.json"
F63 = "_artifacts/v0.6.3_bench_flux1_dev.json"
VAL = "_artifacts/validation_klein_base_9b.json"
D4 = "_artifacts/klein_base_vs_distilled_4b_q4.json"
D9 = "_artifacts/klein_base_vs_distilled_9b_q4_512x512.json"
RJ = "_artifacts/comparison/report.json"
SW_Q = "_artifacts/sweeps/qwen-image_plain-q4_2026-09-06.json"
SW_Z = "_artifacts/sweeps/z-image-base_2026-05-31.json"
SW_K = "_artifacts/sweeps/flux1-krea-dev_2026-09-05.json"

README = "README.md"
COMPARISON = "COMPARISON.md"
CMP_DEV = "docs/comparison/flux1-dev.md"
CMP_KREA = "docs/comparison/flux1-krea-dev.md"
CMP_B4 = "docs/comparison/klein-base-4b.md"
CMP_B9 = "docs/comparison/klein-base-9b.md"
CMP_QWEN = "docs/comparison/qwen-image.md"
CMP_Z = "docs/comparison/z-image-base.md"
CMP_DISTILLED = "docs/comparison/klein-distilled-vs-base.md"
V_DEV = "docs/variants/flux1-dev.md"
V_KREA = "docs/variants/flux1-krea-dev.md"
V_K4 = "docs/variants/flux2-klein-4b.md"
V_K9 = "docs/variants/flux2-klein-9b.md"
V_B4 = "docs/variants/flux2-klein-base-4b.md"
V_B9 = "docs/variants/flux2-klein-base-9b.md"
V_QWEN = "docs/variants/qwen-image.md"
V_Z = "docs/variants/z-image-base.md"
M3 = "docs/m3-plus-tradeoff.md"
CALIBRATION = "docs/calibration.md"

GIB = Decimal(1024**3)


@dataclass(frozen=True, slots=True)
class Value:
    """A figure computed from a record: medians, ratios, percentages and counts, never typed in."""

    name: str
    of: Callable[[dict[str, Any]], Decimal]
    # The most the value's skipped steps alone allow, N / (N - skipped), for a speedup; None for other figures.
    limit: Callable[[dict[str, Any]], Decimal] | None = None


def _d(x: Any) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(x)


def _med(record: dict[str, Any], key: str) -> Decimal:
    return _d(statistics.median(record[key]))


def _ratio(top: str, bottom: str) -> Callable[[dict[str, Any]], Decimal]:
    return lambda r: _med(r, top) / _med(r, bottom)


def _ceiling(record: dict[str, Any]) -> Decimal:
    """The most the skipped steps alone can give: N / (N - median skipped)."""
    steps = _d(record["num_inference_steps"])
    return steps / (steps - _med(record, "skipped_counts"))


SPEEDUP = Value("combined", _ratio("vanilla_seconds", "wrapper_seconds"), _ceiling)
GATING = Value("gating", _ratio("nogate_seconds", "wrapper_seconds"))
COMPILE = Value("compile", _ratio("vanilla_seconds", "nogate_seconds"))
CEILING = Value("ceiling", _ceiling)
SKIPPED = Value("skipped", lambda r: _med(r, "skipped_counts"))
GATE_COMPUTED = Value("computed", lambda r: _med(r, "computed_counts"))
COMPUTED = Value("steps computed", lambda r: _d(r["num_inference_steps"]) - _med(r, "skipped_counts"))
LONGEST_RUN = Value("longest run", lambda r: _med(r, "max_consecutive_skips"))
VANILLA = Value("vanilla s", lambda r: _med(r, "vanilla_seconds"))
NOGATE = Value("no-gate s", lambda r: _med(r, "nogate_seconds"))
WRAPPER = Value("gated s", lambda r: _med(r, "wrapper_seconds"))
VANILLA_PEAK = Value("vanilla GiB", lambda r: _med(r, "vanilla_peak_memory_gb"))
NOGATE_PEAK = Value("no-gate GiB", lambda r: _med(r, "nogate_peak_memory_gb"))
WRAPPER_PEAK = Value("gated GiB", lambda r: _med(r, "wrapper_peak_memory_gb"))
PEAK_DROP_PCT = Value(
    "peak drop %", lambda r: (1 - _med(r, "wrapper_peak_memory_gb") / _med(r, "vanilla_peak_memory_gb")) * 100
)
NOGATE_GAP = Value("no-gate gap s", lambda r: _med(r, "vanilla_seconds") - _med(r, "nogate_seconds"))
VANILLA_SPREAD = Value(
    "vanilla spread s", lambda r: _d(max(r["vanilla_seconds"])) - _d(min(r["vanilla_seconds"]))
)
FASTEST_RATIO = Value(
    "fastest/fastest", lambda r: _d(min(r["vanilla_seconds"])) / _d(min(r["wrapper_seconds"]))
)
VANILLA_MIN = Value("fastest vanilla s", lambda r: _d(min(r["vanilla_seconds"])))
VANILLA_MAX = Value("slowest vanilla s", lambda r: _d(max(r["vanilla_seconds"])))
NOGATE_MIN = Value("fastest no-gate s", lambda r: _d(min(r["nogate_seconds"])))
NOGATE_MAX = Value("slowest no-gate s", lambda r: _d(max(r["nogate_seconds"])))
# The step total a skip count is quoted out of ("33 of 50"): the record's own num_inference_steps.
STEPS_TOTAL = Value("steps", lambda r: _d(r["num_inference_steps"]))

_PEAKS = ("vanilla_peak_memory_gb", "nogate_peak_memory_gb", "wrapper_peak_memory_gb")
_CACHES = ("vanilla_cache_memory_gb", "nogate_cache_memory_gb", "wrapper_cache_memory_gb")
_TIMINGS = ("vanilla_seconds", "nogate_seconds", "wrapper_seconds")


def _every(record: dict[str, Any], keys: Sequence[str]) -> list[Decimal]:
    return [_d(x) for key in keys for x in record[key]]


def _peak_plus_cache(record: dict[str, Any]) -> list[Decimal]:
    """Each rep's active peak plus the cache reading of the same rep, in every condition."""
    return [
        _d(peak) + _d(cache)
        for peaks, caches in zip(_PEAKS, _CACHES, strict=True)
        for peak, cache in zip(record[peaks], record[caches], strict=True)
    ]


PEAK_LOW = Value("lowest peak GiB", lambda r: min(_every(r, _PEAKS)))
PEAK_HIGH = Value("highest peak GiB", lambda r: max(_every(r, _PEAKS)))
CACHE = Value("cache GiB", lambda r: _d(statistics.median(_every(r, _CACHES))))
PEAK_CACHE_LOW = Value("lowest peak + cache GiB", lambda r: min(_peak_plus_cache(r)))
PEAK_CACHE_HIGH = Value("highest peak + cache GiB", lambda r: max(_peak_plus_cache(r)))
TOTAL_MINUTES = Value("all timings min", lambda r: sum(_every(r, _TIMINGS), Decimal(0)) / 60)


def rep(key: str, index: int) -> Value:
    """Rep ``index`` of ``key``, in record order."""
    return Value(f"{key}[{index}]", lambda r: _d(r[key][index]))


def ranked(key: str, index: int) -> Value:
    """The ``index``-th fastest rep of ``key``."""
    return Value(f"sorted {key}[{index}]", lambda r: _d(sorted(r[key])[index]))


def against(other: str, of: Callable[[dict[str, Any], dict[str, Any]], Decimal], name: str) -> Value:
    """A figure comparing this record with ``other`` (the README's v0.10.0 to v0.12.0 drift sentence)."""
    return Value(name, lambda r: of(r, load_record(other)))


def plain(key: str) -> Value:
    """A single-run record's own field (the validation record holds one run per condition)."""
    return Value(key, lambda r: _d(r[key]))


VAL_SPEEDUP = Value(
    "single-run speedup",
    lambda r: _d(r["vanilla_seconds"]) / _d(r["wrapper_seconds"]),
)


def _warm(record: dict[str, Any], condition: str) -> Decimal:
    """The distilled-vs-base page's headline: the median of the reps after the cold first one."""
    return _d(statistics.median(record["conditions"][condition]["rep_seconds"][1:]))


def _skip_limit(condition: str) -> Callable[[dict[str, Any]], Decimal]:
    """N / (N - median skipped) for one condition of the distilled-vs-base records."""

    def of(r: dict[str, Any]) -> Decimal:
        c = r["conditions"][condition]
        steps = _d(c["num_inference_steps"])
        return steps / (steps - _d(c["skipped_median"]))

    return of


def warm(condition: str) -> Value:
    return Value(f"{condition} warm s", lambda r: _warm(r, condition))


def warm_ratio(top: str, bottom: str) -> Value:
    # Only the TeaCache condition skips steps, so only a ratio over it has a skip limit.
    limit = _skip_limit(bottom) if bottom == "base-teacache" else None
    return Value(f"{top}/{bottom} warm", lambda r: _warm(r, top) / _warm(r, bottom), limit)


def median_s(condition: str) -> Value:
    return Value(
        f"{condition} median s", lambda r: _d(statistics.median(r["conditions"][condition]["rep_seconds"]))
    )


def median_ratio(top: str, bottom: str) -> Value:
    def of(r: dict[str, Any]) -> Decimal:
        c = r["conditions"]
        return _d(statistics.median(c[top]["rep_seconds"])) / _d(statistics.median(c[bottom]["rep_seconds"]))

    return Value(f"{top}/{bottom} median", of)


def trim_pct() -> Value:
    def of(r: dict[str, Any]) -> Decimal:
        c = r["conditions"]
        base = _d(statistics.median(c["base"]["rep_seconds"]))
        return (1 - _d(statistics.median(c["base-teacache"]["rep_seconds"])) / base) * 100

    return Value("base trim %", of)


def condition(name: str, key: str) -> Value:
    return Value(f"{name}.{key}", lambda r: _d(r["conditions"][name][key]))


def rj(slug: str, *path: str) -> Value:
    """A field of one model's single run in the comparison report."""

    def of(r: dict[str, Any]) -> Decimal:
        node: Any = r["variants"][slug]
        for key in path:
            node = node[key]
        return _d(node)

    return Value(f"{slug}.{'.'.join(path)}", of)


def rj_gib(slug: str, cond: str, key: str) -> Value:
    return Value(f"{slug}.{cond}.{key} GiB", lambda r: _d(r["variants"][slug][cond][key]) / GIB)


def rj_speedup(slug: str) -> Value:
    def of(r: dict[str, Any]) -> Decimal:
        v = r["variants"][slug]
        return _d(v["a"]["generation_seconds"]) / _d(v["b"]["generation_seconds"])

    return Value(f"{slug} wall speedup", of)


def _rj_computed(r: dict[str, Any], slug: str) -> Decimal:
    v = r["variants"][slug]
    return _d(v["steps"]) - _d(v["b"]["skipped"])


def rj_computed(slug: str) -> Value:
    return Value(f"{slug} steps computed", lambda r: _rj_computed(r, slug))


def rj_ceiling(slug: str) -> Value:
    """The most the single run's skipped steps allow: steps / (steps - skipped)."""
    return Value(f"{slug} ceiling", lambda r: _d(r["variants"][slug]["steps"]) / _rj_computed(r, slug))


def _sweep_steps(r: dict[str, Any]) -> Decimal:
    return _d(r["num_inference_steps"] if "num_inference_steps" in r else r["recipe"]["num_inference_steps"])


def _sweep_row(r: dict[str, Any], threshold: str) -> dict[str, Any]:
    rows = [row for row in r["thresholds"] if _d(row["threshold"]) == Decimal(threshold)]
    assert len(rows) == 1, f"expected one sweep row at {threshold}, found {len(rows)}"
    return rows[0]


def sweep_at(threshold: str, key: str) -> Value:
    """A sweep record's ``key`` at one threshold."""
    return Value(f"sweep {threshold} {key}", lambda r: _d(_sweep_row(r, threshold)[key]))


def _sweep_computed(r: dict[str, Any], threshold: str) -> Decimal:
    return _sweep_steps(r) - _d(_sweep_row(r, threshold)["skipped"])


def sweep_computed(threshold: str) -> Value:
    return Value(f"sweep {threshold} steps computed", lambda r: _sweep_computed(r, threshold))


def sweep_ceiling(threshold: str) -> Value:
    """The most one threshold's skipped steps allow: steps / (steps - skipped)."""
    return Value(f"sweep {threshold} ceiling", lambda r: _sweep_steps(r) / _sweep_computed(r, threshold))


def sweep_plain_share(threshold: str) -> Value:
    """The plain run's time, as printed to 0.1 s, scaled to the steps one threshold computes (the calibration note
    works from the printed 128.7 s)."""

    def of(r: dict[str, Any]) -> Decimal:
        return Decimal(half_up(_d(r["vanilla_seconds"]), 1)) * _sweep_computed(r, threshold) / _sweep_steps(r)

    return Value(f"sweep {threshold} plain share s", of)


def sweep_speedup(threshold: str) -> Value:
    """A sweep's single-rep speedup at one threshold: the plain run's time over that threshold's gated run."""
    return Value(
        f"sweep {threshold} speedup",
        lambda r: _d(r["vanilla_seconds"]) / _d(_sweep_row(r, threshold)["wrapper_seconds"]),
    )


def skip_limit(condition: str) -> Value:
    """N / (N - median skipped) for one condition of the distilled-vs-base records, as the page prints it."""
    return Value(f"{condition} skip limit", _skip_limit(condition))


SWEEP_PLAIN = Value("sweep plain s", lambda r: _d(r["vanilla_seconds"]))
SWEEP_STEPS = Value("sweep steps", _sweep_steps)


# Ceiling exemptions, written into each entry that needs one.
DECOMPOSED = (
    "decomposed"  # the figure's unit or marked footnote holds its own record's gating and compile figures
)


@dataclass(frozen=True, slots=True)
class Figure:
    doc: str
    record: str
    value: Value
    digits: int
    where: Place
    row: str = ""
    exempt: str = ""


Slot = tuple[str, Value, int] | tuple[str, Value, int, str]


def _t(doc: str, template: str, *slots: Slot) -> tuple[Figure, ...]:
    """Entries for the figures in one sentence, copied as plain text with ``{}`` where each figure sits."""
    head, *tail = template.split("{}")
    assert len(tail) == len(slots), f"{doc}: {template!r} has {len(tail)} slots for {len(slots)} figures"
    figures: list[Figure] = []
    for index, slot in enumerate(slots):
        record, value, digits, *rest = slot
        pattern = re.escape(head) + "".join(
            ("{f}" if i == index else "{n}") + re.escape(part) for i, part in enumerate(tail)
        )
        figures.append(Figure(doc, record, value, digits, Near(pattern), exempt=rest[0] if rest else ""))
    return tuple(figures)


def _cell(
    doc: str, row: str, column: str, pattern: str, record: str, value: Value, digits: int, exempt: str = ""
) -> Figure:
    return Figure(doc, record, value, digits, Row(column, pattern), row=row, exempt=exempt)


def _cells(doc: str, row: str, record: str, *cells: tuple[str, str, Value, int]) -> tuple[Figure, ...]:
    return tuple(
        _cell(doc, row, column, pattern, record, value, digits) for column, pattern, value, digits in cells
    )


WRAPPER_REPS = rep("wrapper_seconds", 0), rep("wrapper_seconds", 1), rep("wrapper_seconds", 2)
VANILLA_REPS = rep("vanilla_seconds", 0), rep("vanilla_seconds", 1), rep("vanilla_seconds", 2)
NOGATE_REPS = rep("nogate_seconds", 0), rep("nogate_seconds", 1), rep("nogate_seconds", 2)


def _reps(record: str, values: Sequence[Value], digits: int = 1) -> tuple[Slot, ...]:
    return tuple((record, v, digits) for v in values)


def _krea_comparison(doc: str) -> tuple[Figure, ...]:
    """The Krea paragraph COMPARISON.md and its comparison page share word for word."""
    return _t(
        doc,
        "this variant's multi-run speedup as {}×. The {}× and the gated run's {}× over the no-gate run are both above "
        "its {}× limit (footnote ⁷), and the record does not say why. The no-gate wrapper's {}× over plain mflux is "
        "within",
        (K11, SPEEDUP, 2),
        (K11, SPEEDUP, 2),
        (K11, GATING, 2),
        (K11, CEILING, 2),
        (K11, COMPILE, 2),
    )


HEADLINE = r"^{f}×"
STEPS = r"^{f} of {n}$"
STEPS_OF = r"^{n} of {f}$"

FIGURES: tuple[Figure, ...] = (
    # README headline table (:20-25): unpinned before this test.
    _cell(README, "Qwen-Image (original checkpoint)", "Speedup", HEADLINE, Q11, SPEEDUP, 2),
    _cell(README, "Qwen-Image (original checkpoint)", "Steps skipped", STEPS, Q11, SKIPPED, 0),
    _cell(README, "Qwen-Image (original checkpoint)", "Steps skipped", STEPS_OF, Q11, STEPS_TOTAL, 0),
    _cell(README, "FLUX.2 Klein base 9B", "Speedup", HEADLINE, B9, SPEEDUP, 2),
    _cell(README, "FLUX.2 Klein base 9B", "Steps skipped", STEPS, B9, SKIPPED, 0),
    _cell(README, "FLUX.2 Klein base 9B", "Steps skipped", STEPS_OF, B9, STEPS_TOTAL, 0),
    _cell(README, "FLUX.2 Klein base 4B", "Speedup", HEADLINE, B4, SPEEDUP, 2, DECOMPOSED),
    _cell(README, "FLUX.2 Klein base 4B", "Steps skipped", STEPS, B4, SKIPPED, 0),
    _cell(README, "FLUX.2 Klein base 4B", "Steps skipped", STEPS_OF, B4, STEPS_TOTAL, 0),
    _cell(README, "Z-Image base", "Speedup", HEADLINE, Z10, SPEEDUP, 2),
    _cell(README, "Z-Image base", "Steps skipped", STEPS, Z10, SKIPPED, 0),
    _cell(README, "Z-Image base", "Steps skipped", STEPS_OF, Z10, STEPS_TOTAL, 0),
    _cell(README, "FLUX.1 Krea [dev]", "Speedup", HEADLINE, K11, SPEEDUP, 2),
    _cell(README, "FLUX.1 Krea [dev]", "Steps skipped", STEPS, K11, SKIPPED, 0),
    _cell(README, "FLUX.1 Krea [dev]", "Steps skipped", STEPS_OF, K11, STEPS_TOTAL, 0),
    _cell(README, "FLUX.1 dev", "Speedup", HEADLINE, F18, SPEEDUP, 2),
    _cell(README, "FLUX.1 dev", "Steps skipped", STEPS, F18, SKIPPED, 0),
    _cell(README, "FLUX.1 dev", "Steps skipped", STEPS_OF, F18, STEPS_TOTAL, 0),
    # README prose under the table.
    *_t(
        README,
        "FLUX.1 dev's {}×⁶ comes entirely from the {} skipped steps",
        (F18, SPEEDUP, 2),
        (F18, SKIPPED, 0),
    ),
    *_t(
        README,
        "took the same time as plain mflux ({} s both). It is under its {}× limit (footnote ⁶ has the arithmetic). "
        "An older {}× from mflux 0.18.0 was above that limit",
        (F18, VANILLA, 2),
        (F18, CEILING, 2),
        (F10, SPEEDUP, 2),
    ),
    *_t(
        README,
        "Krea's {}×⁷ is above its {}× limit (footnote ⁷)",
        (K11, SPEEDUP, 2),
        (K11, CEILING, 2),
    ),
    *_t(README, "skips around 25% of steps ({} of 25, Benchmarks footnote ⁶)", (F18, SKIPPED, 0)),
    # Supported-models entries.
    *_t(
        README,
        "**{}× combined** ({} s → {} s, Benchmarks footnote ¹)",
        (Q11, SPEEDUP, 2),
        (Q11, VANILLA, 1),
        (Q11, WRAPPER, 1),
    ),
    *_t(
        README,
        "it skips {}/48 active steps for **{}× combined wall-clock** ({} s → {} s median; {}× gating + {}× "
        "compile-avoidance), at SSIM 0.993 against vanilla and a peak-memory drop from ~{} GiB to ~{} GiB (footnote ²)",
        (B9, SKIPPED, 0),
        (B9, SPEEDUP, 2, DECOMPOSED),
        (B9, VANILLA, 1),
        (B9, WRAPPER, 1),
        (B9, GATING, 2),
        (B9, COMPILE, 2),
        (B9, VANILLA_PEAK, 0),
        (B9, WRAPPER_PEAK, 1),
    ),
    *_t(
        README,
        "v0.5.0 advertised {}× on this variant (Klein base-9B;",
        (VAL, VAL_SPEEDUP, 2),
    ),
    *_t(
        README,
        "Plain mflux peaked at {} GiB there, above the {} GiB cap the run set",
        (VAL, plain("vanilla_peak_memory_gb"), 2),
        (VAL, plain("mlx_memory_cap_gb"), 0),
    ),
    *_t(README, "The wrapper peaked at {} GiB. That fits", (VAL, plain("wrapper_peak_memory_gb"), 2)),
    *_t(
        README,
        "the gate skips {}/48 active steps for a **{}× combined speedup**³ on M1 Max, almost all of it "
        "step-skipping, with compile-avoidance at noise level ({}×)",
        (B4, SKIPPED, 0),
        (B4, SPEEDUP, 2, DECOMPOSED),
        (B4, COMPILE, 2),
    ),
    *_t(
        README,
        "it skips {}/48 active steps for **{}× combined wall-clock**⁵ ({} s → {} s), entirely from gating — the "
        "no-gate wrapper runs at vanilla speed ({}×)",
        (Z10, SKIPPED, 0),
        (Z10, SPEEDUP, 2),
        (Z10, VANILLA, 1),
        (Z10, WRAPPER, 1),
        (Z10, COMPILE, 2),
    ),
    *_t(
        README, "Peak memory drops {} GiB → {} GiB (the eager", (Z10, VANILLA_PEAK, 1), (Z10, WRAPPER_PEAK, 1)
    ),
    *_t(
        README,
        "**{}× combined**⁷ ({} s → {} s), {} of 26 active steps skipped in every rep",
        (K11, SPEEDUP, 2),
        (K11, VANILLA, 1),
        (K11, WRAPPER, 1),
        (K11, SKIPPED, 0),
    ),
    *_t(
        README,
        "The gated run is about {}× faster than the no-gate run, and the no-gate wrapper's {}× over plain mflux is "
        "within the rep-to-rep range",
        (K11, GATING, 2),
        (K11, COMPILE, 2),
    ),
    *_t(
        README,
        "Both the {}× and the {}× are above its {}× limit (footnote ⁷)",
        (K11, SPEEDUP, 2),
        (K11, GATING, 2),
        (K11, CEILING, 2),
    ),
    *_t(
        README,
        "about {}× faster than Base at 50 steps on 4B ({} s against {} s; 768×1024, 3 reps) and about {}× on 9B "
        "({} s against {} s; 512×512, 2 reps), and the wrapper trims {} percent off Base on 4B and {} percent on 9B",
        (D4, median_ratio("base", "distilled"), 0),
        (D4, median_s("distilled"), 0),
        (D4, median_s("base"), 0),
        (D9, median_ratio("base", "distilled"), 0),
        (D9, median_s("distilled"), 0),
        (D9, median_s("base"), 0),
        (D4, trim_pct(), 0),
        (D9, trim_pct(), 0),
    ),
    # README Benchmarks table: the z-image row and the mechanism cells (the other cells are pinned in
    # test_bench_artifacts.py).
    *_cells(
        README,
        "`z-image-base` (CFG)⁵",
        Z10,
        ("Vanilla", r"^{f}s$", VANILLA, 1),
        ("Wrapper", r"^{f}s$", WRAPPER, 1),
        ("Speedup", r"^\*\*{f}×\*\*$", SPEEDUP, 2),
        ("Skipped", r"^\*\*{f} / {n}\*\*$", SKIPPED, 0),
    ),
    *_cells(
        README,
        "`qwen-image` (CFG)¹",
        Q11,
        ("Mechanism", r"\({f}× against its own no-gate run", GATING, 2),
        ("Mechanism", r"no-gate wrapper's {f}× is inside", COMPILE, 2),
    ),
    *_cells(
        README,
        "`flux1-krea-dev`⁷",
        K11,
        ("Mechanism", r"gate off: about {f}×", GATING, 2),
        ("Mechanism", r"no-gate wrapper's {f}× over plain mflux is within", COMPILE, 2),
        ("Mechanism", r"about {n}×, above its {f}× limit \(footnote ⁷\)", CEILING, 2),
    ),
    # Footnote ¹.
    *_t(README, "¹ **{}×** for `qwen-image` under CFG", (Q11, SPEEDUP, 2)),
    *_t(
        README,
        ": {} s plain mflux against {} s with TeaCache ({}× against the wrapper's own no-gate run), {} of the 48 "
        "active steps skipped in every rep. The record is",
        (Q11, VANILLA, 1),
        (Q11, WRAPPER, 1),
        (Q11, GATING, 2),
        (Q11, SKIPPED, 0),
    ),
    # Footnote ².
    *_t(
        README,
        "**{}× combined** = {}× gating × {}× compile-avoidance (vanilla {} s, no-gate {} s, gated {} s medians)",
        (B9, SPEEDUP, 2, DECOMPOSED),
        (B9, GATING, 2),
        (B9, COMPILE, 2),
        (B9, VANILLA, 1),
        (B9, NOGATE, 1),
        (B9, WRAPPER, 1),
    ),
    *_t(README, "{}/50 skips in every rep, never two consecutive", (B9, SKIPPED, 0)),
    *_t(
        README,
        "v0.10.0 measured {}× ({} s → {} s, `_artifacts/v0.10.0_bench_klein_base_9b.json`) and v0.6.0 {}× "
        "(`_artifacts/v0.6.0_bench_klein_base_9b.json`)",
        (B9_10, SPEEDUP, 2),
        (B9_10, VANILLA, 1),
        (B9_10, WRAPPER, 1),
        (B9_6, SPEEDUP, 2),
    ),
    *_t(
        README,
        "Wrapper peak memory ~{} GiB vs vanilla's ~{} GiB",
        (B9, WRAPPER_PEAK, 1),
        (B9, VANILLA_PEAK, 0),
    ),
    # Footnote ³.
    *_t(
        README,
        "**{}× combined** = {}× from step-skipping (wrapped-no-gate vs wrapped-gated) × {}× from "
        "`mx.compile`-path avoidance (vanilla vs wrapped-no-gate); vanilla {} s, no-gate {} s, gated {} s medians",
        (B4, SPEEDUP, 2, DECOMPOSED),
        (B4, GATING, 2),
        (B4, COMPILE, 2),
        (B4, VANILLA, 1),
        (B4, NOGATE, 1),
        (B4, WRAPPER, 1),
    ),
    *_t(README, "{} of the 48 active steps skipped in every rep, never two in a row; SSIM", (B4, SKIPPED, 0)),
    *_t(
        README,
        "v0.10.0 measured {}× with that extra skip ({}/50, longest run {};",
        (B4_10, SPEEDUP, 2),
        (B4_10, SKIPPED, 0),
        (B4_10, LONGEST_RUN, 0),
    ),
    *_t(
        README,
        "and v0.6.0 {}× (`_artifacts/v0.6.0_bench_klein_base_4b.json`",
        (B4_6, SPEEDUP, 2),
    ),
    *_t(
        README,
        "vanilla itself slowed by {}% ({} s to {} s), so the gated time's {}% rise ({} s)",
        (
            B4,
            against(
                B4_10,
                lambda r, o: (_med(r, "vanilla_seconds") / _med(o, "vanilla_seconds") - 1) * 100,
                "vanilla drift %",
            ),
            1,
        ),
        (B4_10, VANILLA, 1),
        (B4, VANILLA, 1),
        (
            B4,
            against(
                B4_10,
                lambda r, o: (_med(r, "wrapper_seconds") / _med(o, "wrapper_seconds") - 1) * 100,
                "gated rise %",
            ),
            1,
        ),
        (
            B4,
            against(
                B4_10, lambda r, o: _med(r, "wrapper_seconds") - _med(o, "wrapper_seconds"), "gated rise s"
            ),
            1,
        ),
    ),
    *_t(README, "the v0.4.1-era {}× attribution stays withdrawn", (B4_41, COMPILE, 2)),
    *_t(
        README,
        "has medians of {} s plain, {} s no-gate and {} s gated",
        (B4_41, VANILLA, 1),
        (B4_41, NOGATE, 1),
        (B4_41, WRAPPER, 1),
    ),
    *_t(
        README,
        "the medians were {} s, {} s and {} s. The no-gate",
        (B4_6, VANILLA, 1),
        (B4_6, NOGATE, 1),
        (B4_6, WRAPPER, 1),
    ),
    *_t(README, "slower in v0.4.1 ({} s against {} s)", (B4_41, WRAPPER, 1), (B4_6, WRAPPER, 1)),
    # Footnote ⁴.
    *_t(
        README, "it measured {}× to {}× on the 50-step Klein base recipes", (B4, COMPILE, 2), (B9, COMPILE, 2)
    ),
    # Footnote ⁵.
    *_t(
        README,
        "Result: **{}× combined** (vanilla {} s → wrapper {} s median), entirely from step-skipping at {} of the 48 "
        "active steps",
        (Z10, SPEEDUP, 2),
        (Z10, VANILLA, 1),
        (Z10, WRAPPER, 1),
        (Z10, SKIPPED, 0),
    ),
    *_t(
        README,
        "the no-gate wrapper ran at {} s median against vanilla's {} s ({}×), so the whole {}× is gating",
        (Z10, NOGATE, 1),
        (Z10, VANILLA, 1),
        (Z10, COMPILE, 2),
        (Z10, GATING, 2),
    ),
    *_t(
        README, "Peak memory {} GiB → {} GiB, from the eager", (Z10, VANILLA_PEAK, 1), (Z10, WRAPPER_PEAK, 1)
    ),
    *_t(README, "(the no-gate wrapper shows the same ~{} GiB)", (Z10, NOGATE_PEAK, 1)),
    *_t(
        README,
        "v0.7.0 reported {}× at this recipe (`_artifacts/_bench_z_image_v0_7_0.json`, wrapper {} s;",
        (Z7, SPEEDUP, 2),
        (Z7, WRAPPER, 1),
    ),
    # Footnote ⁶.
    *_t(
        README,
        "Plain mflux took {} s (reps {} / {} / {}). The no-gate run, which computes every step, took {} s "
        "({} / {} / {})",
        (F18, VANILLA, 2),
        *_reps(F18, VANILLA_REPS, 2),
        (F18, NOGATE, 2),
        *_reps(F18, NOGATE_REPS, 2),
    ),
    *_t(
        README,
        "The gated wrapper took {} s ({} / {} / {}), a median speedup of {}×. {} of 25 steps were skipped in every "
        "rep, never two in a row, and peak memory was {} GiB.",
        (F18, WRAPPER, 2),
        *_reps(F18, WRAPPER_REPS, 2),
        (F18, SPEEDUP, 2),
        (F18, SKIPPED, 0),
        (F18, WRAPPER_PEAK, 2),
    ),
    *_t(
        README,
        "{} of 25 steps were computed: {} the gate decided",
        (F18, COMPUTED, 0),
        (F18, GATE_COMPUTED, 0),
    ),
    *_t(
        README, "at most about {}× (25/19), and {}× is under that limit", (F18, CEILING, 2), (F18, SPEEDUP, 2)
    ),
    *_t(
        README,
        "about {} s on mflux 0.17.5 (the v0.6.3 record, steady reps), {} s on mflux 0.18.0 (the v0.10.0 record, "
        "reps spread from {} to {} s) and {} s on mflux 0.24.0",
        (F63, VANILLA, 0),
        (F10, VANILLA, 2),
        (F10, VANILLA_MIN, 1),
        (F10, VANILLA_MAX, 1),
        (F18, VANILLA, 2),
    ),
    *_t(
        README,
        "**{}× combined** as the median of three cold reps (vanilla {} s → gated {} s). That is above the {}× limit",
        (F10, SPEEDUP, 2),
        (F10, VANILLA, 2),
        (F10, WRAPPER, 2),
        (F10, CEILING, 2),
    ),
    *_t(
        README,
        "varied between runs ({} / {} / {} s) while the gated wrapper held steady ({} / {} / {} s)",
        *_reps(
            F10, (ranked("vanilla_seconds", 0), ranked("vanilla_seconds", 1), ranked("vanilla_seconds", 2))
        ),
        *_reps(
            F10, (ranked("wrapper_seconds", 0), ranked("wrapper_seconds", 1), ranked("wrapper_seconds", 2))
        ),
    ),
    *_t(README, "Even fastest to fastest, {}× is above the limit", (F10, FASTEST_RATIO, 2)),
    *_t(
        README, "The no-gate wrapper took {} s, {}× ahead of plain mflux", (F10, NOGATE, 2), (F10, COMPILE, 2)
    ),
    *_t(
        README,
        "The earlier {}× (v0.6.3, mflux 0.17.5: plain {} s → gated {} s) is also above the {}× limit",
        (F63, SPEEDUP, 2),
        (F63, VANILLA, 2),
        (F63, WRAPPER, 2),
        (F63, CEILING, 2),
    ),
    # Footnote ⁷.
    *_t(
        README,
        "Result: **{}× combined** as the median of three cold reps (vanilla {} s → gated {} s). The gated run was "
        "about {}× faster than the no-gate run; the no-gate wrapper's {}× over plain mflux is within",
        (K11, SPEEDUP, 2),
        (K11, VANILLA, 1),
        (K11, WRAPPER, 1),
        (K11, GATING, 2),
        (K11, COMPILE, 2),
    ),
    *_t(README, "{} of the 26 active steps skipped in every rep (the table's", (K11, SKIPPED, 0)),
    *_t(README, "so they can give at most about {}× (28/18)", (K11, CEILING, 2)),
    *_t(
        README,
        "The three TeaCache reps were {} / {} / {} s: one was much faster than the other two, against {} / {} / {} s "
        "for plain mflux",
        *_reps(K11, WRAPPER_REPS),
        *_reps(K11, VANILLA_REPS),
    ),
    *_t(
        README,
        "Peak memory {} GiB. Full report: `_artifacts/v0.11.0_bench_krea_dev.json`",
        (K11, WRAPPER_PEAK, 1),
    ),
    # How the speedup happens, chip table, limitations.
    *_t(
        README,
        "On FLUX.1-dev at 25 steps, {} of 25 steps are skippable and this is where the {}× speedup",
        (F18, SKIPPED, 0),
        (F18, SPEEDUP, 2),
    ),
    *_t(
        README,
        "the subprocess-per-rep bench measures it at {}× to {}× (footnotes ³ and ²)",
        (B4, COMPILE, 2),
        (B9, COMPILE, 2),
    ),
    *_t(
        README,
        "base-4b skips {}/50 steps for a measured {}× combined speedup (subprocess-per-rep, footnote ³), and base-9b "
        "skips {}/50 steps for {}× (footnote ²)",
        (B4, SKIPPED, 0),
        (B4, SPEEDUP, 2, DECOMPOSED),
        (B9, SKIPPED, 0),
        (B9, SPEEDUP, 2),
    ),
    *_t(
        README,
        "chip-independent ({}×¹ Qwen-Image, {}×⁷ Krea (above its {}× limit), {}×⁶ FLUX.1-dev",
        (Q11, SPEEDUP, 2),
        (K11, SPEEDUP, 2),
        (K11, CEILING, 2),
        (F18, SPEEDUP, 2),
    ),
    *_t(
        README,
        "Qwen-Image {}×¹, klein-base-9b {}×², klein-base-4b {}×³, z-image-base {}×⁵, FLUX.1 Krea {}×⁷ (above its {}× "
        "limit), FLUX.1-dev {}×⁶",
        (Q11, SPEEDUP, 2),
        (B9, SPEEDUP, 2),
        (B4, SPEEDUP, 2, DECOMPOSED),
        (Z10, SPEEDUP, 2),
        (K11, SPEEDUP, 2),
        (K11, CEILING, 2),
        (F18, SPEEDUP, 2),
    ),
    *_t(
        README,
        "measures **{}× combined**³ on M1 Max under the subprocess-per-rep harness (v0.12.0, where {} skipped steps "
        "alone allow {}×; v0.10.0 measured {}× and v0.6.0 {}× (above its {}× limit, 50/41) before the gate's range "
        "check, v0.4.1 advertised {}× combined under same-process measurement, above its {}× limit (50/41)",
        (B4, SPEEDUP, 2),
        (B4, SKIPPED, 0),
        (B4, CEILING, 2),
        (B4_10, SPEEDUP, 2),
        (B4_6, SPEEDUP, 2),
        (B4_6, CEILING, 2),
        (B4_41, SPEEDUP, 2),
        (B4_41, CEILING, 2),
    ),
    *_t(
        README,
        "measures **{}× combined wall-clock** vs vanilla on M1 Max at that recipe (v0.12.0, footnote ³: {}× gating + "
        "{}× compile-avoidance; {}/{} skips, SSIM 0.985; v0.10.0 measured {}× and v0.6.0 {}×, which is above its {}× "
        "limit, 50/41). v0.4.1's same-process bench reported {}× combined (also above that {}× limit) and decomposed "
        "it {}× gating / {}× compile-avoidance",
        (B4, SPEEDUP, 2, DECOMPOSED),
        (B4, GATING, 2),
        (B4, COMPILE, 2),
        (B4, SKIPPED, 0),
        (B4, STEPS_TOTAL, 0),
        (B4_10, SPEEDUP, 2),
        (B4_6, SPEEDUP, 2),
        (B4_6, CEILING, 2),
        (B4_41, SPEEDUP, 2),
        (B4_41, CEILING, 2),
        (B4_41, GATING, 2),
        (B4_41, COMPILE, 2),
    ),
    *_t(
        README,
        "measures **{}× combined wall-clock**² ({}× gating + {}× compile-avoidance; {} skipped steps alone allow {}×, "
        "and v0.10.0 and v0.6.0 measured {}× and {}×, both above that limit, 50/37), {}/{} steps skipped",
        (B9, SPEEDUP, 2),
        (B9, GATING, 2),
        (B9, COMPILE, 2),
        (B9, SKIPPED, 0),
        (B9, CEILING, 2),
        (B9_10, SPEEDUP, 2),
        (B9_6, SPEEDUP, 2),
        (B9, SKIPPED, 0),
        (B9, STEPS_TOTAL, 0),
    ),
    *_t(README, "The {}×⁶ measurement is from M1 Max / FLUX.1-dev", (F18, SPEEDUP, 2)),
    *_t(
        README,
        "The later median ratio, {}×, is below {}×.",
        (Q11, SPEEDUP, 2),
        (Q10, SPEEDUP, 2),
    ),
    *_t(
        README,
        "the 0.10.0 Qwen-Image figures ({}× against plain mflux, plus an unexplained {}× with the gate off)",
        (Q10, SPEEDUP, 2),
        (Q10, COMPILE, 2),
    ),
    *_t(
        README,
        "measured an active peak of {}–{} GiB after model load (the load phase reads about zero, so this is also the "
        "whole-run peak) and {}× combined",
        (Q11, PEAK_LOW, 2),
        (Q11, PEAK_HIGH, 2),
        (Q11, SPEEDUP, 2),
    ),
    *_t(
        README,
        "gives an estimate of roughly {}–{} GiB, computed rather than observed",
        (Q11, PEAK_CACHE_LOW, 1),
        (Q11, PEAK_CACHE_HIGH, 1),
    ),
    *_t(
        README,
        "The {}× is above the {}× that {} skipped steps of 50 allow (50/17). The record does not say why, and it does "
        "not explain the same report's {}× no-gate gap",
        (Q10, SPEEDUP, 2),
        (Q10, CEILING, 2),
        (Q10, SKIPPED, 0),
        (Q10, COMPILE, 2),
    ),
    # COMPARISON.md prose (outside the generated blocks).
    *_t(
        COMPARISON,
        "measured this variant at {}× on mflux 0.24.0 and MLX 0.32.2, all from {} of 25 skipped steps",
        (F18, SPEEDUP, 2),
        (F18, SKIPPED, 0),
    ),
    *_t(
        COMPARISON,
        "That is under its {}× limit (footnote ⁶ has the arithmetic). The wrapper",
        (F18, CEILING, 2),
    ),
    *_t(COMPARISON, "the same time as plain mflux ({} s both)", (F18, VANILLA, 2)),
    *_t(COMPARISON, "An older {}× from mflux 0.18.0 is above that limit", (F10, SPEEDUP, 2)),
    *_krea_comparison(COMPARISON),
    *_t(
        COMPARISON,
        "splits this variant's {}× multi-run speedup (measured with 0.12.0) into {}× from skipped steps and a "
        "noise-level {}× from",
        (B4, SPEEDUP, 2, DECOMPOSED),
        (B4, GATING, 2),
        (B4, COMPILE, 2),
    ),
    *_t(
        COMPARISON,
        "measured this variant's {}× multi-run speedup as entirely step-skipping",
        (Z10, SPEEDUP, 2),
    ),
    *_t(
        COMPARISON,
        "splits this variant's {}× multi-run speedup into {}× from skipped steps and a small {}× from",
        (B9, SPEEDUP, 2, DECOMPOSED),
        (B9, GATING, 2),
        (B9, COMPILE, 2),
    ),
    *_t(
        COMPARISON,
        "a peak-memory drop from about {} GiB to {} GiB at that recipe",
        (B9, VANILLA_PEAK, 0),
        (B9, WRAPPER_PEAK, 1),
    ),
    *_t(COMPARISON, "the record behind this variant's {}× multi-run speedup", (Q11, SPEEDUP, 2)),
    *_t(
        COMPARISON,
        "The single run above skipped {} of {} steps, which allows at most about {}× ({}/{}); its {}× is under that",
        (RJ, rj("flux1-krea-dev", "b", "skipped"), 0),
        (RJ, rj("flux1-krea-dev", "steps"), 0),
        (RJ, rj_ceiling("flux1-krea-dev"), 2),
        (RJ, rj("flux1-krea-dev", "steps"), 0),
        (RJ, rj_computed("flux1-krea-dev"), 0),
        (RJ, rj_speedup("flux1-krea-dev"), 2),
    ),
    *_t(
        COMPARISON,
        "so A and B both peak at {} GiB here",
        (RJ, rj_gib("z-image-base", "a", "mlx_peak_generation_bytes"), 1),
    ),
    *_t(
        COMPARISON,
        "so A and B both peak at {} GiB here",
        (RJ, rj_gib("z-image-base", "b", "mlx_peak_generation_bytes"), 1),
    ),
    # docs/comparison prose.
    *_t(CMP_DEV, "measured this variant at {}× combined, all from step-skipping", (F18, SPEEDUP, 2)),
    *_t(CMP_DEV, "The {}× is under its {}× limit (footnote ⁶)", (F18, SPEEDUP, 2), (F18, CEILING, 2)),
    *_t(CMP_DEV, "took the same time as plain mflux ({} s both)", (F18, VANILLA, 2)),
    *_t(CMP_DEV, "An older run on mflux 0.18.0 measured {}× ([", (F10, SPEEDUP, 2)),
    *_t(
        CMP_DEV,
        "That is above the same limit. Its plain-mflux reps ran from {} to {} s and its no-gate reps from {} to {} s, "
        "a no-gate lead of {}× on the medians",
        (F10, VANILLA_MIN, 1),
        (F10, VANILLA_MAX, 1),
        (F10, NOGATE_MIN, 1),
        (F10, NOGATE_MAX, 1),
        (F10, COMPILE, 2),
    ),
    *_krea_comparison(CMP_KREA),
    *_t(CMP_KREA, "On this run the gate skips {} of 28 steps", (RJ, rj("flux1-krea-dev", "b", "skipped"), 0)),
    *_t(
        CMP_B4,
        "splits this variant's {}× multi-run speedup (measured with 0.12.0) into {}× from skipped steps and a "
        "noise-level {}× from",
        (B4, SPEEDUP, 2, DECOMPOSED),
        (B4, GATING, 2),
        (B4, COMPILE, 2),
    ),
    *_t(
        CMP_B4,
        "the MLX peak-generation difference below ({} GiB vs {} GiB)",
        (RJ, rj_gib("klein-base-4b", "a", "mlx_peak_generation_bytes"), 1),
        (RJ, rj_gib("klein-base-4b", "b", "mlx_peak_generation_bytes"), 1),
    ),
    *_t(CMP_B4, "On this run the gate skips {} of 50 steps", (RJ, rj("klein-base-4b", "b", "skipped"), 0)),
    *_t(
        CMP_B9,
        "splits this variant's {}× multi-run speedup into {}× from skipped steps and a small {}× from",
        (B9, SPEEDUP, 2, DECOMPOSED),
        (B9, GATING, 2),
        (B9, COMPILE, 2),
    ),
    *_t(
        CMP_B9,
        "a peak-memory drop from about {} GiB to {} GiB at that recipe",
        (B9, VANILLA_PEAK, 0),
        (B9, WRAPPER_PEAK, 1),
    ),
    *_t(
        CMP_B9,
        "in the table above ({} GiB vs {} GiB)",
        (RJ, rj_gib("klein-base-9b", "a", "mlx_peak_generation_bytes"), 1),
        (RJ, rj_gib("klein-base-9b", "b", "mlx_peak_generation_bytes"), 1),
    ),
    *_t(CMP_B9, "On this run the gate skips {} of 50 steps", (RJ, rj("klein-base-9b", "b", "skipped"), 0)),
    *_t(CMP_QWEN, "the record behind this variant's {}× multi-run speedup", (Q11, SPEEDUP, 2)),
    *_t(CMP_QWEN, "On this run the gate skips {} of 50 steps", (RJ, rj("qwen-image", "b", "skipped"), 0)),
    *_t(
        CMP_Z,
        "measured this variant's {}× multi-run speedup, measured on mflux 0.18.0, as entirely step-skipping",
        (Z10, SPEEDUP, 2),
    ),
    *_t(
        CMP_Z,
        "so A and B both peak at {} GiB in the table above",
        (RJ, rj_gib("z-image-base", "b", "mlx_peak_generation_bytes"), 1),
    ),
    *_t(CMP_Z, "On this run the gate skips {} of 50 steps", (RJ, rj("z-image-base", "b", "skipped"), 0)),
    *_t(CMP_Z, "most of why SSIM lands at {}", (RJ, rj("z-image-base", "ssim"), 2)),
    *_t(
        CMP_DISTILLED,
        "| distilled | 4 | {} | {}× | {} | 1.000 | 0.376 |",
        (D4, warm("distilled"), 0),
        (D4, warm_ratio("base", "distilled"), 2),
        (D4, condition("distilled", "peak_memory_gb"), 1),
    ),
    *_t(
        CMP_DISTILLED,
        "| base | 50 | {} | 1.00× | {} | 0.376 | 1.000 |",
        (D4, warm("base"), 0),
        (D4, condition("base", "peak_memory_gb"), 1),
    ),
    *_t(
        CMP_DISTILLED,
        "| base+TeaCache | 50 (−{} skipped) | {} | {}× | {} | 0.384 |",
        (D4, condition("base-teacache", "skipped_median"), 0),
        (D4, warm("base-teacache"), 0),
        (D4, warm_ratio("base", "base-teacache"), 2),
        (D4, condition("base-teacache", "peak_memory_gb"), 1),
    ),
    *_t(
        CMP_DISTILLED,
        "| distilled | 4 | {} | {}× | {} | 1.000 | 0.541 |",
        (D9, warm("distilled"), 0),
        (D9, warm_ratio("base", "distilled"), 2),
        (D9, condition("distilled", "peak_memory_gb"), 1),
    ),
    *_t(
        CMP_DISTILLED,
        "| base | 50 | {} | 1.00× | {} | 0.541 | 1.000 |",
        (D9, warm("base"), 0),
        (D9, condition("base", "peak_memory_gb"), 1),
    ),
    *_t(
        CMP_DISTILLED,
        "| base+TeaCache | 50 (−{} skipped) | {} | {}× | {} | 0.553 |",
        (D9, condition("base-teacache", "skipped_median"), 0),
        (D9, warm("base-teacache"), 0),
        (D9, warm_ratio("base", "base-teacache"), 2),
        (D9, condition("base-teacache", "peak_memory_gb"), 1),
    ),
    *_t(
        CMP_DISTILLED,
        "the two distilled reps differ by about {}% ({} s against {} s)",
        (
            D9,
            Value(
                "distilled rep gap %",
                lambda r: (
                    (
                        _d(r["conditions"]["distilled"]["rep_seconds"][0])
                        / _d(r["conditions"]["distilled"]["rep_seconds"][1])
                        - 1
                    )
                    * 100
                ),
            ),
            0,
        ),
        (D9, Value("distilled rep 0", lambda r: _d(r["conditions"]["distilled"]["rep_seconds"][0])), 1),
        (D9, Value("distilled rep 1", lambda r: _d(r["conditions"]["distilled"]["rep_seconds"][1])), 1),
    ),
    # docs/variants.
    *_t(
        V_DEV,
        "skips {}/25 steps (never two in a row) and produces a measured {}× wall-clock speedup",
        (F18, SKIPPED, 0),
        (F18, SPEEDUP, 2),
    ),
    *_t(
        V_DEV,
        "plain mflux {} s, no-gate wrapper {} s, gated wrapper {} s, commit",
        (F18, VANILLA, 2),
        (F18, NOGATE, 2),
        (F18, WRAPPER, 2),
    ),
    *_t(
        V_DEV,
        "Of the 25 steps, {} were computed: {} the gate decided",
        (F18, COMPUTED, 0),
        (F18, GATE_COMPUTED, 0),
    ),
    *_t(V_DEV, "can therefore give at most about {}× (25/19)", (F18, CEILING, 2)),
    *_t(
        V_DEV,
        "measured {}×, and {}× fastest to fastest. Both are above the {}× limit worked out above. In that run the "
        "plain-mflux reps varied between {} and {} s",
        (F10, SPEEDUP, 2),
        (F10, FASTEST_RATIO, 2),
        (F10, CEILING, 2),
        (F10, VANILLA_MIN, 1),
        (F10, VANILLA_MAX, 1),
    ),
    *_cells(
        V_KREA,
        "vanilla",
        K11,
        ("Rep seconds", r"^{f} /", VANILLA_REPS[0], 1),
        ("Rep seconds", r"/ {f} /", VANILLA_REPS[1], 1),
        ("Rep seconds", r"/ {f}$", VANILLA_REPS[2], 1),
        ("Median", r"^{f} s$", VANILLA, 1),
    ),
    *_cells(
        V_KREA,
        "wrapper, no gate (`rel_l1_thresh=0.0`)",
        K11,
        ("Rep seconds", r"^{f} /", NOGATE_REPS[0], 1),
        ("Rep seconds", r"/ {f} /", NOGATE_REPS[1], 1),
        ("Rep seconds", r"/ {f}$", NOGATE_REPS[2], 1),
        ("Median", r"^{f} s$", NOGATE, 1),
    ),
    *_cells(
        V_KREA,
        "wrapper, default 0.30",
        K11,
        ("Rep seconds", r"^{f} /", WRAPPER_REPS[0], 1),
        ("Rep seconds", r"/ {f} /", WRAPPER_REPS[1], 1),
        ("Rep seconds", r"/ {f}$", WRAPPER_REPS[2], 1),
        ("Median", r"^{f} s$", WRAPPER, 1),
        ("Skipped", r"^{f} in every rep", SKIPPED, 0),
    ),
    *_t(
        V_KREA,
        "**{}× combined** against plain mflux (limit {}×). The gated run is about **{}×** faster than the no-gate run "
        "({} s → {} s). The no-gate wrapper's {}× over plain mflux",
        (K11, SPEEDUP, 2),
        (K11, CEILING, 2),
        (K11, GATING, 2),
        (K11, NOGATE, 1),
        (K11, WRAPPER, 1),
        (K11, COMPILE, 2),
    ),
    *_t(
        V_KREA,
        "Both the {}× and the {}× are above the {}× limit that {} skipped steps of {} allow",
        (K11, SPEEDUP, 2),
        (K11, GATING, 2),
        (K11, CEILING, 2),
        (K11, SKIPPED, 0),
        (K11, STEPS_TOTAL, 0),
    ),
    *_t(V_KREA, "Peak memory {} GiB in every condition", (K11, WRAPPER_PEAK, 1)),
    *_t(
        V_KREA,
        "against one plain run of {} s; the three-rep bench below has a plain median of {} s",
        (SW_K, SWEEP_PLAIN, 0),
        (K11, VANILLA, 0),
    ),
    *_t(
        V_KREA,
        "(for example {}× at 0.80, where {} skipped steps out of {} give at most {}×)",
        (SW_K, sweep_at("0.8", "speedup_vs_vanilla_single_rep"), 2),
        (SW_K, sweep_at("0.8", "skipped"), 0),
        (SW_K, SWEEP_STEPS, 0),
        (SW_K, sweep_ceiling("0.8"), 2),
    ),
    *_t(
        V_K4,
        "the same mechanism measures {}× (`_artifacts/v0.12.0_bench_klein_base_4b.json`",
        (B4, COMPILE, 2),
    ),
    *_t(V_K9, "The same mechanism measures {}× on the 50-step klein-base-9b CFG recipe", (B9, COMPILE, 2)),
    *_cells(
        V_B4, "vanilla", B4, ("Median", r"^{f} s$", VANILLA, 1), ("Peak", r"^~{f} GiB$", VANILLA_PEAK, 1)
    ),
    *_cells(
        V_B4,
        "wrapper, no gate (compile-avoidance only)",
        B4,
        ("Median", r"^{f} s$", NOGATE, 1),
        ("Peak", r"^~{f} GiB$", NOGATE_PEAK, 1),
    ),
    *_cells(
        V_B4,
        "wrapper, gated (full TeaCache)",
        B4,
        ("Median", r"^{f} s$", WRAPPER, 1),
        ("Peak", r"^~{f} GiB$", WRAPPER_PEAK, 1),
    ),
    *_t(
        V_B4,
        "**Combined speedup: {}×**. The {} skipped steps of 50 alone allow {}× (50/42); the rest is the {}× compile "
        "avoidance",
        (B4, SPEEDUP, 2),
        (B4, SKIPPED, 0),
        (B4, CEILING, 2),
        (B4, COMPILE, 2),
    ),
    *_t(
        V_B4,
        "(v0.10.0 measured {}× and v0.6.0 {}×, with one more skip; the {}× is above its {}× limit, 50/41",
        (B4_10, SPEEDUP, 2),
        (B4_6, SPEEDUP, 2),
        (B4_6, SPEEDUP, 2),
        (B4_6, CEILING, 2),
    ),
    *_t(V_B4, "**Gating contribution (v0.4.1 effect): {}×**", (B4, GATING, 2)),
    *_t(V_B4, "avoidance (v0.4 effect): {}×** — effectively noise", (B4, COMPILE, 2)),
    *_t(V_B4, "memory (~{}% peak drop)", (B4, PEAK_DROP_PCT, 0)),
    *_t(V_B4, "Skip count stable across reps: {} of 48 active steps", (B4, SKIPPED, 0)),
    *_t(
        V_B4,
        "v0.4.1 reported {}× combined on this variant, decomposed as {}× gating × {}× compile-avoidance",
        (B4_41, SPEEDUP, 2, DECOMPOSED),
        (B4_41, GATING, 2),
        (B4_41, COMPILE, 2),
    ),
    *_t(
        V_B4,
        "landed {}× (v0.6.0, above its {}× limit of {} skipped steps in 50, 50/41) and {}× (v0.10.0)",
        (B4_6, SPEEDUP, 2),
        (B4_6, CEILING, 2),
        (B4_6, SKIPPED, 0),
        (B4_10, SPEEDUP, 2),
    ),
    *_t(V_B4, "(`_artifacts/v0.10.0_bench_klein_base_4b.json`, the {}×, mflux 0.18.0)", (B4_10, SPEEDUP, 2)),
    *_t(
        V_B4,
        "essentially all the work ({}–{}×)",
        (B4, GATING, 2),
        (B4_6, GATING, 2),
    ),
    *_t(
        V_B4,
        "has medians of {} s plain, {} s no-gate and {} s gated",
        (B4_41, VANILLA, 1),
        (B4_41, NOGATE, 1),
        (B4_41, WRAPPER, 1),
    ),
    *_t(
        V_B4,
        "the medians were {} s, {} s and {} s. The no-gate",
        (B4_6, VANILLA, 1),
        (B4_6, NOGATE, 1),
        (B4_6, WRAPPER, 1),
    ),
    *_cells(
        V_B9, "vanilla", B9, ("Median", r"^{f} s$", VANILLA, 1), ("Peak", r"^~{f} GiB$", VANILLA_PEAK, 0)
    ),
    *_cells(
        V_B9,
        "wrapper, no gate (compile-avoidance only)",
        B9,
        ("Median", r"^{f} s$", NOGATE, 1),
        ("Peak", r"^~{f} GiB$", NOGATE_PEAK, 1),
    ),
    *_cells(
        V_B9,
        "wrapper, gated (full TeaCache)",
        B9,
        ("Median", r"^{f} s$", WRAPPER, 1),
        ("Peak", r"^~{f} GiB$", WRAPPER_PEAK, 1),
    ),
    *_t(
        V_B9,
        "**Combined speedup: {}×**. The {} skipped steps of 50 alone allow {}× (50/37); the rest is the {}× compile "
        "avoidance",
        (B9, SPEEDUP, 2),
        (B9, SKIPPED, 0),
        (B9, CEILING, 2),
        (B9, COMPILE, 2),
    ),
    *_t(
        V_B9,
        "(v0.10.0 measured {}× and v0.6.0 {}× at the same recipe; both are above that limit too",
        (B9_10, SPEEDUP, 2),
        (B9_6, SPEEDUP, 2),
    ),
    *_t(V_B9, "**Gating contribution (v0.4.1 effect): {}×**", (B9, GATING, 2)),
    *_t(V_B9, "avoidance (v0.4 effect): {}×** — small", (B9, COMPILE, 2)),
    *_t(V_B9, "Skip count stable across reps: {} of 48 active steps", (B9, SKIPPED, 0)),
    *_t(V_B9, "v0.5.0 reported {}× combined on this variant", (VAL, VAL_SPEEDUP, 2)),
    *_t(
        V_B9,
        "Plain mflux took {} s and the wrapper {} s. Plain mflux peaked at {} GiB, above the {} GiB cap",
        (VAL, plain("vanilla_seconds"), 1),
        (VAL, plain("wrapper_seconds"), 1),
        (VAL, plain("vanilla_peak_memory_gb"), 2),
        (VAL, plain("mlx_memory_cap_gb"), 0),
    ),
    *_t(V_B9, "The wrapper peaked at {} GiB. That fits", (VAL, plain("wrapper_peak_memory_gb"), 2)),
    *_t(
        V_B9,
        "The {}× at v0.6.0 (`_artifacts/v0.6.0_bench_klein_base_9b.json`, mflux 0.17.5) and the {}× at v0.10.0 "
        "(`_artifacts/v0.10.0_bench_klein_base_9b.json`, mflux 0.18.0) and v0.12.0 are a different recipe",
        (B9_6, SPEEDUP, 2),
        (B9_10, SPEEDUP, 2),
    ),
    *_t(
        V_B9,
        "and the {}× at v0.10.0 (`_artifacts/v0.10.0_bench_klein_base_9b.json`, mflux 0.18.0) and v0.12.0 are",
        (B9, SPEEDUP, 2),
    ),
    *_t(
        V_B9,
        "About {}× of each of the {}× and {}× comes from compile avoidance",
        (B9_6, COMPILE, 2),
        (B9_6, SPEEDUP, 2),
        (B9_10, SPEEDUP, 2),
    ),
    *_t(V_B9, "About {}× of each of the", (B9_10, COMPILE, 2)),
    *_t(
        V_B9,
        "The {}× headline number ({}× from skipped steps × {}× from compile avoidance) is reproducible",
        (B9, SPEEDUP, 2, DECOMPOSED),
        (B9, GATING, 2),
        (B9, COMPILE, 2),
    ),
    *_t(
        V_QWEN,
        "the wrapper took the generation from {} s to {} s ({}× on this run, preview decoding included), skipping {} "
        "of 50 steps at SSIM {} against vanilla",
        (RJ, rj("qwen-image", "a", "generation_seconds"), 1),
        (RJ, rj("qwen-image", "b", "generation_seconds"), 1),
        (RJ, rj_speedup("qwen-image"), 2),
        (RJ, rj("qwen-image", "b", "skipped"), 0),
        (RJ, rj("qwen-image", "ssim"), 2),
    ),
    *_cells(
        V_QWEN,
        "vanilla",
        Q11,
        ("Rep seconds", r"^{f} /", VANILLA_REPS[0], 1),
        ("Rep seconds", r"/ {f} /", VANILLA_REPS[1], 1),
        ("Rep seconds", r"/ {f}$", VANILLA_REPS[2], 1),
        ("Median", r"^{f} s$", VANILLA, 1),
    ),
    *_cells(
        V_QWEN,
        "wrapper, no gate (`rel_l1_thresh=0.0`)",
        Q11,
        ("Rep seconds", r"^{f} /", NOGATE_REPS[0], 1),
        ("Rep seconds", r"/ {f} /", NOGATE_REPS[1], 1),
        ("Rep seconds", r"/ {f}$", NOGATE_REPS[2], 1),
        ("Median", r"^{f} s$", NOGATE, 1),
    ),
    *_cells(
        V_QWEN,
        "wrapper, default 0.30",
        Q11,
        ("Rep seconds", r"^{f} /", WRAPPER_REPS[0], 1),
        ("Rep seconds", r"/ {f} /", WRAPPER_REPS[1], 1),
        ("Rep seconds", r"/ {f}$", WRAPPER_REPS[2], 1),
        ("Median", r"^{f} s$", WRAPPER, 1),
        ("Skipped", r"^{f} in every rep", SKIPPED, 0),
    ),
    *_t(
        V_QWEN,
        "That run gave {}× combined against plain mflux, of which {}× is gating. The no-gate wrapper's {}× over "
        "plain mflux",
        (Q11, SPEEDUP, 2),
        (Q11, GATING, 2),
        (Q11, COMPILE, 2),
    ),
    *_t(V_QWEN, "In every rep {} of the 48 active steps skipped", (Q11, SKIPPED, 0)),
    *_t(
        V_QWEN,
        "its vanilla time ({} s) gave {}×, which is above the {}× that {} skipped steps of 50 allow (50/17). The record "
        "does not say why. It does not explain the {}× no-gate gap either: {} s against a {} s spread in its vanilla "
        "reps",
        (Q10, VANILLA, 1),
        (Q10, SPEEDUP, 2),
        (Q10, CEILING, 2),
        (Q10, SKIPPED, 0),
        (Q10, COMPILE, 2),
        (Q10, NOGATE_GAP, 1),
        (Q10, VANILLA_SPREAD, 1),
    ),
    *_t(V_QWEN, "The later median ratio, {}×, is below {}×.", (Q11, SPEEDUP, 2), (Q10, SPEEDUP, 2)),
    *_t(V_QWEN, "The report's own timings add up to about {} minutes", (Q10, TOTAL_MINUTES, 0)),
    *_t(
        V_QWEN,
        "The peak, {}–{} GiB in every condition, is the active peak",
        (Q11, PEAK_LOW, 2),
        (Q11, PEAK_HIGH, 2),
    ),
    *_t(
        V_QWEN,
        "The cache pool read {} GiB after decode. Adding that reading gives roughly {}–{} GiB",
        (Q11, CACHE, 2),
        (Q11, PEAK_CACHE_LOW, 1),
        (Q11, PEAK_CACHE_HIGH, 1),
    ),
    *_t(
        V_QWEN,
        "against one plain run of {} s. The three-rep bench below has a plain median of {} s, so the 0.25 and 0.30 rows",
        (SW_Q, SWEEP_PLAIN, 0),
        (Q11, VANILLA, 0),
    ),
    *_t(
        V_QWEN,
        "({} steps over {} computed gives {}×, over {} gives {}×)",
        (SW_Q, SWEEP_STEPS, 0),
        (SW_Q, sweep_computed("0.25"), 0),
        (SW_Q, sweep_ceiling("0.25"), 2),
        (SW_Q, sweep_computed("0.3"), 0),
        (SW_Q, sweep_ceiling("0.3"), 2),
    ),
    *_cells(V_Z, "vanilla", Z10, ("Median", r"^{f} s$", VANILLA, 1), ("Peak", r"^{f} GiB$", VANILLA_PEAK, 1)),
    *_cells(
        V_Z,
        "wrapper, no gate (compile path bypass only)",
        Z10,
        ("Median", r"^{f} s$", NOGATE, 1),
        ("Peak", r"^{f} GiB$", NOGATE_PEAK, 1),
    ),
    *_cells(
        V_Z,
        "wrapper, gated (full TeaCache)",
        Z10,
        ("Median", r"^{f} s$", WRAPPER, 1),
        ("Peak", r"^{f} GiB$", WRAPPER_PEAK, 1),
    ),
    *_t(
        V_Z,
        "**Combined speedup: {}×** (gating {}×, compile-avoidance {}×)",
        (Z10, SPEEDUP, 2),
        (Z10, GATING, 2),
        (Z10, COMPILE, 2),
    ),
    *_t(V_Z, "runs at vanilla speed ({} s vs {} s;", (Z10, NOGATE, 1), (Z10, VANILLA, 1)),
    *_t(V_Z, "Peak memory drops from {} GiB to {} GiB", (Z10, VANILLA_PEAK, 1), (Z10, WRAPPER_PEAK, 1)),
    *_t(V_Z, "{} of 48 active steps skipped at `rel_l1_thresh=0.12`", (Z10, SKIPPED, 0)),
    *_t(
        V_Z,
        "against one plain run of {} s, while the three-rep bench above has a plain median of {} s",
        (SW_Z, SWEEP_PLAIN, 0),
        (Z10, VANILLA, 0),
    ),
    *_t(
        V_Z,
        "v0.7.0 reported {}× at this recipe ({} s → {} s,",
        (Z7, SPEEDUP, 2),
        (Z7, VANILLA, 1),
        (Z7, WRAPPER, 1),
    ),
    # docs/m3-plus-tradeoff.md and docs/calibration.md.
    *_t(
        M3,
        "`_artifacts/v0.12.0_bench_klein_base_9b.json`): {}× on Klein base 4B, which is noise, and a small {}× on "
        "Klein base 9B",
        (B4, COMPILE, 2),
        (B9, COMPILE, 2),
    ),
    *_t(
        CALIBRATION,
        "That gives {} skips instead of {}, and {}× ({}× from skipped steps × {}× from compile avoidance) instead of {}×.",
        (B4, SKIPPED, 0),
        (B4_10, SKIPPED, 0),
        (B4, SPEEDUP, 2, DECOMPOSED),
        (B4, GATING, 2),
        (B4, COMPILE, 2),
        (B4_10, SPEEDUP, 2),
    ),
    *_t(
        CALIBRATION,
        "for the same model: {} s against {} s for Qwen-Image, {} s against {} s for Z-Image, {} s against {} s for "
        "Krea",
        (SW_Q, SWEEP_PLAIN, 0),
        (Q11, VANILLA, 0),
        (SW_Z, SWEEP_PLAIN, 0),
        (Z10, VANILLA, 0),
        (SW_K, SWEEP_PLAIN, 0),
        (K11, VANILLA, 0),
    ),
    *_t(
        CALIBRATION,
        "(at 0.80, {} computed steps out of {} at the plain {} s give {} s, and the gated run took {} s)",
        (SW_K, sweep_computed("0.8"), 0),
        (SW_K, SWEEP_STEPS, 0),
        (SW_K, SWEEP_PLAIN, 1),
        (SW_K, sweep_plain_share("0.8"), 2),
        (SW_K, sweep_at("0.8", "wrapper_seconds"), 2),
    ),
    # The single-rep speedup column of the sweep tables on the variant pages.
    *(
        _cell(V_KREA, t, "Single-rep speedup", HEADLINE, SW_K, sweep_speedup(t), 2)
        for t in ("0.30", "0.35", "0.40", "0.60")
    ),
    *(
        _cell(V_QWEN, t, "Single-rep speedup", HEADLINE, SW_Q, sweep_speedup(t), 2)
        for t in ("0.15", "0.20", "0.25", "0.30")
    ),
    # The skip limits stated under the distilled-vs-base tables.
    *_t(
        CMP_DISTILLED,
        "Eight skipped steps out of 50 leave 42 computed, so skipping alone gives at most about {}× (50/42)",
        (D4, skip_limit("base-teacache"), 2),
    ),
    *_t(
        CMP_DISTILLED,
        "Fourteen skipped steps out of 50 leave 36 computed, so skipping alone gives at most about {}× (50/36)",
        (D9, skip_limit("base-teacache"), 2),
    ),
)

# Docs that state a record's gaps without quoting a prose figure from it: each comparison page's generated details
# block quotes the comparison report.
COMPARISON_SLUGS = (
    "flux1-dev",
    "flux1-krea-dev",
    "klein-base-4b",
    "klein-base-9b",
    "qwen-image",
    "z-image-base",
)
NAMES_WITHOUT_FIGURES: tuple[tuple[str, str], ...] = tuple(
    (f"docs/comparison/{s}.md", RJ) for s in COMPARISON_SLUGS
)

DOC_RECORD_PAIRS: tuple[tuple[str, str], ...] = tuple(
    sorted({(f.doc, f.record) for f in FIGURES} | set(NAMES_WITHOUT_FIGURES))
)

# Every record a figure comes from.
RECORDS: tuple[str, ...] = tuple(sorted({f.record for f in FIGURES} | {RJ}))

# Docs whose quoted install stamps are checked.
SCOPED_DOCS: tuple[str, ...] = tuple(sorted({f.doc for f in FIGURES} | {d for d, _ in NAMES_WITHOUT_FIGURES}))


def _pair_params() -> list[Any]:
    return [pytest.param(doc, record, id=f"{doc}->{Path(record).name}") for doc, record in DOC_RECORD_PAIRS]


@cache
def _doc(doc: str) -> str:
    return (DOC_ROOT / doc).read_text(encoding="utf-8")


@cache
def _record(record: str) -> dict[str, Any]:
    return load_record(record)


def paragraphs(text: str) -> list[str]:
    """``text`` split into paragraphs: blocks between blank lines, with every table row a unit of its own."""
    units: list[str] = []
    block: list[str] = []
    for line in [*text.splitlines(), ""]:
        if not line.strip() or line.lstrip().startswith("|"):
            if block:
                units.append("\n".join(block))
                block = []
            if line.strip():
                units.append(line)
        else:
            block.append(line)
    return units


def paragraph_of(text: str, figure: Figure) -> list[str]:
    """Every paragraph or table row holding the figure's place."""
    if isinstance(figure.where, Row):
        return [
            u
            for u in paragraphs(text)
            if u.lstrip().startswith("|") and u.strip().strip("|").split("|")[0].strip() == figure.row
        ]
    anchor = compile_place(figure.where.pattern)
    return [u for u in paragraphs(text) if anchor.search(flatten(u))]


def _unit(figure: Figure) -> str:
    units = paragraph_of(_doc(figure.doc), figure)
    assert len(units) == 1, f"{figure.doc}: the place of {_id(figure)} is in {len(units)} paragraphs"
    return flatten(units[0])


def _marker_scope(figure: Figure) -> str:
    """Where a footnote marker counts for the figure: its own table cell, or its whole paragraph."""
    if isinstance(figure.where, Row):
        return " ".join(table_cells(_doc(figure.doc), figure.row, figure.where.column))
    return _unit(figure)


def _footnote(doc: str, marker: str) -> list[str]:
    """The footnote paragraph that ``marker`` opens in ``doc`` (empty when none does)."""
    notes = [flatten(u) for u in paragraphs(_doc(doc)) if u.startswith(marker + " ")]
    assert len(notes) <= 1, f"{doc}: footnote {marker} is defined {len(notes)} times"
    return notes


def footnotes_of(figure: Figure) -> list[str]:
    """The footnotes the figure's own cell or paragraph points at with an explicit marker."""
    return [
        note for marker in footnote_markers(_marker_scope(figure)) for note in _footnote(figure.doc, marker)
    ]


def _x(value: Decimal) -> str:
    return f"{half_up(value, 2)}×"


def _id(f: Figure) -> str:
    return f"{f.doc}:{Path(f.record).stem}:{f.value.name}:{f.where}"


@pytest.mark.parametrize("figure", FIGURES, ids=[_id(f) for f in FIGURES])
def test_published_bench_figure_equals_its_record(figure: Figure) -> None:
    """Bug caught: a doc figure that no longer equals the record it comes from, or that the doc no longer prints
    at its place."""
    expected = half_up(figure.value.of(_record(figure.record)), figure.digits)
    printed = figures_at(_doc(figure.doc), figure.where, figure.row)
    assert printed == [expected], (
        f"{figure.doc} ({figure.where}) prints {printed}; {figure.record} {figure.value.name} rounds to {expected}"
    )


# --- L26: the rounding rule --------------------------------------------------------------------------------------


def test_half_up_rounds_a_tie_away_from_zero() -> None:
    """Bug caught: half-even (banker's) rounding, which prints 1.00 for 1.005 and 0.12 for 0.125."""
    assert half_up(Decimal("1.005"), 2) == "1.01"
    assert half_up(Decimal("0.125"), 2) == "0.13"
    assert half_up(Decimal("2.5"), 0) == "3"


# --- L24: what each record leaves out, pinned against its JSON keys ---------------------------------------------

# What each record's docs must disclose: the provenance fields it lacks, by key presence (commit, dirty flag, date,
# MLX version, mflux version, hardware), plus "uncommitted" when its own ``git_dirty`` is true. Typed from the JSON
# keys, not computed: a record that gains or loses a field turns its row red (G1).
ALL_SIX = frozenset({"commit", "dirty", "date", "mlx", "mflux", "hardware"})
OLD_BENCH = frozenset({"commit", "dirty", "date", "mlx"})
GAPS: dict[str, frozenset[str]] = {
    Q11: frozenset({"date", "mlx", UNCOMMITTED}),
    K11: frozenset({"date", "mlx", UNCOMMITTED}),
    B4: frozenset({"date", "mlx"}),
    B9: frozenset({"date", "mlx"}),
    F18: frozenset(),
    Q10: OLD_BENCH,
    F10: OLD_BENCH,
    Z10: OLD_BENCH,
    B4_10: OLD_BENCH,
    B9_10: OLD_BENCH,
    B4_6: OLD_BENCH,
    B9_6: OLD_BENCH,
    F63: OLD_BENCH,
    Z7: OLD_BENCH,
    B4_41: ALL_SIX,
    VAL: frozenset({"commit", "dirty", "mlx", "mflux"}),
    D4: frozenset({"commit", "dirty"}),
    D9: frozenset({"commit", "dirty"}),
    RJ: frozenset({"dirty"}),
    SW_Q: ALL_SIX,
    SW_Z: ALL_SIX,
    SW_K: ALL_SIX,
}


def gaps_of(record: str) -> frozenset[str]:
    """What the record's JSON leaves out, plus ``UNCOMMITTED`` when it says its tree was dirty at run time."""
    data = _record(record)
    return frozenset(missing_provenance(data) | ({UNCOMMITTED} if dirty_at_run_time(data) else set()))


def test_missing_provenance_reads_each_field_by_presence() -> None:
    """Bug caught: a field read by truthiness (``git_dirty: false`` counted as missing), or from the wrong level."""
    assert missing_provenance({}) == {"commit", "dirty", "date", "mlx", "mflux", "hardware"}
    full = {
        "generated_at": "2026-11-16",
        "hardware": {
            "chip": "M1 Max",
            "git_commit": "e7cb468",
            "git_dirty": False,
            "mlx_version": "0.32.2",
            "mflux_version": "0.24.0",
        },
    }
    assert missing_provenance(full) == set()
    assert missing_provenance({"hardware": {"chip": "M1 Max", "mflux_version": "0.17.5"}}) == {
        "commit",
        "dirty",
        "date",
        "mlx",
    }


def test_missing_provenance_needs_the_field_in_every_variant() -> None:
    """Bug caught: in the comparison report, one variant's commit counted for all six, so a variant that lost
    ``git_sha_a`` and ``git_sha_b`` still passes as "records the commit of each run" (N8)."""
    both = {"git_sha_a": "aaa0000", "git_sha_b": "bbb1111"}
    record: dict[str, Any] = {
        "variants": {"one": {"provenance": both}, "two": {"provenance": {"git_sha_b": "ccc2222"}}}
    }
    assert "commit" not in missing_provenance(record)
    record["variants"]["two"] = {"provenance": {}}
    assert "commit" in missing_provenance(record)
    record["generated_at"] = "2026-11-16"
    assert "date" not in missing_provenance(record)


def test_gaps_table_lists_every_record() -> None:
    """Bug caught: a record quoted by a doc with no pinned gaps, so its gap check would demand nothing."""
    assert set(GAPS) == set(RECORDS)


@pytest.mark.parametrize("record", RECORDS)
def test_record_leaves_out_the_pinned_fields(record: str) -> None:
    """Bug caught: a record that gains or loses a provenance field while the docs still describe the old set (G1: an
    ``mlx_version`` added to the Krea record)."""
    assert gaps_of(record) == GAPS[record]


def test_unstated_gaps_needs_a_negated_wording() -> None:
    """Bug caught: a gap counted as stated from a bare word, so "names its commit and its dirty flag" (M03) or
    "names commit 040056a" passes as "no commit"; or a list form ("names no MLX or mflux version, hardware, commit or
    uncommitted-changes flag") that loses a field."""
    every = frozenset({"commit", "dirty", "date", "mlx", "mflux", "hardware", UNCOMMITTED})
    assert unstated_gaps("each record names its commit and its dirty flag", every) == every
    assert unstated_gaps("names commit 040056a and a clean tree", frozenset({"commit"})) == {"commit"}
    assert unstated_gaps("the no-gate run took the same commit", frozenset({"commit"})) == {"commit"}
    assert unstated_gaps("It has no commit or dirty field, no date and no MLX version.", every) == {
        "mflux",
        "hardware",
        UNCOMMITTED,
    }
    listed = "It names no MLX or mflux version, hardware, commit or uncommitted-changes flag; MLX 0.31.2 per uv.lock."
    assert unstated_gaps(listed, every) == {"date", UNCOMMITTED}
    assert unstated_gaps("neither record names a commit or a dirty flag", every) == every - {
        "commit",
        "dirty",
    }
    assert unstated_gaps(
        "records the commit of each run but not whether each working tree was clean", every
    ) == (every - {"dirty"})
    assert unstated_gaps("notes uncommitted changes in the working tree", every) == every - {UNCOMMITTED}
    assert unstated_gaps("Neither record has an MLX version", frozenset({"mlx"})) == set()
    assert unstated_gaps(
        "What the record leaves out: its date, 2026-09-05, is the file time", every
    ) == every - {"date"}
    flag = "none holds a commit or a flag saying whether the working tree had uncommitted changes"
    assert unstated_gaps(flag, frozenset({"commit", "dirty"})) == set()
    # A gap stated in one sentence does not carry into the next.
    assert unstated_gaps("It has no date field. The MLX version is 0.31.2.", frozenset({"date", "mlx"})) == {
        "mlx"
    }


def test_unstated_gaps_does_not_take_a_positive_claim_for_a_gap() -> None:
    """Bug caught: "notes no uncommitted changes" counted as saying the tree was dirty (N10); a "no" early in a list
    covering a later positive claim, so "has no date field (…), names commit …, and records MLX 0.31.2" passes as
    "no MLX version" (N6); or "whether … had uncommitted changes", which says nothing either way."""
    dirty = frozenset({UNCOMMITTED})
    assert unstated_gaps("notes no uncommitted changes in the working tree", dirty) == dirty
    assert unstated_gaps("the run had no uncommitted changes", dirty) == dirty
    assert unstated_gaps("No uncommitted changes were recorded.", dirty) == dirty
    assert unstated_gaps("it ran without uncommitted changes", dirty) == dirty
    assert unstated_gaps("there were not uncommitted changes", dirty) == dirty
    assert (
        unstated_gaps("none holds a flag saying whether the working tree had uncommitted changes", dirty)
        == dirty
    )
    assert unstated_gaps("the run had uncommitted changes", dirty) == set()
    n6 = (
        "It has no date field (the date comes from its chunk timestamps), names commit 040056a, notes uncommitted "
        "changes in the working tree without listing files, and records MLX 0.31.2; MLX 0.31.2 per the lock file."
    )
    assert unstated_gaps(n6, frozenset({"date", "mlx", UNCOMMITTED})) == {"mlx"}
    assert unstated_gaps(
        "The record has no date field and its MLX version is 0.32.2", frozenset({"mlx"})
    ) == {"mlx"}
    reversed_list = "it has no commit field and names its dirty flag, date and MLX version"
    assert unstated_gaps(reversed_list, frozenset({"commit", "dirty", "date", "mlx"})) == {
        "dirty",
        "date",
        "mlx",
    }
    # The README footnote shape still states every gap it lists.
    kept = (
        "It has no date field (the date comes from its chunk timestamps), names commit 040056a, notes uncommitted "
        "changes in the working tree without listing files, and has no MLX version field"
    )
    assert unstated_gaps(kept, frozenset({"date", "mlx", UNCOMMITTED})) == set()


def test_clauses_ends_a_clause_only_at_a_stop_before_white_space() -> None:
    """Bug caught: a split at every "." or ";" (cutting "MLX 0.31.2", "1.35×" or a URL mid-clause), or no split at
    ";" (keeping two records' clauses, as in the calibration note's list, in one)."""
    text = "They allow 1.35× (50/42); MLX 0.31.2 per [PR #42](https://github.com/a/b.c). The v0.6.0 record has no date."
    assert clauses(text) == [
        "They allow 1.35× (50/42);",
        "MLX 0.31.2 per [PR #42](https://github.com/a/b.c).",
        "The v0.6.0 record has no date.",
    ]


def test_span_from_runs_from_the_anchor_to_the_next_stop() -> None:
    """Bug caught: a span that runs past the next clause naming another record (so it borrows that record's gap
    wording), one cut short at its own anchor clause when that clause also names the other record, or one that
    starts before the anchor."""
    text = "A is fast. A has no date. B has no MLX version; it is slow. C too."
    assert span_from(text, "A has", ["B has"]) == "A has no date."
    assert span_from(text, "A is", ["B has"]) == "A is fast. A has no date."
    assert span_from(text, "B has", ["A is"]) == "B has no MLX version; it is slow. C too."
    assert span_from("A and B have no date. A is fast. B is slow.", "A and B", ["B"]) == (
        "A and B have no date. A is fast."
    )
    assert span_from(text, "D has", ["B has"]) is None


def test_span_from_rejects_an_anchor_in_two_clauses() -> None:
    """Bug caught: an anchor that matches two clauses silently picks the first, so a record's span can start at a
    sentence about another record."""
    with pytest.raises(ValueError, match="2 clauses"):
        span_from("A has x. A has y.", "A has", [])


def test_states_limit_needs_the_limit_beside_a_limit_word() -> None:
    """Bug caught: a limit counted as stated because the same "N.NN×" is printed as another quantity (a gating
    figure, another run's speedup), so deleting the limit clause stays green."""
    assert states_limit("The 13 skipped steps of 50 alone allow 1.35× (50/37).", "1.35×")
    assert states_limit("both are above the 1.35× that 13 skipped steps of 50 allow (50/37)", "1.35×")
    assert states_limit("both above the 1.35× limit, 50/37", "1.35×")
    assert states_limit("Ten skipped steps out of 28 can give at most about 1.56× (28/18).", "1.56×")
    assert not states_limit("**1.37× combined** = 1.35× gating × 1.02× compile-avoidance.", "1.35×")
    assert not states_limit("1.35× gating. Both are above the limit.", "1.35×")
    assert not states_limit("both above the 11.35× limit", "1.35×")
    assert not states_limit("the limit, 50/37, is 1.35× (mflux 0.17.5 measured 1.35×)", "1.36×")


# --- L23: each doc names each record it quotes, with that record's gaps in one paragraph --------------------------


@pytest.mark.parametrize(("doc", "record"), _pair_params())
def test_doc_names_the_bench_record_it_quotes(doc: str, record: str) -> None:
    """Bug caught: a doc quotes a bench figure but no longer says which committed record it comes from."""
    assert record in _doc(doc), f"{doc} quotes {record} but does not name it"


# Pairs whose gaps sit in a paragraph of their own after their table or section, found by this anchor (the path is
# in the paragraph that introduces the table).
GAP_PARAGRAPH: dict[tuple[str, str], str] = {
    (V_KREA, K11): "The record names commit 040056a",
    (V_KREA, SW_K): "What the record leaves out: its date, 2026-09-05",
    (V_QWEN, SW_Q): "What the record leaves out: its date, 2026-09-06",
    (V_Z, SW_Z): "What the record leaves out: its date, 2026-05-31",
    (CALIBRATION, SW_Q): "What the records leave out, for all four",
    (CALIBRATION, SW_Z): "What the records leave out, for all four",
    (CALIBRATION, SW_K): "What the records leave out, for all four",
}


# Pairs whose record is named in a paragraph or footnote that names another record with gaps too. There the record's
# gaps count only inside its own span: from the clause holding this anchor up to the next clause that names another
# such record by path or holds another record's anchor below. The anchor must be found in exactly one clause of the
# paragraphs naming the record. "Neither record …" and "Both records …" are one statement about two records, so
# both records share its anchor.
GAP_SPAN: dict[tuple[str, str], str] = {
    (README, B4): "The record names commit ec4d110",
    (README, B4_10): "The v0.10.0 record has",
    (README, B4_6): "The v0.6.0 record (mflux 0.17.5)",
    (README, B4_41): "The v0.4.1 record names",
    (README, B9): "The record names commit ec4d110",
    (README, B9_10): "The v0.10.0 record has",
    (README, B9_6): "The v0.6.0 record (mflux 0.17.5)",
    (README, Z10): "⁵ `z-image-base` under CFG",
    (README, Z7): "v0.7.0 reported 1.17× at this recipe",
    (README, F10): "History, the 1.57× run.",
    (README, F63): "The earlier 1.46× (v0.6.3",
    (README, K11): "The record is `_artifacts/v0.11.0_bench_krea_dev.json`",
    (README, D4): "the wrapper trims 17 percent off Base",
    (README, D9): "the wrapper trims 17 percent off Base",
    (CALIBRATION, B4): "`_artifacts/v0.12.0_bench_klein_base_4b.json` is a clean-tree run",
    (CALIBRATION, B4_10): "`_artifacts/v0.10.0_bench_klein_base_4b.json` has",
    (CALIBRATION, Q11): "`_artifacts/v0.11.0_bench_qwen_image.json` has",
    (CALIBRATION, Z10): "`_artifacts/v0.10.0_bench_z_image.json` has",
    (CALIBRATION, K11): "`_artifacts/v0.11.0_bench_krea_dev.json` has",
    **{
        (page, RJ): "The run report, `_artifacts/comparison/report.json`, records"
        for page in (CMP_KREA, CMP_B4, CMP_B9, CMP_QWEN, CMP_Z)
    },
    (CMP_DISTILLED, D4): "Both records name mflux 0.18.0 and MLX 0.31.2",
    (CMP_DISTILLED, D9): "Both records name mflux 0.18.0 and MLX 0.31.2",
    (M3, B4): "Both records are clean-tree runs at ec4d110",
    (M3, B9): "Both records are clean-tree runs at ec4d110",
    (V_B4, B4_10): "The v0.10.0 record (`_artifacts/v0.10.0_bench_klein_base_4b.json`",
    (V_B4, B4_6): "The v0.6.0 record (mflux 0.17.5)",
    (V_B4, B4_41): "The v0.4.1 record names",
    (V_B9, VAL): "It came from one run of each condition",
    (V_B9, B9_6): "The 1.36× at v0.6.0",
    (V_B9, B9_10): "The 1.36× at v0.6.0",
}

# Gaps a doc states for a GAP_SPAN pair only in a clause about another record, so they are still read at paragraph
# level. Each entry is a gap the record's own span does not state today; once the doc states it there, the entry
# must go (``test_paragraph_level_gaps_are_still_missing_from_their_span``).
PARAGRAPH_LEVEL_GAPS: dict[tuple[str, str], frozenset[str]] = {
    (README, B4): frozenset({"date"}),
    (README, B9): frozenset({"date"}),
    (README, B4_41): frozenset({"dirty"}),
    (V_B4, B4_41): frozenset({"dirty"}),
    (V_B9, B9_6): frozenset({"commit", "dirty"}),
    (V_B9, B9_10): frozenset({"commit", "dirty"}),
}

GAPPED_RECORDS: frozenset[str] = frozenset(r for r, gaps in GAPS.items() if gaps)


def _naming(text: str, record: str) -> tuple[list[str], list[str]]:
    """The paragraphs (footnotes included) of ``text`` that name ``record`` by its path: those naming no other record
    with gaps by path, and those naming one. Only paths count: a paragraph that names ``record`` and describes
    another record in words ("The v0.10.0 record has ...") lands in the first list and is read whole, so that other
    record's gap wording still counts for ``record``."""
    others = GAPPED_RECORDS - {record}
    named = [flatten(u) for u in paragraphs(text) if record in u]
    alone = [u for u in named if not any(o in u for o in others)]
    return alone, [u for u in named if u not in alone]


def _anchored(text: str, doc: str, record: str) -> tuple[str, str] | None:
    """The paragraph holding the record's GAP_SPAN anchor and the record's span in it (None for a pair without an
    anchor). Fails when the anchor is in no clause, or in more than one, of the paragraphs naming the record."""
    anchor = GAP_SPAN.get((doc, record))
    if anchor is None:
        return None
    stops = [o for o in GAPPED_RECORDS if o != record]
    stops += [a for (d, r), a in GAP_SPAN.items() if d == doc and r != record and a != anchor]
    _, shared = _naming(text, record)
    found = [(u, s) for u in shared if (s := span_from(u, anchor, stops)) is not None]
    assert len(found) == 1, (
        f"{doc}: {record}: anchor {anchor!r} is in {len(found)} paragraphs naming the record"
    )
    return found[0]


def own_span(text: str, doc: str, record: str) -> str | None:
    """The record's span under its GAP_SPAN anchor (None for a pair without one)."""
    anchored = _anchored(text, doc, record)
    return None if anchored is None else anchored[1]


def _gap_units(text: str, doc: str, record: str) -> list[str]:
    """Where ``record``'s gaps may be stated in ``text``: each paragraph naming it and no other record with gaps, its
    GAP_PARAGRAPH anchor paragraph, and its GAP_SPAN span."""
    units, _ = _naming(text, record)
    anchor = GAP_PARAGRAPH.get((doc, record))
    if anchor is not None:
        units += [flatten(u) for u in paragraphs(text) if anchor in flatten(u)]
    span = own_span(text, doc, record)
    return units if span is None else [*units, span]


def gaps_left_out(text: str, doc: str, record: str) -> set[str]:
    """The pinned gaps of ``record`` that ``text`` (read as ``doc``) does not state: empty when one unit states them
    all, else the shortfall of the unit that comes closest. A PARAGRAPH_LEVEL_GAPS field is read in the whole
    paragraph that holds the record's span."""
    carved = PARAGRAPH_LEVEL_GAPS.get((doc, record), frozenset())
    required = GAPS[record] - carved
    shortfalls = [unstated_gaps(u, required) for u in _gap_units(text, doc, record)]
    left_out = min(shortfalls, key=len, default=set(required))
    if not carved:
        return left_out
    anchored = _anchored(text, doc, record)
    assert anchored is not None, f"{doc}: {record}: PARAGRAPH_LEVEL_GAPS entry without a GAP_SPAN anchor"
    return left_out | unstated_gaps(anchored[0], carved)


def test_gap_span_pairs_are_pairs_the_docs_quote() -> None:
    """Bug caught: a GAP_SPAN or PARAGRAPH_LEVEL_GAPS entry left for a pair no doc quotes any more, so it checks
    nothing."""
    assert set(GAP_SPAN) <= set(DOC_RECORD_PAIRS)
    assert set(PARAGRAPH_LEVEL_GAPS) <= set(GAP_SPAN)


@pytest.mark.parametrize(("doc", "record"), sorted(PARAGRAPH_LEVEL_GAPS))
def test_paragraph_level_gaps_are_still_missing_from_their_span(doc: str, record: str) -> None:
    """Bug caught: a gap kept at paragraph level after the doc started stating it in the record's own span, so a
    later deletion of that statement is again covered by a neighbour's wording."""
    span = own_span(_doc(doc), doc, record)
    assert span is not None
    carved = PARAGRAPH_LEVEL_GAPS[(doc, record)]
    assert unstated_gaps(span, carved) == set(carved), (
        f"{doc}: {record}: its span now states some of {carved}"
    )


@pytest.mark.parametrize(("doc", "record"), _pair_params())
def test_doc_states_each_record_with_its_gaps(doc: str, record: str) -> None:
    """Bug caught: a doc that quotes a record but drops its gap sentence, or reverses it into a completeness claim
    ("neither record names a commit or a dirty flag" turned into "each record names its commit and its dirty flag",
    M03). One paragraph or footnote must name the record and give a negated wording for each pinned gap. A gap
    stated only in a figure's marked footnote is enough (M01, by design)."""
    left_out = gaps_left_out(_doc(doc), doc, record)
    assert not left_out, f"{doc}: {record}: no paragraph naming it states {sorted(left_out)}"


def _mutated(doc: str, old: str, new: str, line_start: str | None = None) -> str:
    """``doc``'s text with ``old`` replaced by ``new``: inside the one line that starts with ``line_start`` when it is
    given, else in the whole doc. ``old`` must occur there exactly once, so a stale mutant fails instead of
    leaving the text unchanged."""
    text = _doc(doc)
    if line_start is None:
        assert text.count(old) == 1, f"{doc}: {old!r} occurs {text.count(old)} times"
        return text.replace(old, new)
    lines = [line for line in text.splitlines() if line.startswith(line_start)]
    assert len(lines) == 1, f"{doc}: {len(lines)} lines start with {line_start!r}"
    assert lines[0].count(old) == 1, f"{doc}: {old!r} occurs {lines[0].count(old)} times in {line_start!r}"
    return text.replace(lines[0], lines[0].replace(old, new))


@dataclass(frozen=True)
class GapMutant:
    """One edit that removes or falsifies one record's gap wording where the paragraph names other records too."""

    name: str
    doc: str
    record: str
    field: str
    old: str
    new: str
    line_start: str | None = None


_FN2 = "² `flux2-klein-base-9b`"
_FN3 = "³ `flux2-klein-base-4b`"
_FN5 = "⁵ `z-image-base`"
_FLUX1_HISTORY = "History, the 1.57× run."
_OLD_GAPS = "no commit or dirty field, no date and no MLX version; its install-time version string names commit c9db998"
_GENERIC = "its provenance fields; its install-time version string names commit c9db998"

# Each mutant stays green under a paragraph-level check: another record named in the same footnote or paragraph
# states the same gap. Each must turn its own record red, on the field it removes.
GAP_MUTANTS: tuple[GapMutant, ...] = (
    GapMutant("readme-4b-mlx-deleted", README, B4, "mlx", " and has no MLX version field;", ";", _FN3),
    GapMutant(
        "readme-4b-mlx-reversed",
        README,
        B4,
        "mlx",
        "and has no MLX version field;",
        "and records MLX 0.31.2;",
        _FN3,
    ),
    GapMutant("readme-9b-mlx-deleted", README, B9, "mlx", " and has no MLX version field;", ";", _FN2),
    GapMutant("readme-z7-gaps-generic", README, Z7, "mlx", _OLD_GAPS, _GENERIC, _FN5),
    GapMutant("readme-f63-gaps-generic", README, F63, "mlx", _OLD_GAPS, _GENERIC, _FLUX1_HISTORY),
    GapMutant(
        "calibration-qwen-mlx-date-deleted",
        CALIBRATION,
        Q11,
        "mlx",
        ", no MLX version and\nno date field;",
        ";",
    ),
    GapMutant(
        "calibration-krea-uncommitted-deleted",
        CALIBRATION,
        K11,
        UNCOMMITTED,
        "krea_dev.json` has\nuncommitted changes, no date field",
        "krea_dev.json` has\nno date field",
    ),
    GapMutant(
        "calibration-krea-date-deleted",
        CALIBRATION,
        K11,
        "date",
        "uncommitted changes, no date field and no MLX version.",
        "uncommitted changes, no MLX version.",
    ),
)


@pytest.mark.parametrize("mutant", GAP_MUTANTS, ids=[m.name for m in GAP_MUTANTS])
def test_gap_check_does_not_borrow_a_neighbours_gap_wording(mutant: GapMutant) -> None:
    """Bug caught: a gap check that reads a whole paragraph, so where it names several records one record's
    "no MLX version" counts for another whose own wording was deleted or turned into "records MLX 0.31.2"."""
    assert mutant.field not in gaps_left_out(_doc(mutant.doc), mutant.doc, mutant.record)
    mutated = _mutated(mutant.doc, mutant.old, mutant.new, mutant.line_start)
    assert mutant.field in gaps_left_out(mutated, mutant.doc, mutant.record)


def test_a_span_ends_at_the_next_records_anchor_without_a_path() -> None:
    """Bug caught: a span that ends only at another record's path. In footnote ³ the next path after "The v0.10.0
    record has …" is the closing "Full report", so that span would run on into the v0.6.0 record's "no commit,
    dirty flag, date or MLX version" and borrow it."""
    old = "The v0.10.0 record has no commit or dirty field, no date and no MLX version;"
    mutated = _mutated(README, old, "The v0.10.0 record has its provenance fields;", _FN3)
    assert gaps_left_out(_doc(README), README, B4_10) == set()
    assert gaps_left_out(mutated, README, B4_10) == {"commit", "dirty", "date", "mlx"}


def test_a_reworded_span_anchor_fails_instead_of_falling_back() -> None:
    """Bug caught: a GAP_SPAN anchor the doc no longer holds, after which the record is read at paragraph level
    again (or not at all) and a rewording silently drops the per-record check."""
    mutated = _mutated(README, "The earlier 1.46× (v0.6.3", "The older 1.46× (v0.6.3", _FLUX1_HISTORY)
    with pytest.raises(AssertionError, match=r"anchor 'The earlier 1\.46× \(v0\.6\.3' is in 0 paragraphs"):
        gaps_left_out(mutated, README, F63)


def test_a_paragraph_level_gap_is_still_required() -> None:
    """Bug caught: a PARAGRAPH_LEVEL_GAPS field that is dropped from the check instead of read at paragraph level, so
    the 9B page could lose the only "no commit and no dirty flag" in the paragraph about the older records."""
    old = "the record names no mflux or MLX version, no commit and no dirty flag)"
    mutated = _mutated(V_B9, old, "the record names no mflux or MLX version)")
    assert gaps_left_out(_doc(V_B9), V_B9, B9_6) == set()
    assert gaps_left_out(mutated, V_B9, B9_6) == {"commit", "dirty"}


# A footnote's own citation of its record: "The record is `…`" or "Full report: `…`".
_CITATION = re.compile(r"(?:The record is|Full report:) \[?`(_artifacts/[\w./-]+\.json)`")

README_ROWS: tuple[tuple[str, str], ...] = tuple(
    sorted({(f.row, f.record) for f in FIGURES if f.doc == README and isinstance(f.where, Row)})
)


@pytest.mark.parametrize(("row", "record"), README_ROWS)
def test_readme_row_marker_leads_to_its_own_records_footnote(row: str, record: str) -> None:
    """Bug caught: a headline row whose marker sends the reader to another row's footnote (Z-Image ``1.31×⁵`` turned
    into ``1.31×⁶``, M06), or a footnote whose citation names another record (M04b)."""
    lines = [
        u
        for u in paragraphs(_doc(README))
        if u.lstrip().startswith("|") and u.strip("| ").startswith(row + " ")
    ]
    assert len(lines) == 1, f"README row {row!r} occurs {len(lines)} times"
    markers = footnote_markers(lines[0])
    assert markers, f"README row {row!r} has no footnote marker"
    for marker in markers:
        notes = _footnote(README, marker)
        assert notes, f"README row {row!r}: footnote {marker} is not defined"
        cited = _CITATION.findall(notes[0])
        assert cited, f"README footnote {marker} has no 'The record is' or 'Full report:' citation"
        assert set(cited) == {record}, f"README row {row!r} ({record}): footnote {marker} cites {cited}"


# --- Install-time version stamps the docs quote ------------------------------------------------------------------

# Each quoted record's ``mlx_teacache_version`` stamp: the commit its ``+g<sha>`` names (None when it names none), and
# whether its ``.d<date>`` suffix marks uncommitted changes at install time. Typed from the JSON.
STAMPS: dict[str, tuple[str | None, bool]] = {
    Q10: ("90fca33c9", False),
    F10: ("90fca33c9", False),
    Z10: ("90fca33c9", False),
    B4_10: ("90fca33c9", False),
    B9_10: ("90fca33c9", False),
    F63: ("c9db99832", True),
    Z7: ("c9db99832", True),
    B4_6: (None, False),
    B9_6: (None, False),
}


@pytest.mark.parametrize("record", list(STAMPS))
def test_install_stamp_is_the_pinned_one(record: str) -> None:
    """Bug caught: a record's stamp misread (the ``+g`` sha or the ``.d`` suffix), so the doc check below compares
    against the wrong commit."""
    stamp = install_stamp(_record(record))
    assert (stamp.sha, stamp.uncommitted) == STAMPS[record], stamp.version


def test_stamp_claims_reads_commit_and_install_mark() -> None:
    """Bug caught: a claim's sha cut short, or its "marks uncommitted changes" clause dropped or invented."""
    text = (
        "its install-time version string names commit 90fca33. Its version string names no commit; "
        "its install-time version string names commit c9db998 and marks uncommitted changes at install time)"
    )
    assert stamp_claims(text) == [("90fca33", False), (None, False), ("c9db998", True)]


# Docs that describe at least one install-time version string ("its install-time version string names commit …",
# "its version string names no commit"). Typed from the docs, not derived from `stamp_claims`.
STAMP_DOCS: tuple[str, ...] = (
    README,
    COMPARISON,
    CALIBRATION,
    CMP_DEV,
    CMP_Z,
    V_DEV,
    V_B4,
    V_B9,
    V_QWEN,
    V_Z,
)


@pytest.mark.parametrize("doc", STAMP_DOCS)
def test_each_stamp_doc_yields_a_claim(doc: str) -> None:
    """Bug caught: the docs' stamp wording changes (as "version stamp" became "version string") and the claim pattern
    no longer reads it, so the check below finds no claims and passes whatever the docs say."""
    assert doc in SCOPED_DOCS, f"{doc}: its stamp claims are never checked against its records"
    assert stamp_claims(flatten(_doc(doc))), f"{doc}: no install-time version string claim read"


@pytest.mark.parametrize("doc", SCOPED_DOCS)
def test_quoted_install_stamps_match_a_record_the_doc_quotes(doc: str) -> None:
    """Bug caught: a doc credits a record's install-time stamp with the wrong commit, or drops (or invents) the
    uncommitted-changes mark its ``.d<date>`` suffix carries. Each claim must match the pinned stamp of a record the
    doc quotes; which of them is not checked."""
    own = {STAMPS[r] for d, r in DOC_RECORD_PAIRS if d == doc and r in STAMPS}
    wrong = [
        (sha, marked)
        for sha, marked in stamp_claims(flatten(_doc(doc)))
        if not any(
            marked == pinned_marked
            and (sha is None if pinned_sha is None else sha is not None and pinned_sha.startswith(sha))
            for pinned_sha, pinned_marked in own
        )
    ]
    assert not wrong, (
        f"{doc}: stamp claims matching no quoted record: {wrong}; its records' stamps: {sorted(map(str, own))}"
    )


# --- L21: every above-limit combined speedup states its own limit -------------------------------------------------


def _limit_of(figure: Figure) -> Decimal | None:
    return None if figure.value.limit is None else figure.value.limit(_record(figure.record))


CEILING_CASES = tuple(
    f for f in FIGURES if (limit := _limit_of(f)) is not None and f.value.of(_record(f.record)) > limit
)


def test_above_limit_values_are_the_known_set() -> None:
    """Bug caught: a limit computed from the wrong fields (``computed_counts`` instead of N minus the skipped count),
    so the rule below checks the wrong values. Limits by hand: 28/18, 50/42, 50/36, 50/38, 50/41, 50/37, 25/19,
    50/17. The set is the multi-run combined speedups and the distilled page's base/base-teacache ratios; gating
    ratios and single-run speedups carry no limit."""
    above = {(Path(f.record).name, f.value.name, _x(_limit_of(f) or Decimal(0))) for f in CEILING_CASES}
    assert above == {
        ("v0.11.0_bench_krea_dev.json", "combined", "1.56×"),
        ("klein_base_vs_distilled_4b_q4.json", "base/base-teacache warm", "1.19×"),
        ("klein_base_vs_distilled_9b_q4_512x512.json", "base/base-teacache warm", "1.39×"),
        ("v0.12.0_bench_klein_base_4b.json", "combined", "1.19×"),
        ("v0.12.0_bench_klein_base_9b.json", "combined", "1.35×"),
        ("v0.10.0_bench_klein_base_9b.json", "combined", "1.35×"),
        ("v0.6.0_bench_klein_base_4b.json", "combined", "1.22×"),
        ("v0.6.0_bench_klein_base_9b.json", "combined", "1.35×"),
        ("_bench_report_klein_base_4b_v0_4_1.json", "combined", "1.22×"),
        ("v0.10.0_bench_flux1_dev.json", "combined", "1.32×"),
        ("v0.6.3_bench_flux1_dev.json", "combined", "1.32×"),
        ("v0.10.0_bench_qwen_image.json", "combined", "2.94×"),
    }


def _decomposed_here(figure: Figure, unit: str) -> bool:
    """True when ``unit`` holds this record's own gating and compile figures (FIGURES entries placed there)."""
    own = {
        f.value.name
        for f in FIGURES
        if f.doc == figure.doc
        and f.record == figure.record
        and f.value in (GATING, COMPILE)
        and isinstance(f.where, Near)
        and compile_place(f.where.pattern).search(unit)
    }
    return own == {GATING.name, COMPILE.name}


def _under_table(figure: Figure) -> list[str]:
    """When the figure sits in a table row, the paragraph right under its table (where a page states the row's limit);
    otherwise nothing."""
    units = paragraphs(_doc(figure.doc))
    own = paragraph_of(_doc(figure.doc), figure)
    rows = [i for i, u in enumerate(units) if u in own]
    assert len(rows) == 1, f"{figure.doc}: the place of {_id(figure)} is in {len(rows)} paragraphs"
    if not units[rows[0]].lstrip().startswith("|"):
        return []
    after = [u for u in units[rows[0] + 1 :] if not u.lstrip().startswith("|")]
    return [flatten(after[0])] if after else []


@pytest.mark.parametrize("figure", CEILING_CASES, ids=[_id(f) for f in CEILING_CASES])
def test_above_limit_value_is_quoted_with_its_own_limit(figure: Figure) -> None:
    """Bug caught: a speedup above N / (N - skipped) printed without that limit beside it (or in its marked
    footnote), which reads as if skipped steps alone gave it; or a ``decomposed`` exemption that passes on another
    record's gating and compile figures (R4)."""
    unit = _unit(figure)
    limit = _x(_limit_of(figure) or Decimal(0))
    places = [unit, *footnotes_of(figure), *_under_table(figure)]
    if figure.exempt == DECOMPOSED:
        assert any(_decomposed_here(figure, p) for p in places), (
            f"{figure.doc}: {_id(figure)} is exempt as decomposed, but no unit holds its own gating and compile"
        )
        return
    assert not figure.exempt, f"unknown exemption {figure.exempt!r}"
    assert any(states_limit(p, limit) for p in places), (
        f"{figure.doc}: {_x(figure.value.of(_record(figure.record)))} from {figure.record} is quoted without {limit} "
        "beside allow, limit or at most"
    )


# --- L22: no "×" figure in a doc is left unchecked ---------------------------------------------------------------

# Generated blocks (rewritten by docs/_generate_comparison.py) are left out: FIGURES does not read them.
_GENERATED = re.compile(r"<!-- (\S+?)[ _]?START -->.*?<!-- \1[ _]?END -->", re.DOTALL)
_TIMES = re.compile(r"(?<![\d.])(\d+\.\d+)×")

RESTATED = "a restatement of the pinned "

# "×" figures not pinned at their place: how many times each doc prints the value beyond its FIGURES places, and why.
UNPINNED_TIMES: dict[tuple[str, str], tuple[int, str]] = {
    (README, "1.3"): (
        2,
        "a prediction for untested chips (FLUX.2 likely 1.1–1.3× on M3 and M4, two chip rows)",
    ),
    (README, "1.0"): (1, "a prediction for untested chips (FLUX.2 may approach 1.0× on M5)"),
    (README, "2.68"): (
        5,
        "v0.5.0's withdrawn Klein base 9B figure, and Qwen-Image's headline (tables, the correction)",
    ),
    (README, "1.37"): (4, RESTATED + "Klein base 9B headline (tables, the correction, footnote ²)"),
    (README, "1.20"): (2, RESTATED + "Klein base 4B headline (tables, footnote ³)"),
    (README, "1.62"): (2, RESTATED + "Krea headline (tables, footnote ⁷)"),
    (README, "1.56"): (1, RESTATED + "Krea limit (headline table row)"),
    (README, "1.30"): (
        6,
        RESTATED + "FLUX.1-dev headline (tables, footnote ⁶, the mechanism paragraph)",
    ),
    (README, "1.32"): (3, RESTATED + "FLUX.1-dev limit (table cells, the mechanism paragraph)"),
    (README, "1.57"): (4, RESTATED + "FLUX.1-dev history figure and the Krea gate-on/gate-off ratio"),
    (README, "1.35"): (3, RESTATED + "Klein base 9B limit (footnote ²) and gating figure"),
    (README, "1.02"): (1, RESTATED + "Klein base 9B compile-avoidance factor (the 1.01× to 1.02× range)"),
    (README, "1.01"): (1, RESTATED + "Klein base 4B compile-avoidance factor (the 1.01× to 1.02× range)"),
    (README, "1.19"): (1, RESTATED + "Klein base 4B limit (footnote ³)"),
    (README, "1.22"): (1, RESTATED + "v0.6.0 Klein base 4B limit, 50/41 (the drift sentence)"),
    (COMPARISON, "1.20"): (1, RESTATED + "Klein base 4B headline (factors need not multiply)"),
    (COMPARISON, "1.19"): (1, RESTATED + "Klein base 4B limit"),
    (COMPARISON, "1.37"): (1, RESTATED + "Klein base 9B headline (factors need not multiply)"),
    (COMPARISON, "1.35"): (1, RESTATED + "Klein base 9B limit (footnote ²) and gating figure"),
    (CMP_B4, "1.19"): (1, RESTATED + "Klein base 4B limit"),
    (CMP_B9, "1.35"): (1, RESTATED + "Klein base 9B limit"),
    (CMP_DISTILLED, "1.00"): (2, "the base row's ratio to itself, in each of the two tables"),
    (CMP_DISTILLED, "1.20"): (1, RESTATED + "4B base/base-teacache ratio under its table"),
    (CMP_DISTILLED, "1.42"): (1, RESTATED + "9B base/base-teacache ratio under its table"),
    (M3, "1.0"): (1, "a prediction for untested chips (the speedup approaches 1.0×)"),
    (V_DEV, "1.30"): (3, RESTATED + "FLUX.1-dev headline"),
    (V_KREA, "3.12"): (1, RESTATED + "sweep row at 0.80, quoted as an above-the-limit example"),
    (V_B4, "1.01"): (1, RESTATED + "compile-avoidance factor"),
    (V_B4, "1.22"): (1, RESTATED + "v0.10.0 figure, as the top of the 1.18–1.22× gating range"),
    (V_B4, "1.26"): (1, RESTATED + "v0.4.1 figure (the note on the v0.4.1 claim)"),
    (V_B4, "1.09"): (1, "v0.4.1's withdrawn compile-avoidance factor, restated"),
    (V_B9, "1.37"): (1, RESTATED + "headline"),
    (V_B9, "1.35"): (1, RESTATED + "gating figure and limit"),
    (V_B9, "2.68"): (1, "v0.5.0's withdrawn figure, restated"),
    (V_B9, "0.85"): (1, "a memory cap factor (0.85× the recommended working set), not a speedup"),
    (V_QWEN, "2.68"): (1, RESTATED + "headline (the section intro)"),
    (V_Z, "1.31"): (1, RESTATED + "headline (gating equals combined)"),
}


def _swept_text(doc: str) -> str:
    return _GENERATED.sub("", _doc(doc))


def _pinned_times(doc: str) -> Counter[str]:
    """The values FIGURES pins in ``doc`` at a place where "×" follows the figure, counted once per place."""
    text = _swept_text(doc)
    flat = flatten(text)
    places: dict[tuple[object, ...], str] = {}
    for f in FIGURES:
        if f.doc != doc:
            continue
        if isinstance(f.where, Near):
            for m in compile_place(f.where.pattern).finditer(flat):
                if flat.startswith("×", m.end(1)):
                    places["near", m.start(1)] = m.group(1)
        elif f.where.pattern is not None:
            for i, cell in enumerate(table_cells(text, f.row, f.where.column)):
                for m in compile_place(f.where.pattern).finditer(cell):
                    if cell.startswith("×", m.end(1)):
                        places["cell", f.row, f.where.column, i, m.start(1)] = m.group(1)
    return Counter(places.values())


def unpinned_times(doc: str) -> Counter[str]:
    """How many ``N.NN×`` tokens of each value ``doc`` prints, outside generated blocks, beyond the FIGURES places
    that pin that value there."""
    return Counter(_TIMES.findall(_swept_text(doc))) - _pinned_times(doc)


@pytest.mark.parametrize("doc", SCOPED_DOCS)
def test_every_times_figure_is_pinned_or_listed(doc: str) -> None:
    """Bug caught: a doc gains a speedup that no record backs ("Qwen-Image 2512 runs 3.10× faster", M08), even when
    it reuses a value pinned elsewhere in the doc (1.62×, N2) or an allowlisted one (a second 1.3×, N1)."""
    allowed = {token: n for (d, token), (n, _) in UNPINNED_TIMES.items() if d == doc}
    extra = {token: n for token, n in unpinned_times(doc).items() if n > allowed.get(token, 0)}
    assert not extra, f"{doc}: N.NN× tokens beyond their FIGURES places and UNPINNED_TIMES counts: {extra}"


@pytest.mark.parametrize(("doc", "token"), list(UNPINNED_TIMES))
def test_unpinned_times_entry_is_still_printed(doc: str, token: str) -> None:
    """Bug caught: an allowlist entry, or its count, left behind after its figure went, which would excuse a new
    figure later."""
    count, _ = UNPINNED_TIMES[doc, token]
    assert unpinned_times(doc)[token] == count, (
        f"{doc} prints {token}× unpinned {unpinned_times(doc)[token]} times"
    )
