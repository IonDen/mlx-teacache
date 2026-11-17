"""Deprecation of TeaCacheStats.cfg_fallback_steps and TeaCacheStats.total_steps_seen.

Both names keep working until 1.0. Every public use warns at the caller's line;
internal bookkeeping and the dataclass machinery (construction without the
keyword, repr, equality, replace, copy, pickling) never warn.
"""

import copy
import dataclasses
import pickle
import warnings
from collections.abc import Callable

import pytest

from mlx_teacache._kernel.stats import StepDecision, TeaCacheStats


def _decision(idx: int, kind: str) -> StepDecision:
    return StepDecision(
        step_idx=idx,
        timestep=float(idx),
        rel_l1=0.1,
        accumulated_distance=0.0,
        decision=kind,
    )


def _stats_with_fallback_steps(n: int) -> TeaCacheStats:
    """A stats object whose deprecated counter is n, built through the public constructor."""
    with pytest.warns(DeprecationWarning):
        return TeaCacheStats(generations=1, computed_count=2, cfg_fallback_steps=n)


def test_reading_cfg_fallback_steps_warns_at_the_caller() -> None:
    s = TeaCacheStats()
    with pytest.warns(DeprecationWarning, match=r"GenerationStats\.cfg_was_active") as rec:
        value = s.cfg_fallback_steps
    assert value == 0
    assert len(rec) == 1
    assert rec[0].filename == __file__
    assert "removed in 1.0" in str(rec[0].message)


def test_writing_cfg_fallback_steps_warns_at_the_caller() -> None:
    s = TeaCacheStats()
    with pytest.warns(DeprecationWarning, match=r"GenerationStats\.cfg_was_active") as rec:
        s.cfg_fallback_steps = 4
    assert len(rec) == 1
    assert rec[0].filename == __file__
    with pytest.warns(DeprecationWarning):
        assert s.cfg_fallback_steps == 4
    with pytest.warns(DeprecationWarning):
        assert s.total_steps_seen == 4


def test_writing_cfg_fallback_steps_under_an_error_filter_leaves_the_old_value() -> None:
    s = _stats_with_fallback_steps(2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(DeprecationWarning):
            s.cfg_fallback_steps = 9
    with pytest.warns(DeprecationWarning):
        assert s.cfg_fallback_steps == 2


def test_total_steps_seen_warns_exactly_once_and_equals_total_active_steps() -> None:
    s = TeaCacheStats()
    s.record(_decision(0, "computed"))
    s.record(_decision(1, "computed"))
    s.finalize_last_generation(num_inference_steps=2, cfg_was_active=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        value = s.total_steps_seen
    assert value == 2
    assert len(rec) == 1
    assert issubclass(rec[0].category, DeprecationWarning)
    assert rec[0].filename == __file__
    assert "total_active_steps" in str(rec[0].message)
    assert "removed in 1.0" in str(rec[0].message)


def test_total_steps_seen_still_counts_fallback_steps() -> None:
    s = _stats_with_fallback_steps(3)
    with pytest.warns(DeprecationWarning):
        assert s.total_steps_seen == 2 + 3


def test_recording_and_finalizing_never_warn() -> None:
    s = TeaCacheStats()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s.record(_decision(0, "cfg-fallback"))
        s.finalize_last_generation(num_inference_steps=1, cfg_was_active=True)
        s.record(_decision(0, "computed"))
        s.discard_current_generation()
        assert s.total_active_steps == 0
    with pytest.warns(DeprecationWarning):
        assert s.cfg_fallback_steps == 1


def test_constructor_keyword_cfg_fallback_steps_warns_at_the_caller_and_is_kept() -> None:
    with pytest.warns(DeprecationWarning, match=r"GenerationStats\.cfg_was_active") as rec:
        s = TeaCacheStats(cfg_fallback_steps=3)
    assert len(rec) == 1
    assert rec[0].filename == __file__
    assert "removed in 1.0" in str(rec[0].message)
    with pytest.warns(DeprecationWarning):
        assert s.cfg_fallback_steps == 3


def test_positional_arguments_keep_their_order() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s = TeaCacheStats(1, 2, 3, 4, 5)
    assert (s.generations, s.computed_count, s.forced_count, s.skipped_count, s.numerical_miss_count) == (
        1,
        2,
        3,
        4,
        5,
    )
    with pytest.warns(DeprecationWarning) as rec:
        s = TeaCacheStats(0, 0, 0, 0, 0, 6)
    assert rec[0].filename == __file__
    with pytest.warns(DeprecationWarning):
        assert s.cfg_fallback_steps == 6


_SILENT_OPERATIONS: dict[str, Callable[[TeaCacheStats], object]] = {
    "construct-without-keyword": lambda s: TeaCacheStats(),
    "repr": repr,
    "eq": lambda s: s == TeaCacheStats(),
    "replace": dataclasses.replace,
    "replace-with-change": lambda s: dataclasses.replace(s, skipped_count=1),
    "copy": copy.copy,
    "deepcopy": copy.deepcopy,
    "pickle-round-trip": lambda s: pickle.loads(pickle.dumps(s)),
    "asdict": dataclasses.asdict,
}


@pytest.mark.parametrize("operation", _SILENT_OPERATIONS.values(), ids=_SILENT_OPERATIONS.keys())
def test_dataclass_machinery_does_not_warn(operation: Callable[[TeaCacheStats], object]) -> None:
    s = _stats_with_fallback_steps(1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        operation(s)


_COPIES: dict[str, Callable[[TeaCacheStats], TeaCacheStats]] = {
    "replace": dataclasses.replace,
    "copy": copy.copy,
    "deepcopy": copy.deepcopy,
    "pickle-round-trip": lambda s: pickle.loads(pickle.dumps(s)),
}


@pytest.mark.parametrize("make_copy", _COPIES.values(), ids=_COPIES.keys())
def test_copies_keep_the_fallback_count(make_copy: Callable[[TeaCacheStats], TeaCacheStats]) -> None:
    s = _stats_with_fallback_steps(5)
    clone = make_copy(s)
    assert clone == s
    with pytest.warns(DeprecationWarning):
        assert clone.cfg_fallback_steps == 5


def test_equality_still_tells_fallback_counts_apart() -> None:
    assert _stats_with_fallback_steps(1) != _stats_with_fallback_steps(2)
