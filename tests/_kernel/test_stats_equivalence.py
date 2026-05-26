"""Phase A discipline: _kernel.stats preserves v0.5.x stats contract."""

from __future__ import annotations


def test_step_decision_fields_match_v05():
    import dataclasses

    from mlx_teacache._kernel.stats import StepDecision

    actual = {f.name for f in dataclasses.fields(StepDecision)}
    assert actual == {"step_idx", "timestep", "rel_l1", "accumulated_distance", "decision"}


def test_generation_stats_fields_match_v05():
    import dataclasses

    from mlx_teacache._kernel.stats import GenerationStats

    actual = {f.name for f in dataclasses.fields(GenerationStats)}
    assert actual == {"num_steps", "cfg_was_active", "decisions"}


def test_decision_literal_includes_cfg_fallback():
    """v0.4.1 deprecated cfg-fallback but kept the Literal value; v0.6.0 preserves it."""
    import typing

    from mlx_teacache._kernel.stats import Decision

    assert "cfg-fallback" in typing.get_args(Decision)


def test_teacachestats_public_counter_fields():
    from mlx_teacache._kernel.stats import TeaCacheStats

    s = TeaCacheStats()
    assert s.generations == 0
    assert s.computed_count == 0
    assert s.forced_count == 0
    assert s.skipped_count == 0
    assert s.numerical_miss_count == 0
    assert s.cfg_fallback_steps == 0
    assert s.last_generation is None
    assert s.speedup_estimate == 1.0


def test_failed_generation_leaves_no_public_trace():
    """The commit/discard contract: record() touches staging only;
    discard_current_generation() drops staging; public counters
    are unchanged."""
    from mlx_teacache._kernel.stats import StepDecision, TeaCacheStats

    s = TeaCacheStats()
    s.record(
        StepDecision(
            step_idx=0,
            timestep=1.0,
            rel_l1=None,
            accumulated_distance=0.0,
            decision="computed",
        )
    )
    s.discard_current_generation()
    assert s.computed_count == 0
    assert s.generations == 0
    assert s.last_generation is None


def test_finalize_commits_to_public_counters():
    from mlx_teacache._kernel.stats import StepDecision, TeaCacheStats

    s = TeaCacheStats()
    for i in range(4):
        s.record(
            StepDecision(
                step_idx=i,
                timestep=float(i),
                rel_l1=None,
                accumulated_distance=0.0,
                decision="computed",
            )
        )
    s.finalize_last_generation(num_inference_steps=4, cfg_was_active=False)
    assert s.computed_count == 4
    assert s.generations == 1
    assert s.last_generation is not None
    assert s.last_generation.num_steps == 4
    assert s.last_generation.cfg_was_active is False


def test_shim_re_exports_identity():
    from mlx_teacache._kernel.stats import TeaCacheStats as KS
    from mlx_teacache.stats import TeaCacheStats as LS

    assert LS is KS
    from mlx_teacache._kernel.stats import StatsFrozenError as KSE
    from mlx_teacache.stats import StatsFrozenError as LSE

    assert LSE is KSE


def test_speedup_estimate_returns_one_when_all_steps_skipped():
    """denom <= 0 branch in speedup_estimate (stats.py:112-113): if every
    active step was skipped, division would blow up — fall back to 1.0."""
    from mlx_teacache._kernel.stats import TeaCacheStats

    s = TeaCacheStats(skipped_count=5)
    # total_active_steps == skipped_count, so denom = 0
    assert s.total_active_steps == 5
    assert s.speedup_estimate == 1.0


def test_finalize_last_generation_on_frozen_stats_raises():
    """frozen-stats guard on finalize_last_generation (stats.py:141)."""
    import pytest

    from mlx_teacache._kernel.stats import StatsFrozenError, TeaCacheStats

    s = TeaCacheStats()
    s._frozen = True
    with pytest.raises(StatsFrozenError):
        s.finalize_last_generation(num_inference_steps=4, cfg_was_active=False)
