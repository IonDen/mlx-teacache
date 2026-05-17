import pytest

from mlx_teacache.stats import (
    StatsFrozenError,
    StepDecision,
    TeaCacheStats,
    _Staging,
)


def _make_decision(idx: int, kind: str = "computed", rel_l1: float | None = 0.1) -> StepDecision:
    return StepDecision(
        step_idx=idx,
        timestep=float(idx),
        rel_l1=rel_l1,
        accumulated_distance=0.0,
        decision=kind,
    )


def test_fresh_stats_zero():
    s = TeaCacheStats()
    assert s.generations == 0
    assert s.computed_count == 0
    assert s.forced_count == 0
    assert s.skipped_count == 0
    assert s.numerical_miss_count == 0
    assert s.cfg_fallback_steps == 0
    assert s.last_generation is None
    assert s.total_active_steps == 0
    assert s.total_steps_seen == 0


def test_record_mutates_staging_not_public():
    s = TeaCacheStats()
    s.record(_make_decision(0, "computed"))
    s.record(_make_decision(1, "skipped"))
    assert s.computed_count == 0  # public counters unchanged
    assert s.skipped_count == 0
    assert s.last_generation is None


def test_finalize_commits_to_public_and_snapshots():
    s = TeaCacheStats()
    s.record(_make_decision(0, "computed"))
    s.record(_make_decision(1, "skipped"))
    s.record(_make_decision(2, "forced"))
    s.finalize_last_generation(num_inference_steps=3, cfg_was_active=False)
    assert s.generations == 1
    assert s.computed_count == 1
    assert s.skipped_count == 1
    assert s.forced_count == 1
    assert s.last_generation is not None
    assert s.last_generation.num_steps == 3
    assert s.last_generation.cfg_was_active is False
    assert len(s.last_generation.decisions) == 3


def test_discard_clears_staging_no_public_change():
    s = TeaCacheStats()
    s.record(_make_decision(0, "computed"))
    s.discard_current_generation()
    assert s.computed_count == 0
    assert s.generations == 0
    assert s.last_generation is None


def test_finalize_after_discard_starts_fresh():
    s = TeaCacheStats()
    s.record(_make_decision(0, "computed"))
    s.discard_current_generation()
    s.record(_make_decision(0, "skipped"))
    s.finalize_last_generation(num_inference_steps=1, cfg_was_active=False)
    assert s.computed_count == 0
    assert s.skipped_count == 1


def test_speedup_estimate_returns_1_when_no_active_steps():
    s = TeaCacheStats()
    assert s.speedup_estimate == 1.0


def test_speedup_estimate_active_step_math():
    s = TeaCacheStats()
    for _ in range(15):
        s.record(_make_decision(0, "computed"))
    for _ in range(10):
        s.record(_make_decision(0, "skipped"))
    s.finalize_last_generation(num_inference_steps=25, cfg_was_active=False)
    # 25 active steps, 10 skipped ⇒ 25/15 ≈ 1.666
    assert abs(s.speedup_estimate - 25 / 15) < 1e-6


def test_speedup_estimate_excludes_cfg_fallback():
    s = TeaCacheStats()
    for _ in range(10):
        s.record(_make_decision(0, "cfg-fallback"))
    s.finalize_last_generation(num_inference_steps=10, cfg_was_active=True)
    # All cfg-fallback ⇒ active_steps == 0 ⇒ speedup is 1.0
    assert s.speedup_estimate == 1.0
    assert s.cfg_fallback_steps == 10
    assert s.total_steps_seen == 10
    assert s.total_active_steps == 0


def test_counter_sum_invariant():
    s = TeaCacheStats()
    for kind in ["computed", "computed", "forced", "skipped", "numerical-miss"]:
        s.record(_make_decision(0, kind))
    for _ in range(3):
        s.record(_make_decision(0, "cfg-fallback"))
    s.finalize_last_generation(num_inference_steps=8, cfg_was_active=True)
    assert s.total_active_steps == 5
    assert s.cfg_fallback_steps == 3
    assert s.total_steps_seen == 8


def test_freeze_blocks_further_mutation():
    s = TeaCacheStats()
    s.record(_make_decision(0, "computed"))
    s.finalize_last_generation(num_inference_steps=1, cfg_was_active=False)
    s._freeze()
    with pytest.raises(StatsFrozenError):
        s.record(_make_decision(0, "computed"))
    with pytest.raises(StatsFrozenError):
        s.discard_current_generation()


def test_finalize_length_mismatch_raises_and_discards_staging():
    """Per spec: len(decisions) == num_inference_steps. Mismatch must raise
    InternalStateError AND discard staging (no partial commit)."""
    from mlx_teacache.errors import InternalStateError

    s = TeaCacheStats()
    s.record(_make_decision(0, "computed"))
    s.record(_make_decision(1, "skipped"))
    # Only 2 records but claim num_inference_steps=5 — mismatch
    with pytest.raises(InternalStateError, match="length invariant"):
        s.finalize_last_generation(num_inference_steps=5, cfg_was_active=False)
    # Public counters unchanged
    assert s.computed_count == 0
    assert s.skipped_count == 0
    assert s.generations == 0
    assert s.last_generation is None
    # Staging discarded — fresh record starts a new generation
    s.record(_make_decision(0, "computed"))
    s.finalize_last_generation(num_inference_steps=1, cfg_was_active=False)
    assert s.computed_count == 1


def test_staging_cfg_was_active_defaults_false():
    st = _Staging()
    assert st.cfg_was_active is False


def test_staging_cfg_was_active_clears_on_clear():
    st = _Staging()
    st.cfg_was_active = True
    st.clear()
    assert st.cfg_was_active is False


def test_finalize_records_cfg_was_active_from_staging():
    """finalize_last_generation must propagate _staging.cfg_was_active to
    GenerationStats.cfg_was_active. Replaces the v0.4.0 derivation from
    cfg_fallback_steps > 0 which is no longer correct in v0.4.1+."""
    stats = TeaCacheStats()
    stats._staging.cfg_was_active = True
    stats.record(
        StepDecision(step_idx=0, timestep=1.0, rel_l1=None, accumulated_distance=0.0, decision="computed")
    )
    stats.finalize_last_generation(num_inference_steps=1, cfg_was_active=True)
    assert stats.last_generation is not None
    assert stats.last_generation.cfg_was_active is True
    # cfg_fallback_steps stays at 0 in v0.4.1+ — feature is gone.
    assert stats.cfg_fallback_steps == 0
