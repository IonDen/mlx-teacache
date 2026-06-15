"""Qwen-Image CFG two-call branch-pairing. Pure-core: no mflux, no weights.

Qwen calls the transformer twice per denoising step (positive then negative).
CfgBranchPairer threads one shared gate decision + branch parity across the two
calls and resets on a fresh generation (lifecycle gen-ctx token change).
"""

from mlx_teacache.variants.qwen_image.pairing import CfgBranchPairer


def test_fresh_token_starts_positive() -> None:
    p = CfgBranchPairer()
    p.advance()  # move off the default (branch_idx -> 1) so the reset is non-trivial
    assert not p.is_positive()
    p.on_generation_token(1)  # a fresh token must reset to positive
    assert p.is_positive()


def test_alternates_within_one_generation() -> None:
    p = CfgBranchPairer()
    p.on_generation_token(1)
    assert p.is_positive()        # step 0 positive
    p.advance()
    p.on_generation_token(1)      # same token → no reset
    assert not p.is_positive()    # step 0 negative
    p.advance()
    p.on_generation_token(1)
    assert p.is_positive()        # step 1 positive


def test_new_token_resets_after_midpair_interrupt() -> None:
    p = CfgBranchPairer()
    p.on_generation_token(1)
    p.advance()                   # interrupt lands here: positive done, negative pending
    p.on_generation_token(2)      # fresh generation
    assert p.is_positive()        # reset, not stuck on negative
    assert p.shared_decision is None


def test_shared_decision_survives_to_negative_call() -> None:
    p = CfgBranchPairer()
    p.on_generation_token(1)
    p.shared_decision = "DECISION"
    p.advance()
    p.on_generation_token(1)
    assert p.shared_decision == "DECISION"


def test_new_token_clears_stale_shared_decision() -> None:
    p = CfgBranchPairer()
    p.on_generation_token(1)
    p.shared_decision = "OLD"
    p.on_generation_token(2)
    assert p.shared_decision is None


def test_registry_discovers_qwen_image() -> None:
    from mlx_teacache.variants import _REGISTRY

    assert "qwen-image" in _REGISTRY
    assert _REGISTRY["qwen-image"]["META"]["variant_id"] == "qwen-image"
