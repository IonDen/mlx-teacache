"""The before/after equivalence oracle (tests/_equivalence_oracle.py) must see every variant, catch a one-ulp change
and record the same thing twice in one process; otherwise an A/B run that prints "identical" proves nothing."""

import copy
import json
from pathlib import Path

import pytest

from mlx_teacache.variants import _REGISTRY
from tests import _equivalence_oracle as oracle


@pytest.fixture(scope="module")
def recorded() -> dict:
    return oracle.collect()


def test_threshold_matrix_covers_every_registered_variant(recorded: dict) -> None:
    """Bug: a new variant escapes the oracle, so a refactor that breaks it still compares identical."""
    assert set(recorded["threshold_matrix"]) == set(_REGISTRY)


def test_compare_reports_a_one_ulp_change(recorded: dict) -> None:
    """Bug: compare ignores a field or rounds floats, so a one-ulp drift in a resolved threshold passes."""
    changed = copy.deepcopy(recorded)
    cell = changed["threshold_matrix"]["flux1-dev"]["thresh=0.33,coeffs=builtin"]["handle"]
    assert cell["rel_l1_thresh"] == (0.33).hex()  # 0x1.51eb851eb851fp-2
    mantissa, exponent = cell["rel_l1_thresh"].split("p")
    cell["rel_l1_thresh"] = mantissa[:-1] + ("e" if mantissa[-1] == "f" else "f") + "p" + exponent
    assert oracle.compare(recorded, changed) == [
        "threshold_matrix/flux1-dev/thresh=0.33,coeffs=builtin/handle/rel_l1_thresh"
    ]


def test_compare_ignores_where_and_when_but_not_which_harness(recorded: dict) -> None:
    """Bug: compare flags the library path (so the detached-worktree fallback always differs) or skips the harness
    hash (so two different harnesses compare identical)."""
    moved = copy.deepcopy(recorded)
    moved["meta"]["module_file"] = "/elsewhere/mlx_teacache/__init__.py"
    moved["meta"]["recorded_at"] = "2000-01-01T00:00:00+00:00"
    assert oracle.compare(recorded, moved) == []
    moved["meta"]["harness_sha256"] = "0" * 64
    assert oracle.compare(recorded, moved) == ["meta/harness_sha256"]


def test_compare_cli_exits_1_and_prints_the_differing_path(
    recorded: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bug: the compare command exits 0 on a difference, so the gate never stops a task."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(recorded))
    changed = copy.deepcopy(recorded)
    changed["threshold_matrix"]["qwen-image"]["thresh=None,coeffs=builtin"]["teardown"]["vars"].append(
        "_leak"
    )
    b.write_text(json.dumps(changed))
    assert oracle.main(["compare", str(a), str(a)]) == 0
    assert "identical (" in capsys.readouterr().out
    assert oracle.main(["compare", str(a), str(b)]) == 1
    assert "threshold_matrix/qwen-image/thresh=None,coeffs=builtin/teardown/vars" in capsys.readouterr().out


def test_collect_is_repeatable_in_one_process(recorded: dict) -> None:
    """Bug: state leaks between scenarios (a shared pairer, an unreset cache, a once-per-process warning), so the
    oracle compares noise."""
    again = oracle.collect()
    assert oracle.compare(recorded, again) == []
    assert again == recorded


_STRATEGIES = ("proxy-flux1", "predict-flux2", "predict-z-image", "proxy-qwen")


def _decisions(scenario: dict) -> list[str]:
    return [
        step[4]
        for run in scenario["runs"]
        if run["last_generation"] is not None
        for step in run["last_generation"]["decisions"]
    ]


def test_every_strategy_has_a_skipping_scenario(recorded: dict) -> None:
    """Bug: the oracle never reaches a skipped step on some patch strategy, so a broken residual reconstruction
    there compares equal."""
    skipping = {s["strategy"] for s in recorded["generations"].values() if "skipped" in _decisions(s)}
    assert skipping == set(_STRATEGIES)


@pytest.mark.parametrize(
    "name",
    [
        "flux2-klein-base-4b/cfg/const-coeffs-two-runs",
        "flux2-klein-base-9b/cfg/const-coeffs-two-runs",
        "z-image-base/cfg/const-coeffs-two-runs",
        "qwen-image/const-coeffs-two-runs",
        "qwen-image-2512/const-coeffs-two-runs",
    ],
)
def test_cfg_scenarios_skip_on_both_branches(recorded: dict, name: str) -> None:
    """Bug: the CFG scenarios never skip (or never run CFG), so a negative-branch residual that is never read, or
    read from the positive branch, compares equal."""
    scenario = recorded["generations"][name]
    assert "skipped" in _decisions(scenario)
    assert all(run["last_generation"]["cfg_was_active"] is True for run in scenario["runs"])
    # Qwen calls the transformer once per branch: both outputs of each step are in the record.
    if scenario["strategy"] == "proxy-qwen":
        assert all(len(step) == 2 for run in scenario["runs"] for step in run["steps"])


@pytest.mark.parametrize(
    ("name", "error_class"),
    [
        ("flux1-dev/error/window-refusal", "InvalidStepWindowError"),
        ("flux2-klein-base-4b/error/window-refusal", "InvalidStepWindowError"),
        ("z-image-base/error/window-refusal", "InvalidStepWindowError"),
        ("qwen-image/error/window-refusal", "InvalidStepWindowError"),
        ("flux1-dev/error/shape-change", "TransformerShapeError"),
        ("flux2-klein-base-4b/error/shape-change", "TransformerShapeError"),
        ("z-image-base/error/shape-change", "TransformerShapeError"),
        ("qwen-image/error/shape-change", "TransformerShapeError"),
        ("flux2-klein-base-4b/error/missing-context", "MissingGenerationContextError"),
        ("z-image-base/error/missing-context", "MissingGenerationContextError"),
    ],
)
def test_error_scenarios_record_the_exception(recorded: dict, name: str, error_class: str) -> None:
    """Bug: the harness swallows an error path (or records the wrong one), so a refactor that stops raising, or
    raises something else, compares equal."""
    run = recorded["generations"][name]["runs"][0]
    assert run["error"][0] == error_class
    assert run["error"][1]
    assert run["final_latents"] is None
    assert run["last_generation"] is None


@pytest.mark.parametrize("variant_id", ["flux1-dev", "flux2-klein-base-4b", "z-image-base", "qwen-image"])
def test_a_failed_generation_is_followed_by_a_clean_one(recorded: dict, variant_id: str) -> None:
    """Bug: the raise-mid-loop scenario is not run to the failure and then cleanly, so state carried over from a
    failed generation is never compared."""
    first, second = recorded["generations"][f"{variant_id}/error/raise-at-step-5-then-clean"]["runs"]
    assert first["error"] == ["RuntimeError", "generation failed on purpose at step 5"]
    assert len(first["steps"]) == 6
    assert second["error"] is None
    assert second["last_generation"]["num_steps"] == 10
    assert second["counters"]["generations"] == 1


# ---------- the harness can see what it is meant to see ----------


def test_the_harness_hash_covers_the_fakes_it_builds_models_from(tmp_path: Path) -> None:
    """Bug: the hash covers only the oracle file, so a record made with an edited `_variant_fakes.py` or
    `_fakes.py` still compares identical to one made with the original fakes."""
    assert sorted(p.name for p in oracle._HARNESS_FILES) == [
        "_equivalence_oracle.py",
        "_fakes.py",
        "_variant_fakes.py",
    ]
    copies = []
    for source in oracle._HARNESS_FILES:
        copy_path = tmp_path / source.name
        copy_path.write_bytes(source.read_bytes())
        copies.append(copy_path)
    original = oracle._harness_sha256(copies)
    for edited in copies:
        saved = edited.read_bytes()
        edited.write_bytes(saved + b"# edited\n")
        assert oracle._harness_sha256(copies) != original, edited.name
        edited.write_bytes(saved)
    assert oracle._harness_sha256(copies) == original


def _flux1_call(flux: oracle._Pipeline, **overrides: object) -> "oracle.mx.array":
    """One unpatched surrogate call at step 2, with the pipeline's own extra inputs unless overridden."""
    mx = oracle.mx
    config = flux._config(oracle._Request(seed=1))
    mx.random.seed(3)
    latents = mx.random.normal(flux.latent_shape)
    kwargs = {**flux.extra_inputs, **overrides}
    out = flux.transformer(
        t=2,
        config=config,
        hidden_states=latents,
        prompt_embeds=flux.prompt,
        pooled_prompt_embeds=flux.pooled,
        **kwargs,
    )
    mx.eval(out)
    return out


def _flux1() -> oracle._Pipeline:
    return oracle._build("flux1-dev", cfg=False)


def test_the_flux1_pipeline_passes_kontext_and_controlnet_inputs() -> None:
    """Bug: the FLUX.1 pipeline never passes kontext ids or controlnet samples, so a forward that drops them
    records the same as one that keeps them."""
    extras = _flux1().extra_inputs
    assert sorted(extras) == [
        "controlnet_block_samples",
        "controlnet_single_block_samples",
        "kontext_image_ids",
    ]
    assert len(extras["controlnet_block_samples"]) == 2
    assert len(extras["controlnet_single_block_samples"]) == 2


@pytest.mark.parametrize(
    "change",
    [
        "zero-rotary",
        "no-kontext",
        "no-joint-controlnet",
        "no-single-controlnet",
        "swapped-controlnet-lists",
        "reversed-joint-controlnet",
        "first-joint-block-only",
        "first-single-block-only",
        "other-pos-embed",
    ],
)
def test_the_flux1_surrogate_output_moves_with_each_input_and_block(change: str) -> None:
    """Bug: the surrogate ignores rotary, kontext or controlnet inputs, or has one block per stack, so a gated
    forward that drops or misroutes one of them, or runs only block 0, records the same output."""
    mx = oracle.mx
    flux = _flux1()
    baseline = _flux1_call(flux)
    tr = flux.transformer
    extras = flux.extra_inputs
    if change == "zero-rotary":
        compute = tr.compute_rotary_embeddings
        tr.compute_rotary_embeddings = lambda *a: mx.zeros_like(compute(*a))
        changed = _flux1_call(flux)
    elif change == "no-kontext":
        changed = _flux1_call(flux, kontext_image_ids=None)
    elif change == "no-joint-controlnet":
        changed = _flux1_call(flux, controlnet_block_samples=None)
    elif change == "no-single-controlnet":
        changed = _flux1_call(flux, controlnet_single_block_samples=None)
    elif change == "swapped-controlnet-lists":
        changed = _flux1_call(
            flux,
            controlnet_block_samples=extras["controlnet_single_block_samples"],
            controlnet_single_block_samples=extras["controlnet_block_samples"],
        )
    elif change == "reversed-joint-controlnet":
        changed = _flux1_call(flux, controlnet_block_samples=extras["controlnet_block_samples"][::-1])
    elif change == "first-joint-block-only":
        tr.transformer_blocks = tr.transformer_blocks[:1]
        changed = _flux1_call(flux)
    elif change == "first-single-block-only":
        tr.single_transformer_blocks = tr.single_transformer_blocks[:1]
        changed = _flux1_call(flux)
    else:
        tr.pos_embed = tr.pos_embed * 2.0
        changed = _flux1_call(flux)
    assert baseline.shape == changed.shape
    assert not mx.array_equal(baseline, changed)


@pytest.mark.parametrize("method", ["_apply_joint_transformer_block", "_apply_single_transformer_block"])
def test_the_flux1_surrogate_blocks_reject_an_unknown_keyword(method: str) -> None:
    """Bug: the surrogate's block helpers swallow keywords, so a forward that passes the rotary or a controlnet
    list under a wrong name is silently accepted and the input is lost."""
    mx = oracle.mx
    tr = _flux1().transformer
    hidden = mx.ones((1, 4, tr.dim))
    kwargs = {
        "idx": 0,
        "hidden_states": hidden,
        "encoder_hidden_states": hidden,
        "text_embeddings": mx.ones((1, tr.dim)),
        "image_rotary_embeddings": mx.ones((1, 1, tr.dim)),
    }
    if method == "_apply_joint_transformer_block":
        kwargs |= {"block": tr.transformer_blocks[0], "controlnet_block_samples": None}
    else:
        kwargs |= {"block": tr.single_transformer_blocks[0], "controlnet_single_block_samples": None}
    getattr(tr, method)(**kwargs)  # the keywords the gated forward passes are accepted
    with pytest.raises(TypeError):
        getattr(tr, method)(**kwargs, rotary_embeddings=kwargs["image_rotary_embeddings"])


@pytest.mark.parametrize("strategy", _STRATEGIES)
def test_every_strategy_has_a_skip_first_three_scenario(recorded: dict, strategy: str) -> None:
    """Bug: the oracle only ever runs the default window (skip_first_n_steps=1), so a forward that ignores the
    caller's window, or hard-codes 1, compares equal."""
    scenarios = [
        s
        for s in recorded["generations"].values()
        if s["strategy"] == strategy and s["handle"]["skip_first_n_steps"] == 3
    ]
    assert len(scenarios) == 1
    decisions = _decisions(scenarios[0])
    assert decisions[:3] == ["forced", "forced", "forced"]
    assert "skipped" in decisions[3:]


@pytest.mark.parametrize("strategy", _STRATEGIES)
def test_every_strategy_has_an_img2img_scenario(recorded: dict, strategy: str) -> None:
    """Bug: img2img (a generation that starts at init_time_step > 0) is never run on some strategy, so a forward
    that indexes the gate by the absolute step there compares equal."""
    runs = [
        run
        for name, s in recorded["generations"].items()
        if s["strategy"] == strategy and "/img2img/" in name
        for run in s["runs"]
    ]
    assert runs
    assert all(run["error"] is None and run["last_generation"]["num_steps"] == 6 for run in runs)
    assert any("skipped" in [d[4] for d in run["last_generation"]["decisions"]] for run in runs)


# ---------- expected-diff mode (--expect) ----------

_EXPECTED = {
    "required": ["threshold_matrix/*/thresh=None,coeffs=caller/warnings*"],
    "allowed": ["generations/*/restore-in-loop/*"],
    "forbidden": ["*/rel_l1_thresh*"],
}
_R6_ROW = "threshold_matrix/flux1-dev/thresh=None,coeffs=caller/warnings"


def test_expect_rejects_an_unlisted_difference() -> None:
    """Bug: the expected-diff mode waves through any change, so a moved latent hides behind the intended warning."""
    moved_latent = "generations/flux1-dev/default/runs[0]/final_latents/sha256_f32"
    assert oracle.check_expected([_R6_ROW, moved_latent], _EXPECTED, allow_unmet=False) == (
        [moved_latent],
        [],
    )


def test_expect_rejects_a_forbidden_path_inside_an_allowed_glob() -> None:
    """Bug: an allowed glob also lets a forbidden path through, so a moved threshold inside a restore scenario
    passes as an intended R10 difference."""
    moved_thresh = "generations/flux1-dev/restore-in-loop/handle/rel_l1_thresh"
    refusal = "generations/flux1-dev/restore-in-loop/in_loop_restore"
    assert oracle.check_expected([_R6_ROW, refusal, moved_thresh], _EXPECTED, allow_unmet=False) == (
        [moved_thresh],
        [],
    )


def test_expect_reports_an_unmet_required_glob() -> None:
    """Bug: an intended change (the caller-coefficient FutureWarning row) is silently absent and the run still
    passes."""
    refusal = "generations/flux1-dev/restore-in-loop/in_loop_restore"
    assert oracle.check_expected([refusal], _EXPECTED, allow_unmet=False) == (
        [],
        ["threshold_matrix/*/thresh=None,coeffs=caller/warnings*"],
    )
    assert oracle.check_expected([], _EXPECTED, allow_unmet=False)[1] == _EXPECTED["required"]


def test_allow_unmet_still_rejects_unexpected() -> None:
    """Bug: the per-task mode (--allow-unmet) hides a regression along with the not-yet-landed changes."""
    moved_latent = "generations/qwen-image/default/runs[0]/final_latents/sha256_f32"
    assert oracle.check_expected([moved_latent], _EXPECTED, allow_unmet=True) == ([moved_latent], [])
    assert oracle.check_expected([], _EXPECTED, allow_unmet=True) == ([], [])


def test_compare_cli_with_expect_exits_0_only_when_both_lists_are_empty(
    recorded: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bug: `compare --expect` exits 0 on an unexpected difference or an unmet required glob, so the 0.18 gate
    never stops a task."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    expect = tmp_path / "expect.json"
    a.write_text(json.dumps(recorded))
    changed = copy.deepcopy(recorded)
    changed["threshold_matrix"]["flux1-dev"]["thresh=None,coeffs=caller"]["warnings"].append(
        ["FutureWarning", "pass rel_l1_thresh", "x.py", 1]
    )
    b.write_text(json.dumps(changed))
    expect.write_text(json.dumps(_EXPECTED))
    assert oracle.main(["compare", str(a), str(b), "--expect", str(expect)]) == 0
    capsys.readouterr()
    assert oracle.main(["compare", str(a), str(a), "--expect", str(expect)]) == 1
    assert "threshold_matrix/*/thresh=None,coeffs=caller/warnings*" in capsys.readouterr().out
    assert oracle.main(["compare", str(a), str(a), "--expect", str(expect), "--allow-unmet"]) == 0
    capsys.readouterr()
    changed["generations"]["flux1-dev/default"]["runs"][0]["final_latents"]["sha256_f32"] = "0" * 64
    b.write_text(json.dumps(changed))
    assert oracle.main(["compare", str(a), str(b), "--expect", str(expect), "--allow-unmet"]) == 1
    assert "generations/flux1-dev/default/runs[0]/final_latents/sha256_f32" in capsys.readouterr().out


# ---------- the 0.18 scenarios ----------


def _strategies_named(recorded: dict, kind: str) -> set[str]:
    return {s["strategy"] for name, s in recorded["generations"].items() if name.endswith(f"/{kind}")}


def test_new_scenarios_cover_every_strategy(recorded: dict) -> None:
    """Bug: restore() refusals are left unexercised on one patch strategy, so a refusal missing there (or a wrapper
    still dropped) compares equal."""
    assert _strategies_named(recorded, "restore-in-loop") == set(_STRATEGIES)
    assert _strategies_named(recorded, "wrapper-over-generate-image") == set(_STRATEGIES)
    assert _strategies_named(recorded, "wrapper-over-predict") == {"predict-flux2", "predict-z-image"}


def test_caller_coefficient_and_uncalibrated_scenarios_cover_every_variant(recorded: dict) -> None:
    """Bug: the caller-coefficient no-move witness or the uncalibrated-checkpoint probe skips a variant, so a moved
    decision or a missing warning there is never compared."""
    witnesses = {
        s["handle"]["variant_id"]
        for name, s in recorded["generations"].items()
        if name.endswith("/caller-coeffs-default-thresh")
    }
    assert witnesses == set(_REGISTRY)
    uncalibrated = {
        cell["handle"]["variant_id"]
        for name, cell in recorded["warning_attribution"].items()
        if name.startswith("uncalibrated/")
    }
    assert uncalibrated == set(_REGISTRY)


def test_the_witness_runs_caller_coefficients_without_a_threshold(recorded: dict) -> None:
    """Bug: the witness passes a threshold (or no coefficients), so it never reaches the caller-coefficient default
    it exists to pin."""
    witnesses = {
        name: s
        for name, s in recorded["generations"].items()
        if name.endswith("/caller-coeffs-default-thresh")
    }
    assert len(witnesses) == 10
    for name, s in witnesses.items():
        assert s["handle"]["coefficients"] == [x.hex() for x in (0.0, 0.0, 0.0, 0.0, 0.1)], name
        assert s["handle"]["provenance"]["source"] == "user", name
        # Caller coefficients and no threshold resolve to 0.30 on Krea and 0.20 on every other variant.
        resolved = 0.30 if name.startswith("flux1-krea-dev/") else 0.20
        assert s["handle"]["rel_l1_thresh"] == resolved.hex(), name
        assert s["runs"][0]["error"] is None, name
        assert s["runs"][0]["last_generation"]["num_steps"] == 10, name


@pytest.mark.parametrize("strategy", _STRATEGIES)
def test_the_restore_in_loop_probe_calls_restore_after_step_3(recorded: dict, strategy: str) -> None:
    """Bug: the in-loop hook never fires (or fires at another step), so the R10 in-loop refusal is never exercised."""
    (scenario,) = [
        s
        for name, s in recorded["generations"].items()
        if name.endswith("/restore-in-loop") and s["strategy"] == strategy
    ]
    assert scenario["restore_called_at_step"] == 3
    assert len(scenario["runs"][0]["steps"]) >= 4


@pytest.mark.parametrize("kind", ["wrapper-over-generate-image", "wrapper-over-predict"])
def test_a_wrapper_probe_either_refuses_or_drops_the_wrapper(recorded: dict, kind: str) -> None:
    """Bug: the probe installs its wrapper on an attribute restore() never touches, so restore() succeeds and the
    wrapper survives: the probe then shows neither the old overwrite nor the new refusal."""
    probes = {name: s for name, s in recorded["generations"].items() if name.endswith(f"/{kind}")}
    assert probes
    for name, probe in probes.items():
        refused = probe["restore"] is not None
        assert probe["wrapper_kept"] is refused, name
        assert probe["second_restore"] is None, name
        # Torn down to the same attributes as an ordinary scenario on that variant: nothing left on the model.
        ordinary = recorded["generations"][name.split("/")[0] + "/default"]["teardown"]
        assert probe["teardown"] == ordinary, name
