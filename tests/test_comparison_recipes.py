"""Recipe data and the probe decision for the comparison page (pure-core lane)."""

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _comparison_recipes as cr  # noqa: E402

GIB = 1024**3
V = {"mflux": "0.20.0", "mlx": "0.32.2", "mlx_taef": "0.8.3"}


def test_six_non_distilled_recipes_in_run_order() -> None:
    """Bug: a distilled variant slips in, or the heavy rows are not last."""
    assert [r.slug for r in cr.RECIPES] == [
        "flux1-dev",
        "flux1-krea-dev",
        "klein-base-4b",
        "z-image-base",
        "klein-base-9b",
        "qwen-image",
    ]


def test_qwen_alone_gets_the_vendor_suffix() -> None:
    """Bug: the suffix leaks into every prompt, or never reaches Qwen."""
    assert cr.prompt_for(cr.recipe_for("qwen-image")) == cr.PROMPT + ", Ultra HD, 4K, cinematic composition."
    assert all(cr.prompt_for(r) == cr.PROMPT for r in cr.RECIPES if r.slug != "qwen-image")


def test_prompt_stays_short_enough_for_the_t5_budget() -> None:
    """Bug: an edited prompt grows past ~180 words (~240 T5 tokens), close to FLUX.1's 512-token cap."""
    assert len(cr.PROMPT.split()) <= 180


def test_qwen_recipe_loads_the_original_checkpoint_and_frees_encoders() -> None:
    """Bug: Qwen reverts to the 2512 alias or loses its memory settings."""
    qwen = cr.recipe_for("qwen-image")
    assert (qwen.checkpoint, qwen.free_encoders, qwen.fallback, qwen.cache_gb) == (
        "Qwen/Qwen-Image",
        True,
        (576, 768),
        1.0,
    )


def test_unknown_slug_raises() -> None:
    """Bug: an excluded variant resolves silently."""
    with pytest.raises(KeyError):
        cr.recipe_for("flux1-schnell")


@pytest.mark.parametrize(
    "field,value",
    [
        ("guidance", 5.0),
        ("quantize", 8),
        ("checkpoint", "x/y"),
        ("decoder", "taef2"),
        ("free_encoders", True),
        ("steps", 30),
        ("width", 512),
        ("height", 512),
    ],
)
def test_recipe_stamp_changes_with_every_measured_field(field: str, value: object) -> None:
    """Bug: a field that changes the measurement is missing from the stamp, so an old chunk is reused."""
    r = cr.recipe_for("flux1-dev")
    assert cr.recipe_stamp(r, versions=V) != cr.recipe_stamp(
        dataclasses.replace(r, **{field: value}), versions=V
    )


def test_recipe_stamp_hashes_the_prompt_actually_sent() -> None:
    """Bug: Qwen's stamp hashes the bare prompt, not the suffixed one."""
    stamp = cr.recipe_stamp(cr.recipe_for("qwen-image"), versions=V)
    assert stamp["prompt_sha256"] == cr.prompt_sha256(cr.PROMPT + cr.QWEN_PROMPT_SUFFIX)
    assert "git_sha" not in stamp and stamp["version_mflux"] == "0.20.0"


def test_probe_passes_is_strict_on_both_limits() -> None:
    """Bug: `>=`/`<` flipped at the boundary, so an exact-edge size passes."""
    ok = dict(active_peak_bytes=20 * GIB, cache_limit_bytes=2 * GIB, working_set_bytes=24 * GIB)
    assert cr.probe_passes(**ok, min_host_free_pct=20.0)
    assert not cr.probe_passes(**ok, min_host_free_pct=19.9)
    assert not cr.probe_passes(
        active_peak_bytes=22 * GIB,
        cache_limit_bytes=2 * GIB,
        working_set_bytes=24 * GIB,
        min_host_free_pct=50.0,
    )


def test_probe_record_uses_the_largest_phase_peak() -> None:
    """Bug: only the generation peak is judged, while the load/encode peak is the real maximum."""
    r = cr.recipe_for("klein-base-9b")
    rec = cr.probe_record(
        r,
        phase_peaks={"load": 23 * GIB, "encode": 10 * GIB, "generation": 18 * GIB},
        min_host_free_pct=40.0,
        working_set_bytes=24 * GIB,
        versions=V,
    )
    assert rec["pass"] is False and rec["active_peak_bytes"] == 23 * GIB
    assert rec["width"] == 768 and rec["stamp"] == cr.recipe_stamp(r, versions=V)


def test_probe_record_fails_when_aborted_or_host_never_sampled() -> None:
    """Bug: an aborted probe or a dead sampler (None) counts as a pass."""
    r = cr.recipe_for("klein-base-9b")
    peaks = {"load": 5 * GIB, "encode": 5 * GIB, "generation": 5 * GIB}
    assert (
        cr.probe_record(
            r,
            phase_peaks=peaks,
            min_host_free_pct=60.0,
            working_set_bytes=24 * GIB,
            versions=V,
            aborted="watchdog",
        )["pass"]
        is False
    )
    assert (
        cr.probe_record(r, phase_peaks=peaks, min_host_free_pct=None, working_set_bytes=24 * GIB, versions=V)[
            "pass"
        ]
        is False
    )


def _rec(recipe: cr.Recipe, passed: bool) -> dict[str, object]:
    return {
        "width": recipe.width,
        "height": recipe.height,
        "pass": passed,
        "stamp": cr.recipe_stamp(recipe, versions=V),
    }


def test_resolve_resolution_picks_full_fallback_or_refuses() -> None:
    """Bug: the fallback is never reachable after a failed full-size probe, or a guess is made without probes."""
    r = cr.recipe_for("klein-base-9b")
    fb = cr.with_resolution(r, 576, 768)
    assert cr.resolve_resolution(r, [_rec(r, True)], versions=V) == r
    assert cr.resolve_resolution(r, [_rec(r, False), _rec(fb, True)], versions=V) == fb
    with pytest.raises(SystemExit, match="--probe"):
        cr.resolve_resolution(r, [], versions=V)
    with pytest.raises(SystemExit, match="--probe --fallback"):
        cr.resolve_resolution(r, [_rec(r, False)], versions=V)
    with pytest.raises(SystemExit, match="both"):
        cr.resolve_resolution(r, [_rec(r, False), _rec(fb, False)], versions=V)


def test_resolve_resolution_ignores_probes_from_another_recipe_or_version() -> None:
    """Bug: a stale probe (other mflux version) decides today's resolution."""
    r = cr.recipe_for("klein-base-9b")
    stale = {**_rec(r, True), "stamp": cr.recipe_stamp(r, versions={**V, "mflux": "0.19.1"})}
    with pytest.raises(SystemExit, match="--probe"):
        cr.resolve_resolution(r, [stale], versions=V)


def test_recipes_without_fallback_need_no_probe() -> None:
    """Bug: light rows demand a probe."""
    r = cr.recipe_for("flux1-dev")
    assert cr.resolve_resolution(r, [], versions=V) == r
