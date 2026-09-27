"""Recipe data and the probe decision for the comparison page (pure-core lane)."""

import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _comparison_recipes as cr  # noqa: E402

GIB = 1024**3
V = {"mflux": "0.20.0", "mlx": "0.32.2", "mlx_taef": "0.8.3"}
REPORT_PATH = Path(__file__).resolve().parents[1] / "_artifacts" / "comparison" / "report.json"


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
    others = [r for r in cr.RECIPES if r.slug not in ("qwen-image", "z-image-base")]
    assert others and all(cr.prompt_for(r) == cr.PROMPT for r in others)
    # Z-Image carries its own prompt (approved Z1 wording); the Qwen suffix must not leak into it either.
    assert cr.prompt_for(cr.recipe_for("z-image-base")) == cr.Z_IMAGE_PROMPT


def test_prompt_override_replaces_the_shared_prompt() -> None:
    """Bug: a per-model prompt override is ignored and the shared PROMPT is used anyway."""
    r = dataclasses.replace(cr.recipe_for("flux1-dev"), prompt="a red bicycle in the rain")
    assert cr.prompt_for(r) == "a red bicycle in the rain"


def test_prompt_override_on_qwen_still_gets_the_vendor_suffix() -> None:
    """Bug: overriding the prompt drops Qwen's suffix rule, or the suffix leaks onto a non-Qwen override."""
    qwen = dataclasses.replace(cr.recipe_for("qwen-image"), prompt="a red bicycle in the rain")
    assert cr.prompt_for(qwen) == "a red bicycle in the rain" + cr.QWEN_PROMPT_SUFFIX
    non_qwen = dataclasses.replace(cr.recipe_for("flux1-dev"), prompt="a red bicycle in the rain")
    assert cr.QWEN_PROMPT_SUFFIX not in cr.prompt_for(non_qwen)


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
        ("cache_gb", 3.0),
        ("wired_cap_gb", 20),
        ("prompt", "a different scene entirely"),
        ("negative_prompt", "blurry, low quality"),
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


def test_negative_prompt_sha256_absent_and_stamp_unchanged_when_unset() -> None:
    """Bug: the stamp always emits negative_prompt_sha256 (even as sha256("")), so every chunk and probe
    record written before this field existed is refused by check_chunk_stamp's exact-match comparison
    ("measured under a different recipe") even though nothing about the measurement actually changed."""
    r = cr.recipe_for("flux1-dev")
    assert r.negative_prompt is None
    stamp = cr.recipe_stamp(r, versions=V)
    assert "negative_prompt_sha256" not in stamp
    assert stamp == {
        "slug": "flux1-dev",
        "checkpoint": r.checkpoint,
        "decoder": r.decoder,
        "steps": r.steps,
        "guidance": r.guidance,
        "quantize": r.quantize,
        "width": r.width,
        "height": r.height,
        "free_encoders": r.free_encoders,
        "cache_gb": r.cache_gb,
        "wired_cap_gb": r.wired_cap_gb,
        "seed": cr.SEED,
        "prompt_sha256": cr.prompt_sha256(cr.PROMPT),
        "version_mflux": "0.20.0",
        "version_mlx": "0.32.2",
        "version_mlx_taef": "0.8.3",
    }


def test_negative_prompt_sha256_present_and_hashes_the_real_negative_when_set() -> None:
    """Bug: the stamp hashes a placeholder instead of the actual negative text, so two different negatives
    collide on one stamp and a stale chunk gets reused; or setting a negative doesn't add the key at all."""
    unset = cr.recipe_for("flux1-dev")
    r = dataclasses.replace(unset, negative_prompt="blurry, low quality")
    stamp = cr.recipe_stamp(r, versions=V)
    assert stamp["negative_prompt_sha256"] == cr.prompt_sha256("blurry, low quality")
    assert stamp != cr.recipe_stamp(unset, versions=V)


def test_apply_overrides_replaces_only_the_given_fields() -> None:
    """Bug: apply_overrides touches a field the caller left unset, or ignores one the caller did set."""
    r = cr.recipe_for("flux1-dev")
    out = cr.apply_overrides(r, guidance=7.0, prompt="a bicycle")
    assert out.guidance == 7.0 and out.prompt == "a bicycle"
    assert out.width == r.width and out.height == r.height and out.negative_prompt is None
    assert out.quantize == r.quantize and out.slug == r.slug


def test_apply_overrides_with_nothing_given_returns_an_equal_recipe() -> None:
    """Bug: calling apply_overrides with every field left at None still mutates something."""
    r = cr.recipe_for("flux1-dev")
    assert cr.apply_overrides(r) == r


def test_apply_overrides_sets_width_height_quantize_and_negative_prompt() -> None:
    """Bug: one of the numeric/negative override fields is silently dropped from apply_overrides."""
    r = cr.recipe_for("z-image-base")
    out = cr.apply_overrides(r, width=512, height=512, quantize=4, negative_prompt="watermark")
    assert (out.width, out.height, out.quantize, out.negative_prompt) == (512, 512, 4, "watermark")


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
    r = cr.with_resolution(cr.recipe_for("klein-base-9b"), 768, 1024)
    rec = cr.probe_record(
        r,
        phase_peaks={"load": 23 * GIB, "encode": 10 * GIB, "generation": 18 * GIB},
        min_host_free_pct=40.0,
        working_set_bytes=24 * GIB,
        versions=V,
        peak_footprint_bytes=20
        * GIB,  # comfortably under the footprint gate: only the peak overflow fails this
    )
    assert rec["pass"] is False and rec["active_peak_bytes"] == 23 * GIB
    assert rec["width"] == 768 and rec["stamp"] == cr.recipe_stamp(r, versions=V)


def test_probe_record_fails_when_aborted_or_host_never_sampled() -> None:
    """Bug: an aborted probe or a dead sampler (None) counts as a pass."""
    r = cr.recipe_for("klein-base-9b")
    peaks = {"load": 5 * GIB, "encode": 5 * GIB, "generation": 5 * GIB}
    # Same inputs pass with aborted=None, so the two assertions below are isolating the abort/host-free
    # gates, not accidentally passing because the measurement itself would have failed anyway.
    assert (
        cr.probe_record(
            r,
            phase_peaks=peaks,
            min_host_free_pct=60.0,
            working_set_bytes=24 * GIB,
            versions=V,
            peak_footprint_bytes=20 * GIB,
        )["pass"]
        is True
    )
    assert (
        cr.probe_record(
            r,
            phase_peaks=peaks,
            min_host_free_pct=60.0,
            working_set_bytes=24 * GIB,
            versions=V,
            peak_footprint_bytes=20 * GIB,
            aborted="watchdog",
        )["pass"]
        is False
    )
    assert (
        cr.probe_record(
            r,
            phase_peaks=peaks,
            min_host_free_pct=None,
            working_set_bytes=24 * GIB,
            versions=V,
            peak_footprint_bytes=20 * GIB,
        )["pass"]
        is False
    )


def test_probe_record_fails_when_the_process_footprint_crosses_the_guard_limit() -> None:
    """Bug: the probe judges only MLX's own peak, ignoring the OS process footprint mlx-guard actually kills
    on (25 GiB, running 2-4 GiB above MLX's count) -- a probe can pass while mlx-guard would still kill it."""
    r = cr.recipe_for("klein-base-9b")  # cache_gb=2.0
    peaks = {"load": 15 * GIB, "encode": 10 * GIB, "generation": 12 * GIB}  # MLX peak: well under budget

    within_budget = cr.probe_record(
        r,
        phase_peaks=peaks,
        min_host_free_pct=40.0,
        working_set_bytes=24 * GIB,
        versions=V,
        peak_footprint_bytes=20 * GIB,
    )
    assert within_budget["pass"] is True

    over_guard_limit = cr.probe_record(
        r,
        phase_peaks=peaks,
        min_host_free_pct=40.0,
        working_set_bytes=24 * GIB,
        versions=V,
        peak_footprint_bytes=int(24.5 * GIB),
    )
    assert over_guard_limit["pass"] is False
    assert over_guard_limit["peak_footprint_bytes"] == int(24.5 * GIB)

    footprint_never_sampled = cr.probe_record(
        r, phase_peaks=peaks, min_host_free_pct=40.0, working_set_bytes=24 * GIB, versions=V
    )
    assert footprint_never_sampled["pass"] is False
    assert footprint_never_sampled["peak_footprint_bytes"] is None


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


def test_z_image_recipe_matches_the_approved_z1_quality_probe_exactly() -> None:
    """Bug: the Z-Image recipe drifts from the Z1 quality probe Denis approved on 2026-09-26 (the prompt,
    negative, resolution, guidance or quantize differ by even a character or a pixel), so the page's
    Z-Image images are not the ones that were reviewed. The literal is Z1's record.json stamp (softened
    prompt that drops "flushed cheeks"/"sweat on her forehead", the blotchy-skin negative, 864x1152, g 4)."""
    versions = {"mflux": "0.20.0", "mlx": "0.32.2", "mlx_taef": "0.8.3"}
    assert cr.recipe_stamp(cr.recipe_for("z-image-base"), versions=versions) == {
        "slug": "z-image-base",
        "checkpoint": "Tongyi-MAI/Z-Image",
        "decoder": "zimage",
        "steps": 50,
        "guidance": 4.0,
        "quantize": 8,
        "width": 864,
        "height": 1152,
        "free_encoders": False,
        "cache_gb": 2.0,
        "wired_cap_gb": 24,
        "seed": 42,
        "prompt_sha256": "6920d5f048458eed9425ff7a6d2a1d9eb7be307dff80c00cf8446e4acb51d044",
        "version_mflux": "0.20.0",
        "version_mlx": "0.32.2",
        "version_mlx_taef": "0.8.3",
        "negative_prompt_sha256": "e370ca56997ce475aa66d9063f56c40f1117019f68df51d02f9ac2e674e57343",
    }


def test_rel_l1_thresh_override_is_stamped_only_when_set() -> None:
    """Bug: a probe at a non-default TeaCache threshold stamps like a default-threshold run (so the two are
    confused), or every existing chunk's stamp changes because the key is always emitted."""
    r = cr.recipe_for("klein-base-4b")
    assert r.rel_l1_thresh is None
    assert "rel_l1_thresh" not in cr.recipe_stamp(r, versions=V)
    lower = cr.apply_overrides(r, rel_l1_thresh=0.12)
    assert lower.rel_l1_thresh == 0.12
    assert cr.recipe_stamp(lower, versions=V)["rel_l1_thresh"] == 0.12


def test_teacache_kwargs_pass_the_threshold_only_when_the_recipe_sets_one() -> None:
    """Bug: B ignores a recipe's threshold override and runs at the variant default, or passes
    rel_l1_thresh=None explicitly instead of leaving the library default in charge."""
    r = cr.recipe_for("klein-base-4b")
    assert cr.teacache_kwargs(r) == {}
    assert cr.teacache_kwargs(cr.apply_overrides(r, rel_l1_thresh=0.1)) == {"rel_l1_thresh": 0.1}


def test_klein_base_4b_recipe_matches_the_approved_k1_quality_probe_exactly() -> None:
    """Bug: the Klein 4B recipe drifts from the K1 probe Denis approved on 2026-09-26 (q8, 864x1152, shared
    prompt, no negative, default threshold), so the page's Klein images are not the ones reviewed."""
    versions = {"mflux": "0.20.0", "mlx": "0.32.2", "mlx_taef": "0.8.3"}
    assert cr.recipe_stamp(cr.recipe_for("klein-base-4b"), versions=versions) == {
        "slug": "klein-base-4b",
        "checkpoint": "black-forest-labs/FLUX.2-klein-base-4B",
        "decoder": "taef2",
        "steps": 50,
        "guidance": 4.0,
        "quantize": 8,
        "width": 864,
        "height": 1152,
        "free_encoders": False,
        "cache_gb": 2.0,
        "wired_cap_gb": 22,
        "seed": 42,
        "prompt_sha256": "18b6b14a31c6808afb26f20b0c828c4262dccb78af8e28a1aa38bd696938284e",
        "version_mflux": "0.20.0",
        "version_mlx": "0.32.2",
        "version_mlx_taef": "0.8.3",
    }


def test_klein_base_9b_recipe_follows_the_4b_settings_at_768x1024() -> None:
    """Bug: Klein 9B drops back to q4 (or gains a threshold/prompt override) while 4B runs K1's q8 recipe, so the two
    Klein rows differ in more than size; or 9B goes back to 864x1152, where its first q8 step ran past macOS's ~5 s
    GPU watchdog on this M1 Max (kIOGPUCommandBufferCallbackErrorImpactingInteractivity, 2026-09-26); or 9B stops
    freeing its text encoder."""
    r = cr.recipe_for("klein-base-9b")
    assert (r.quantize, r.width, r.height, r.guidance, r.steps) == (8, 768, 1024, 4.0, 50)
    assert r.prompt is None and r.negative_prompt is None and r.rel_l1_thresh is None
    assert r.free_encoders is True
    assert r.fallback == (576, 768)


def test_every_recipe_matches_the_settings_the_committed_report_recorded() -> None:
    """Bug: a recipe field (prompt, steps, guidance, quantize, resolution, negative) drifts from the value
    that actually produced the committed comparison images, so a live recipe silently describes different
    settings from the ones the report and pages show."""
    report = json.loads(REPORT_PATH.read_text())
    variants = report["variants"]
    checked = 0
    for rec in cr.RECIPES:
        v = variants.get(rec.slug)
        if v is None:
            continue
        checked += 1
        assert v["prompt_sha256"] == cr.prompt_sha256(cr.prompt_for(rec)), rec.slug
        assert (v["steps"], v["guidance"], v["quantize"]) == (rec.steps, rec.guidance, rec.quantize), rec.slug
        assert (v["width"], v["height"]) in {(rec.width, rec.height), rec.fallback}, rec.slug
        if "negative_prompt" in v:
            assert v["negative_prompt"] == rec.negative_prompt, rec.slug
    assert checked == 6
