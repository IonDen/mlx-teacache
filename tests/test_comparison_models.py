"""Pure parts of the comparison model glue: cached prompt override, cache-hit check, encoder release."""

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _comparison_models as models  # noqa: E402
import _comparison_recipes as cr  # noqa: E402


def test_cached_call_returns_the_precomputed_result_on_the_exact_key() -> None:
    """Bug: the override re-encodes, or returns something else."""
    result = object()
    key = {"prompt": "p", "negative_prompt": " ", "guidance": 4.0}
    assert models._cached_call(result, key, "klein")(**key) is result


def test_cached_call_raises_on_any_key_change() -> None:
    """Bug: mflux changes its call (e.g. a new negative-prompt default) and the old embeddings are reused silently."""
    call = models._cached_call(object(), {"prompt": "p", "negative_prompt": " ", "guidance": 4.0}, "klein")
    with pytest.raises(RuntimeError, match="cache miss"):
        call(prompt="p", negative_prompt="", guidance=4.0)


def test_assert_prompt_cache_hit_counts_entries_for_cache_models_only() -> None:
    """Bug: a second (re-encoded) cache entry goes unnoticed, or override models are wrongly checked."""
    flux1, klein = cr.recipe_for("flux1-dev"), cr.recipe_for("klein-base-4b")
    models.assert_prompt_cache_hit(SimpleNamespace(prompt_cache={"p": 0}), flux1)
    with pytest.raises(RuntimeError, match="2 entries"):
        models.assert_prompt_cache_hit(SimpleNamespace(prompt_cache={"p": 0, "q": 1}), flux1)
    models.assert_prompt_cache_hit(SimpleNamespace(), klein)


def _fake_klein(prompt: str, guidance: float) -> SimpleNamespace:
    calls: list[dict] = []

    def _encode_prompt_pair(**kwargs: object) -> tuple:
        calls.append(kwargs)
        return (f"pos:{kwargs['negative_prompt']}", "ids", f"neg:{kwargs['negative_prompt']}", "neg_ids")

    return SimpleNamespace(_encode_prompt_pair=_encode_prompt_pair, _calls=calls)


def test_precompute_prompt_klein_serves_the_real_negative_under_mflux_hardcoded_space_key() -> None:
    """Bug: the harness encodes with the real negative but keys the override on that same real negative, so
    mflux's actual call (always ``negative_prompt=" "``) misses the override and re-encodes with nothing."""
    recipe = dataclasses.replace(cr.recipe_for("klein-base-4b"), negative_prompt="blurry, watermark")
    flux = _fake_klein("a scene", recipe.guidance)
    models.precompute_prompt(flux, recipe, "a scene")

    # The real negative was actually encoded (not silently dropped for " ").
    assert flux._calls == [
        {"prompt": "a scene", "negative_prompt": "blurry, watermark", "guidance": recipe.guidance}
    ]
    # mflux always calls with negative_prompt=" " (flux2_klein.py:76-80); the override must serve the
    # embeddings encoded from the real negative under that exact key.
    result = flux._encode_prompt_pair(prompt="a scene", negative_prompt=" ", guidance=recipe.guidance)
    assert result == ("pos:blurry, watermark", "ids", "neg:blurry, watermark", "neg_ids")


def test_precompute_prompt_klein_override_still_raises_on_a_prompt_change() -> None:
    """Bug: keying the override on mflux's hardcoded negative loosens the check so an unrelated prompt
    change silently reuses stale embeddings instead of raising."""
    recipe = dataclasses.replace(cr.recipe_for("klein-base-4b"), negative_prompt="blurry")
    flux = _fake_klein("a scene", recipe.guidance)
    models.precompute_prompt(flux, recipe, "a scene")
    with pytest.raises(RuntimeError, match="cache miss"):
        flux._encode_prompt_pair(prompt="a DIFFERENT scene", negative_prompt=" ", guidance=recipe.guidance)


def test_precompute_prompt_klein_without_a_negative_keys_on_the_same_space_as_before() -> None:
    """Bug: the default (no recipe negative) path stops matching mflux's hardcoded ' ' call."""
    recipe = cr.recipe_for("klein-base-4b")
    assert recipe.negative_prompt is None
    flux = _fake_klein("a scene", recipe.guidance)
    models.precompute_prompt(flux, recipe, "a scene")
    assert flux._calls == [{"prompt": "a scene", "negative_prompt": " ", "guidance": recipe.guidance}]


def _fake_z_image(prompt: str, guidance: float) -> SimpleNamespace:
    calls: list[dict] = []

    def _encode_prompts(**kwargs: object) -> tuple:
        calls.append(kwargs)
        return (f"pos:{kwargs['negative_prompt']}", f"neg:{kwargs['negative_prompt']}")

    return SimpleNamespace(_encode_prompts=_encode_prompts, _calls=calls)


def test_precompute_prompt_z_image_keys_the_override_on_the_recipe_negative() -> None:
    """Bug: Z-Image's override is keyed on None regardless of the recipe, so a real negative never reaches
    the cached call and generate_image's real encode call misses it."""
    recipe = dataclasses.replace(cr.recipe_for("z-image-base"), negative_prompt="watermark")
    flux = _fake_z_image("a scene", recipe.guidance)
    models.precompute_prompt(flux, recipe, "a scene")

    assert flux._calls == [{"prompt": "a scene", "negative_prompt": "watermark", "guidance": recipe.guidance}]
    result = flux._encode_prompts(prompt="a scene", negative_prompt="watermark", guidance=recipe.guidance)
    assert result == ("pos:watermark", "neg:watermark")
    with pytest.raises(RuntimeError, match="cache miss"):
        flux._encode_prompts(prompt="a scene", negative_prompt=None, guidance=recipe.guidance)


def test_precompute_prompt_flux1_raises_when_a_negative_prompt_is_set() -> None:
    """Bug: FLUX.1 silently accepts a negative prompt that mflux's generate_image ignores, producing an
    image that looks like the negative was honored when it never was."""
    recipe = dataclasses.replace(cr.recipe_for("flux1-dev"), negative_prompt="blurry")
    with pytest.raises(ValueError, match="negative prompt"):
        models.precompute_prompt(SimpleNamespace(), recipe, "a scene")


def test_precompute_prompt_krea_raises_when_a_negative_prompt_is_set() -> None:
    """Bug: only flux1-dev is guarded and flux1-krea-dev slips a negative through silently."""
    recipe = dataclasses.replace(cr.recipe_for("flux1-krea-dev"), negative_prompt="blurry")
    with pytest.raises(ValueError, match="negative prompt"):
        models.precompute_prompt(SimpleNamespace(), recipe, "a scene")


def test_generate_kwargs_for_threads_the_negative_only_where_generate_image_accepts_it() -> None:
    """Bug: Klein's generate_image (no negative_prompt parameter at all) is handed one and raises a
    TypeError at run time, or Z-Image/Qwen's real negative never reaches generate_image."""
    z_image = dataclasses.replace(cr.recipe_for("z-image-base"), negative_prompt="watermark")
    qwen = dataclasses.replace(cr.recipe_for("qwen-image"), negative_prompt="watermark")
    klein = dataclasses.replace(cr.recipe_for("klein-base-4b"), negative_prompt="watermark")
    flux1 = dataclasses.replace(cr.recipe_for("flux1-dev"), negative_prompt=None)

    assert models.generate_kwargs_for(z_image) == {"negative_prompt": "watermark"}
    assert models.generate_kwargs_for(qwen) == {"negative_prompt": "watermark"}
    assert models.generate_kwargs_for(klein) == {}
    assert models.generate_kwargs_for(flux1) == {}


def test_negative_used_for_is_false_when_unset_low_guidance_or_unsupported_loader() -> None:
    """Bug: the quality-probe record claims a negative was used when mflux actually dropped it (guidance <=
    1.0, or a loader like FLUX.1/Krea that has no route for a negative at all)."""
    base = cr.recipe_for("klein-base-4b")  # guidance=4.0
    assert models.negative_used_for(base) is False  # no negative set at all

    with_neg = dataclasses.replace(base, negative_prompt="blurry")
    assert models.negative_used_for(with_neg) is True

    at_threshold = dataclasses.replace(with_neg, guidance=1.0)
    assert models.negative_used_for(at_threshold) is False  # mflux's own > 1.0 check, boundary

    just_above = dataclasses.replace(with_neg, guidance=1.0001)
    assert models.negative_used_for(just_above) is True

    flux1_with_neg = dataclasses.replace(cr.recipe_for("flux1-dev"), negative_prompt="blurry")
    assert models.negative_used_for(flux1_with_neg) is False  # FLUX.1 has no negative route regardless


def test_negative_used_for_qwen_runs_the_negative_branch_at_any_guidance_other_than_one() -> None:
    """Bug: Qwen is judged by the guidance > 1.0 threshold the CFG-gated loaders (Klein/Z-Image) use, when
    Qwen's own generate_image always combines neg + guidance * (pos - neg), so its negative reaches the
    model at any guidance except exactly 1.0 (where the combine is a no-op), including below 1.0."""
    qwen = cr.recipe_for("qwen-image")
    below_one = dataclasses.replace(qwen, negative_prompt="blurry", guidance=0.8)
    assert models.negative_used_for(below_one) is True

    at_one = dataclasses.replace(qwen, negative_prompt="blurry", guidance=1.0)
    assert models.negative_used_for(at_one) is False

    z_image_at_one = dataclasses.replace(
        cr.recipe_for("z-image-base"), negative_prompt="watermark", guidance=1.0
    )
    assert models.negative_used_for(z_image_at_one) is False


def test_release_text_encoders_clears_only_encoders() -> None:
    """Bug: the transformer or VAE is released (black image), or an encoder survives (memory), or only one of
    the three encoder attrs (clip/t5/text) actually gets released."""
    flux = SimpleNamespace(
        clip_text_encoder=object(),
        t5_text_encoder=object(),
        text_encoder=object(),
        transformer=object(),
        vae=object(),
    )
    assert models.release_text_encoders(flux) == ["clip_text_encoder", "t5_text_encoder", "text_encoder"]
    assert flux.clip_text_encoder is None and flux.t5_text_encoder is None and flux.text_encoder is None
    assert flux.transformer is not None and flux.vae is not None


def test_evaluate_modules_batches_by_byte_threshold_and_skips_missing_attrs() -> None:
    """Bug: one mx.eval(module.parameters()) call materializes an entire module at once, which can hold a
    large share of a bf16 model's weights (e.g. Qwen-Image) in memory simultaneously."""
    import mlx.core as mx

    params = {f"w{i}": mx.zeros((1024,), dtype=mx.float32) for i in range(5)}  # 4096 bytes each
    module = SimpleNamespace(parameters=lambda: params)
    flux = SimpleNamespace(text_encoder=module, transformer=None)  # transformer: present but None
    calls: list[list] = []
    clears: list[int] = []
    models.evaluate_modules(
        flux,
        ["text_encoder", "transformer", "vae"],  # vae: missing entirely
        batch_bytes=10_000,
        eval_fn=calls.append,
        clear_fn=lambda: clears.append(1),
    )
    seen = [id(arr) for batch in calls for arr in batch]
    assert seen == [id(arr) for arr in params.values()]  # every array exactly once, in order
    assert len(calls) >= 2
    assert all(sum(arr.nbytes for arr in batch) >= 10_000 for batch in calls[:-1])
    assert clears == [1]  # only text_encoder was present and non-None -> exactly one module evaluated
