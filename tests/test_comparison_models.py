"""Pure parts of the comparison model glue: cached prompt override, cache-hit check, encoder release."""

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
