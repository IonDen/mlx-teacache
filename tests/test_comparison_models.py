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
    """Bug: the transformer or VAE is released (black image), or an encoder survives (memory)."""
    flux = SimpleNamespace(text_encoder=object(), transformer=object(), vae=object())
    assert models.release_text_encoders(flux) == ["text_encoder"]
    assert flux.text_encoder is None and flux.transformer is not None and flux.vae is not None
