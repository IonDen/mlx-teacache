"""Unit tests for the Qwen mixed-precision quantization predicate
(scripts/qwen_mixed_precision.py). Pure function — no weights, no mflux, no MLX.
The predicate decides per-layer precision: bf16 for embeddings/projection, q8 for
the edge transformer blocks, q4 (default) for the middle blocks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qwen_mixed_precision as mp  # noqa: E402


class _Q:  # quantizable: defines to_quantized
    def to_quantized(self) -> None: ...


class _NoQ:  # not quantizable
    pass


def test_non_quantizable_module_is_skipped() -> None:
    assert mp.mixed_precision_predicate("transformer_blocks.0.attn.to_q", _NoQ()) is False


def test_embeddings_and_projection_kept_bf16() -> None:
    for path in (
        "img_in",
        "txt_in",
        "time_text_embed.timestep_embedder.linear_1",
        "proj_out",
        "norm_out.linear",
    ):
        assert mp.mixed_precision_predicate(path, _Q()) is False, path


def test_edge_blocks_get_q8() -> None:
    for idx in (0, 5, 54, 59):  # first 6 + last 6 of 60
        out = mp.mixed_precision_predicate(f"transformer_blocks.{idx}.attn.to_q", _Q())
        assert out == {"group_size": 64, "bits": 8}, idx


def test_middle_blocks_get_default_q4() -> None:
    for idx in (6, 30, 53):  # outside the first-6 / last-6 protected set
        assert mp.mixed_precision_predicate(f"transformer_blocks.{idx}.attn.to_q", _Q()) is True, idx
