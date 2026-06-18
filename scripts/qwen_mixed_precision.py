"""Mixed-precision quantization for Qwen-Image (DEV TOOLING — NOT shipped).

Stock uniform q4 over-quantizes Qwen-Image's quantization-sensitive layers and
produces a grainy "low-res JPEG" texture on a 32 GB Mac. Protecting the first/last
transformer blocks (q8) plus the embeddings + final projection (bf16) clears the
artifact while still fitting 32 GB (~+1.9 GB over uniform q4). Edge-block sensitivity
to low-bit quantization is the documented mitigation (cf. MixDiT / ViDiT-Q and the
Qwen-Image GGUF community builds that protect edge blocks at higher precision).

mlx-teacache itself stays quant-agnostic: this runs at MODEL-CONSTRUCTION time
(before apply_teacache), via mflux's `QwenWeightDefinition.quantization_predicate`
hook. MLX's nn.quantize honors a per-layer class_predicate return:
  False -> keep full precision (bf16);  {"bits": 8} -> q8;  True -> default bits (q4).

The same predicate is documented for users in docs/variants/qwen-image.md so they
can reproduce the showcase quality; this module is the copy our bench/sweep import.
"""

from typing import Any

N_PROTECT = 6  # first N + last N transformer blocks kept at q8
N_BLOCKS = 60
_PROTECT_BLOCKS = set(range(N_PROTECT)) | set(range(N_BLOCKS - N_PROTECT, N_BLOCKS))
_BF16_PREFIXES = ("img_in", "txt_in", "time_text_embed", "proj_out", "norm_out")


def mixed_precision_predicate(path: str, module: Any) -> Any:
    """nn.quantize class_predicate: bf16 for embeddings/projection, q8 for the
    edge transformer blocks, q4 (default) for the middle blocks."""
    if not hasattr(module, "to_quantized"):
        return False
    for prefix in _BF16_PREFIXES:
        if path == prefix or path.startswith(prefix + "."):
            return False  # full precision (bf16)
    if path.startswith("transformer_blocks."):
        try:
            idx = int(path.split(".")[1])
        except (IndexError, ValueError):
            idx = -1
        if idx in _PROTECT_BLOCKS:
            return {"group_size": 64, "bits": 8}  # q8
    return True  # default bits (q4) for the middle blocks


def enable_qwen_mixed_precision() -> None:
    """Install the mixed-precision predicate. Call BEFORE constructing
    QwenImage(quantize=4). Process-global monkeypatch (mflux exposes no per-instance
    hook in 0.17.x); fine for a one-shot generation process."""
    from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

    QwenWeightDefinition.quantization_predicate = staticmethod(mixed_precision_predicate)
