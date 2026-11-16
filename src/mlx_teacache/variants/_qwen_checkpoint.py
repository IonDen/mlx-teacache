"""mflux-free checkpoint rules shared by the two Qwen-Image variants and the shared Qwen forward.

mflux 0.19 and later give every Qwen-Image text-to-image config the Qwen-Image-2512 aliases too, including the
original checkpoint loaded by its repo id (mflux resolves `Qwen/Qwen-Image` as a mirror of the `qwen-image` config,
recording base_model Qwen/Qwen-Image-2512). `is_2512_checkpoint` holds the one rule that splits the two variants:
2512 aliases mean Qwen-Image-2512 unless model_name is the original checkpoint. The qwen_image variant matches only
configs the rule rejects, the qwen_image_2512 variant only configs it accepts, so a config never matches both.

mflux 0.22 also builds Qwen-Image-Edit-2511 with `zero_cond_t` (its config sets
`transformer_overrides={"zero_cond_t": True}`): the reference-image tokens take the timestep-0 modulation. The gated
forward does not implement that path, so neither variant matches a config with it on.

The checkpoint rule every variant shares (`same_repo_id`, `is_calibrated_checkpoint`) lives in `variants/_checkpoint.py`;
the two Qwen detectors re-export `is_calibrated_checkpoint` from here.
"""

from mlx_teacache.variants._checkpoint import is_calibrated_checkpoint, same_repo_id

# The checkpoint the qwen_image variant's coefficients were fitted on, and the aliases mflux 0.19+ adds for
# Qwen-Image-2512.
ORIGINAL_MODEL_NAME = "Qwen/Qwen-Image"
ALIASES_2512: tuple[str, ...] = ("qwen-image-2512", "qwen-2512")


def has_zero_cond_t(model_config: object) -> bool:
    """True when the config builds the transformer with zero_cond_t (Qwen-Image-Edit-2511), which neither Qwen
    variant's gated forward implements."""
    overrides = getattr(model_config, "transformer_overrides", None) or {}
    return bool(overrides.get("zero_cond_t"))


def is_2512_checkpoint(model_config: object) -> bool:
    """True when the config carries a Qwen-Image-2512 alias and does not name the original checkpoint (in any
    casing)."""
    aliases = getattr(model_config, "aliases", None) or []
    return any(name in aliases for name in ALIASES_2512) and not same_repo_id(
        getattr(model_config, "model_name", None), ORIGINAL_MODEL_NAME
    )


__all__ = [
    "ALIASES_2512",
    "ORIGINAL_MODEL_NAME",
    "has_zero_cond_t",
    "is_2512_checkpoint",
    "is_calibrated_checkpoint",
    "same_repo_id",
]
