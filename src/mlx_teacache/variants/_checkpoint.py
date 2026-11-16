"""mflux-free checkpoint rule shared by every variant: does the model's config name the checkpoint the variant's
built-in coefficients were fitted on, or declare it as its base? apply_teacache warns when it does not, and
match_variant reports the same answer as VariantInfo.calibrated."""


def same_repo_id(name: object, repo_id: str) -> bool:
    """True when `name` is the Hugging Face repo id `repo_id`. The Hub resolves repo ids case-insensitively and mflux
    accepts any casing (`--model qwen/qwen-image` loads the original checkpoint), so the comparison ignores case."""
    return isinstance(name, str) and name.casefold() == repo_id.casefold()


def is_calibrated_checkpoint(model_config: object, calibrated: str) -> bool:
    """True unless the model reports a model_name that is neither the calibrated checkpoint nor declares it as its
    base (repo ids compared case-insensitively, see same_repo_id). mflux resolves a local path or a pre-quantized
    mirror by copying the base config and recording the base's model_name in base_model, so such a mirror counts as
    calibrated. On mflux 0.19 and later a pre-quantized mirror resolves against the `qwen-image` alias config, which
    is Qwen-Image-2512, so it is the qwen_image_2512 variant's model (and calibrated there) even when it re-quantizes
    the original Qwen-Image."""
    loaded = getattr(model_config, "model_name", None)
    base = getattr(model_config, "base_model", None)
    return not isinstance(loaded, str) or same_repo_id(loaded, calibrated) or same_repo_id(base, calibrated)
