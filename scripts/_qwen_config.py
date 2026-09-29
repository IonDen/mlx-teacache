"""The Qwen-Image ModelConfig every Qwen script loads through.

On mflux 0.19+ ``ModelConfig.qwen_image()`` and the ``qwen-image`` alias resolve to Qwen-Image-2512, which has no
calibration and is ~58 GB. The calibrated checkpoint is the original ``Qwen/Qwen-Image``, so it is named explicitly
(``from_name`` resolves it on mflux 0.17.5 through 0.20 alike). Only mflux is imported, and only inside the function.
"""

from typing import Any

QWEN_ORIGINAL = "Qwen/Qwen-Image"


def qwen_config(name: str = QWEN_ORIGINAL) -> Any:
    """Resolve ``name`` to a Qwen-Image family config.

    The original checkpoint gets the strict identity check (its name, plus the ``qwen-image`` alias the variant
    detector relies on). Any other name (say Qwen-Image-2512 for a user's own calibration) only has to be a
    Qwen-Image family config."""
    from mflux.models.common.config.model_config import ModelConfig

    config = ModelConfig.from_name(name)
    if name == QWEN_ORIGINAL:
        if config.model_name != QWEN_ORIGINAL or "qwen-image" not in config.aliases:
            raise RuntimeError(
                f"expected the original {QWEN_ORIGINAL} with the 'qwen-image' alias, resolved "
                f"{config.model_name!r} with aliases {list(config.aliases)!r}"
            )
    elif "qwen-image" not in config.aliases:
        raise RuntimeError(
            f"{name!r} resolved to {config.model_name!r}, which is not a Qwen-Image family config"
        )
    return config


def qwen_original_config() -> Any:
    return qwen_config(QWEN_ORIGINAL)
