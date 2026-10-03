"""Weight-free support checks: does TeaCache cover this model, before anything loads?

Imports no mflux. A caller holding an mflux ModelConfig and the pipeline class it is about
to instantiate can ask here and refuse early; apply_teacache reaches the same answer on the
loaded model because both run each variant's detect.matches_config."""

from dataclasses import dataclass

from mlx_teacache.errors import TeaCacheValueError
from mlx_teacache.variants import _REGISTRY


@dataclass(frozen=True, kw_only=True, slots=True)
class VariantInfo:
    """What TeaCache knows about a supported model, available before any weights load.

    variant_id: the id apply_teacache reports as handle.variant_id.
    display_name: a readable model name.
    default_thresh: the per-variant default rel_l1_thresh, or None for the distilled
        few-step models where the gate skips nothing (apply_teacache falls back to 0.20 and warns).
    model_names: the mflux model aliases this variant is detected by. mflux accepts more
        aliases for some models (for example klein-base-4b); they resolve to the same config.
    calibrated: False when the config names a checkpoint other than the one the variant's
        coefficients were fitted on (today only Qwen-Image checks this: on mflux 0.19 and
        later the qwen-image alias loads Qwen-Image-2512). apply_teacache still patches such
        a model and warns, unless the caller passes `coefficients=`. On mflux 0.19 and later a
        pre-quantized mirror resolves against the qwen-image alias config, so it reports False
        even when it re-quantizes the original Qwen-Image. True for every other match.
    """

    variant_id: str
    display_name: str
    default_thresh: float | None
    model_names: tuple[str, ...]
    calibrated: bool


def match_variant(model_config: object, pipeline_class: type) -> VariantInfo | None:
    """The variant apply_teacache would use for a model that `pipeline_class` builds from
    `model_config`, or None when apply_teacache would raise IncompatibleModelError.

    `model_config` is an mflux ModelConfig (anything with an `aliases` list works);
    `pipeline_class` is the class mflux will instantiate, for example ZImage, not an
    instance of it. Reads no weights and imports nothing from mflux."""
    if not isinstance(pipeline_class, type):
        raise TeaCacheValueError(
            "pipeline_class must be a class such as ZImage, "
            f"got an instance of {type(pipeline_class).__name__}"
        )
    for variant_id, entry in _REGISTRY.items():
        if entry["matches_config"](model_config, pipeline_class):
            checkpoint_check = entry["is_calibrated_checkpoint"]
            calibrated = (
                True
                if checkpoint_check is None
                else bool(checkpoint_check(model_config, str(entry["META"]["hf_model_id"])))
            )
            return VariantInfo(
                variant_id=variant_id,
                display_name=str(entry["META"]["display_name"]),
                default_thresh=entry["default_thresh"],
                model_names=entry["model_names"],
                calibrated=calibrated,
            )
    return None
