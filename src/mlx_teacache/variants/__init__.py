"""Variant registry. Walks every subpackage of variants/ at import time
to populate _REGISTRY with (META, matches, load_integration) entries.

config.py + detect.py are imported eagerly (they must be mflux-free).
integration.py is loaded lazily via load_integration() — apply_teacache
calls it after detect picks the winning variant. This is the contract
that keeps `import mlx_teacache` working without the [mflux] extra.
"""

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any, TypedDict, cast

from mlx_teacache._kernel.coefficients import validate_custom
from mlx_teacache._kernel.resolve import ThreshPolicy
from mlx_teacache.errors import CalibrationError, TeaCacheValueError
from mlx_teacache.variants._checkpoint import is_calibrated_checkpoint


class _RegistryEntry(TypedDict):
    META: dict[str, Any]
    matches: Callable[[object], bool]
    matches_config: Callable[[object, type], bool]
    model_names: tuple[str, ...]
    hf_model_id: str
    is_calibrated_checkpoint: Callable[[object, str], bool]
    load_integration: Callable[[], Callable[..., Any]]
    default_thresh: float | None
    non_distilled: bool
    thresh_policy: ThreshPolicy


# The variants whose integration keeps the per-variant default threshold when the caller passes
# coefficients; every other variant falls back to the package default. Mirrors the policy literal at
# each integration's resolve_rule call, so apply_teacache can name the threshold before it patches.
_VARIANT_DEFAULT_WITH_CALLER_COEFFICIENTS: frozenset[str] = frozenset(
    {"flux1-dev", "flux1-schnell", "flux1-krea-dev", "flux2-klein-4b", "flux2-klein-9b"}
)


def _thresh_policy(variant_id: str) -> ThreshPolicy:
    if variant_id in _VARIANT_DEFAULT_WITH_CALLER_COEFFICIENTS:
        return "variant-default"
    return "fallback-for-caller-coefficients"


_REGISTRY: dict[str, _RegistryEntry] = {}


def _make_lazy_loader(module_name: str) -> Callable[[], Callable[..., Any]]:
    def _load() -> Callable[..., Any]:
        integration = importlib.import_module(f"{module_name}.integration")
        return cast(Callable[..., Any], integration.apply)

    return _load


_REQUIRED_META_KEYS = ("variant_id", "display_name", "license", "non_distilled")


def _validate_meta(meta: object, *, subname: str) -> dict[str, Any]:
    """Validate a variant's META mapping, raising a CalibrationError that names
    the subpackage so a malformed variant can't fail `import mlx_teacache` with an
    opaque AttributeError/KeyError."""
    if not isinstance(meta, dict):
        raise CalibrationError(
            variant_id=subname,
            reason=f"variant {subname!r} has a missing or non-dict META",
        )
    missing = [key for key in _REQUIRED_META_KEYS if key not in meta]
    if missing:
        raise CalibrationError(
            variant_id=str(meta.get("variant_id", subname)),
            reason=f"variant {subname!r} META is missing required key(s): {missing}",
        )
    if not isinstance(meta["non_distilled"], bool):
        raise CalibrationError(
            variant_id=str(meta["variant_id"]),
            reason=f"variant {subname!r} META['non_distilled'] must be a bool, got {meta['non_distilled']!r}",
        )
    return meta


_ConfigPredicate = Callable[[object, type], bool]
_CheckpointCheck = Callable[[object, str], bool]


def _detect_contract(
    detect: object, meta: dict[str, Any], *, variant_id: str
) -> tuple[_ConfigPredicate, tuple[str, ...], _CheckpointCheck]:
    """The config-level predicate and model names a detect module must define, plus the checkpoint check (the detect
    module's own is_calibrated_checkpoint if it defines one, else the shared rule), or a CalibrationError naming the
    variant. Every variant needs a non-empty META['hf_model_id'] for the check to compare against."""
    matches_config = getattr(detect, "matches_config", None)
    model_names = getattr(detect, "MODEL_NAMES", None)
    checkpoint_check = getattr(detect, "is_calibrated_checkpoint", None)
    if not callable(matches_config):
        raise CalibrationError(
            variant_id=variant_id, reason="detect.py defines no matches_config(model_config, pipeline_class)"
        )
    if not (
        isinstance(model_names, tuple) and model_names and all(isinstance(n, str) and n for n in model_names)
    ):
        raise CalibrationError(
            variant_id=variant_id,
            reason=f"detect.py MODEL_NAMES must be a non-empty tuple of non-empty names, got {model_names!r}",
        )
    hf_model_id = meta.get("hf_model_id")
    if not (isinstance(hf_model_id, str) and hf_model_id):
        raise CalibrationError(
            variant_id=variant_id,
            reason="META['hf_model_id'] must name the checkpoint the coefficients were calibrated on, "
            f"got hf_model_id={hf_model_id!r}",
        )
    if checkpoint_check is None:
        checkpoint_check = is_calibrated_checkpoint
    elif not callable(checkpoint_check):
        raise CalibrationError(
            variant_id=variant_id,
            reason=f"detect.py is_calibrated_checkpoint must be callable, got {checkpoint_check!r}",
        )
    return (
        cast(_ConfigPredicate, matches_config),
        cast("tuple[str, ...]", model_names),
        cast(_CheckpointCheck, checkpoint_check),
    )


def checkpoint_is_calibrated(entry: _RegistryEntry, model_config: object) -> bool:
    """True unless `model_config` names a checkpoint that is neither the one the entry's built-in coefficients were
    fitted on nor declares it as its base. apply_teacache warns on False and match_variant reports it as
    VariantInfo.calibrated, so the two cannot disagree. A config without a model_name is not checked (True)."""
    return bool(entry["is_calibrated_checkpoint"](model_config, entry["hf_model_id"]))


def _build_one(full: str, subname: str) -> tuple[str, _RegistryEntry]:
    """Import + validate a single variant subpackage. Any failure is surfaced as
    a CalibrationError naming the subpackage, so one broken variant can't take
    down the whole registry with an opaque error."""
    try:
        config = importlib.import_module(f"{full}.config")
        detect = importlib.import_module(f"{full}.detect")
    except Exception as e:
        raise CalibrationError(
            variant_id=subname,
            reason=f"failed to import variant subpackage {subname!r}: {type(e).__name__}: {e}",
        ) from e

    meta = _validate_meta(getattr(config, "META", None), subname=subname)
    variant_id = str(meta["variant_id"])

    coeffs = getattr(config, "COEFFICIENTS", None)
    if coeffs is None:
        raise CalibrationError(
            variant_id=variant_id,
            reason="COEFFICIENTS attribute is missing from variant config",
        )
    try:
        validate_custom(coeffs)
    except TeaCacheValueError as e:
        raise CalibrationError(variant_id=variant_id, reason=str(e)) from e

    default_thresh = cast("float | None", getattr(config, "DEFAULT_THRESH", None))

    matches_config, model_names, checkpoint_check = _detect_contract(detect, meta, variant_id=variant_id)
    return variant_id, _RegistryEntry(
        META=meta,
        matches=detect.matches,
        matches_config=matches_config,
        model_names=model_names,
        hf_model_id=str(meta["hf_model_id"]),
        is_calibrated_checkpoint=checkpoint_check,
        load_integration=_make_lazy_loader(full),
        default_thresh=default_thresh,
        non_distilled=meta["non_distilled"],
        thresh_policy=_thresh_policy(variant_id),
    )


def _register(
    registry: dict[str, _RegistryEntry],
    sources: dict[str, str],
    variant_id: str,
    entry: _RegistryEntry,
    *,
    subname: str,
) -> None:
    """Add one entry, refusing a variant_id another subpackage already declared
    (a copied subpackage must not silently overwrite the original)."""
    if variant_id in registry:
        raise CalibrationError(
            variant_id=variant_id,
            reason=f"subpackages {sources[variant_id]!r} and {subname!r} both declare this variant_id",
        )
    registry[variant_id] = entry
    sources[variant_id] = subname


def _build_registry() -> None:
    package = importlib.import_module(__name__)
    sources: dict[str, str] = {}
    for _, subname, ispkg in pkgutil.iter_modules(package.__path__):
        if not ispkg:
            continue
        variant_id, entry = _build_one(f"{__name__}.{subname}", subname)
        _register(_REGISTRY, sources, variant_id, entry, subname=subname)


_build_registry()

__all__ = ["_REGISTRY", "checkpoint_is_calibrated"]
