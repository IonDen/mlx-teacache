"""The slice of mlx-teacache's public API that the `mflux-teacache` plugin calls, pinned by signature and field.

The plugin is being built at https://github.com/mflux-community/mflux-plugins (folder `teacache/`, PyPI
`mflux-teacache`). It will depend on `mlx-teacache[mflux]`, so a later mlx-teacache release can reach installed
plugins through `pip install -U`. `tests/test_public_api.py` only pins that the names exist; this file pins what the
plugin will read off them: parameter names, kinds and defaults, dataclass fields, the decision vocabulary and the
warning base classes.

A failure here means a change would break mflux-teacache users. Keep the old shape alongside the new one, or ship a
matching plugin release first.

Expected values are written out as literals, not read back from the code under test. The one exception is in the
match_variant test: `default_thresh` and `display_name` are compared with the variant's own config, because the
contract is that they are passed through, not what they are. Nothing here imports mflux, so the file runs in the
pure-core lane.
"""

import dataclasses
import inspect
import types
import typing

import pytest

import mlx_teacache
from mlx_teacache._kernel.stats import Decision
from mlx_teacache.handle import TeaCacheHandle, VariantPatch

P = inspect.Parameter


def _params(fn: object) -> list[tuple[str, object, object]]:
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]  # type: ignore[arg-type]


def _mflux_class(name: str) -> type:
    return type(name, (), {"__module__": f"mflux.models.fake.{name.lower()}"})


def test_apply_teacache_parameters_are_the_four_keywords() -> None:
    """Red if `skip_first_n_steps` loses its `*`-only position, or any keyword is renamed or its default changed.

    Breaks: the plugin calls `apply_teacache(model, rel_l1_thresh=..., coefficients=..., skip_first_n_steps=...,
    skip_last_n_steps=...)` and would raise TypeError or silently run with other defaults."""
    assert _params(mlx_teacache.apply_teacache) == [
        ("flux", P.POSITIONAL_OR_KEYWORD, P.empty),
        ("rel_l1_thresh", P.KEYWORD_ONLY, None),
        ("coefficients", P.KEYWORD_ONLY, None),
        ("skip_first_n_steps", P.KEYWORD_ONLY, 1),
        ("skip_last_n_steps", P.KEYWORD_ONLY, 1),
    ]


def test_match_variant_takes_config_and_class_and_returns_variant_info_or_none() -> None:
    """Red if `pipeline_class` is renamed, made keyword-only, or the return annotation drops `| None`.

    Breaks: the plugin decides before loading weights with `match_variant(model_config, pipeline_class)` and
    treats None as "unsupported model"."""
    assert _params(mlx_teacache.match_variant) == [
        ("model_config", P.POSITIONAL_OR_KEYWORD, P.empty),
        ("pipeline_class", P.POSITIONAL_OR_KEYWORD, P.empty),
    ]
    assert inspect.signature(mlx_teacache.match_variant).return_annotation == (
        mlx_teacache.VariantInfo | None
    )


def test_match_variant_answers_for_a_supported_and_an_unsupported_model() -> None:
    """Red if a supported config stops yielding a VariantInfo, or an unsupported one yields anything but None.

    Breaks: the plugin refuses unsupported models early on `None` and shows `display_name` / `default_thresh` for
    supported ones."""
    config = types.SimpleNamespace(aliases=["z-image", "zimage"], model_name=None)
    info = mlx_teacache.match_variant(config, _mflux_class("ZImage"))
    assert info is not None
    # The default threshold and display name come from the variant's own config: a recalibrated default reaches the
    # plugin through this field, so its value is not part of the contract, only that it is passed through.
    from mlx_teacache.variants.z_image_base import config as z_image_config

    assert (info.variant_id, info.model_names, info.calibrated) == (
        "z-image-base",
        ("z-image", "zimage"),
        True,
    )
    assert info.default_thresh == z_image_config.DEFAULT_THRESH
    assert info.display_name == z_image_config.META["display_name"]
    unknown = types.SimpleNamespace(aliases=["no-such-model"], model_name=None)
    assert mlx_teacache.match_variant(unknown, _mflux_class("ZImage")) is None


def test_variant_info_is_frozen_keyword_only_with_exactly_five_typed_fields() -> None:
    """Red if a field is renamed (for example `calibrated`), added, removed, retyped, or the class stops being
    frozen or keyword-only.

    Breaks: the plugin reads `.variant_id`, `.display_name`, `.default_thresh`, `.model_names` and `.calibrated`
    and builds its own VariantInfo in tests by keyword."""
    info_cls = mlx_teacache.VariantInfo
    assert dataclasses.is_dataclass(info_cls)
    assert info_cls.__dataclass_params__.frozen is True  # type: ignore[attr-defined]
    fields = dataclasses.fields(info_cls)
    assert [(f.name, f.type, f.kw_only) for f in fields] == [
        ("variant_id", str, True),
        ("display_name", str, True),
        ("default_thresh", float | None, True),
        ("model_names", tuple[str, ...], True),
        ("calibrated", bool, True),
    ]
    info = info_cls(
        variant_id="v", display_name="V", default_thresh=None, model_names=("v",), calibrated=True
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.calibrated = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        info_cls("v", "V", None, ("v",), True)  # type: ignore[misc]


def test_check_step_window_signature_first_positional_rest_keyword_only() -> None:
    """Red if `skip_first_n_steps` / `skip_last_n_steps` become positional, are renamed, or their defaults move.

    Breaks: the plugin calls `check_step_window(steps, skip_first_n_steps=..., skip_last_n_steps=...,
    nominal_num_inference_steps=...)` to refuse a too-short run before loading weights."""
    assert _params(mlx_teacache.check_step_window) == [
        ("active_num_steps", P.POSITIONAL_OR_KEYWORD, P.empty),
        ("skip_first_n_steps", P.KEYWORD_ONLY, 1),
        ("skip_last_n_steps", P.KEYWORD_ONLY, 1),
        ("nominal_num_inference_steps", P.KEYWORD_ONLY, None),
    ]


@pytest.mark.parametrize("steps", [0, 1, 2])
def test_check_step_window_refuses_zero_to_two_active_steps(steps: int) -> None:
    """Red if the default window starts accepting 0-2 active steps or raises a different exception class.

    Breaks: the plugin catches `InvalidStepWindowError` to turn a too-short schedule into a clear message; a
    different class would surface as a traceback."""
    with pytest.raises(mlx_teacache.InvalidStepWindowError):
        mlx_teacache.check_step_window(steps)


def test_check_step_window_accepts_three_active_steps() -> None:
    """Red if the default window starts refusing 3 active steps (the boundary above the refused range).

    Breaks: the plugin would reject a 3-step run that TeaCache accepts (with a no-skip warning)."""
    assert mlx_teacache.check_step_window(3) is None


def test_invalid_step_window_error_is_a_teacache_error() -> None:
    """Red if InvalidStepWindowError stops deriving from TeaCacheError.

    Breaks: the plugin's `except TeaCacheError` fallback no longer catches the step-window refusal."""
    assert issubclass(mlx_teacache.InvalidStepWindowError, mlx_teacache.TeaCacheError)


def test_stats_dataclass_fields_the_plugin_reads() -> None:
    """Red if a StepDecision / GenerationStats field the plugin reads (or builds in its test fake) is renamed,
    retyped or removed, or TeaCacheStats loses `generations` / `last_generation`.

    Breaks: the plugin's report reads `stats.generations`, `stats.last_generation.num_steps` and
    `.decisions[].decision`, and its tests construct StepDecision by keyword."""
    assert [(f.name, f.type) for f in dataclasses.fields(mlx_teacache.StepDecision)] == [
        ("step_idx", int),
        ("timestep", float),
        ("rel_l1", float | None),
        ("accumulated_distance", float),
        ("decision", Decision),
    ]
    assert [(f.name, f.type) for f in dataclasses.fields(mlx_teacache.GenerationStats)] == [
        ("num_steps", int),
        ("cfg_was_active", bool),
        ("decisions", tuple[mlx_teacache.StepDecision, ...]),
    ]
    stats_fields = {f.name: f.type for f in dataclasses.fields(mlx_teacache.TeaCacheStats)}
    assert stats_fields["generations"] is int
    assert stats_fields["last_generation"] == (mlx_teacache.GenerationStats | None)


def test_decision_vocabulary_is_the_five_known_strings() -> None:
    """Red if a decision string is renamed, added or removed.

    Breaks: the plugin maps each `StepDecision.decision` string to a report label and counts "skipped"; an
    unknown or renamed string is miscounted."""
    assert set(typing.get_args(Decision)) == {
        "computed",
        "forced",
        "skipped",
        "numerical-miss",
        "cfg-fallback",
    }


def test_stats_commit_exposes_what_the_plugin_reads() -> None:
    """Red if recording and finalizing a generation stops exposing `generations`, `last_generation.num_steps`
    and `last_generation.decisions[0].decision` with these values.

    Breaks: the plugin's post-run report would show zero generations or no step decisions."""
    stats = mlx_teacache.TeaCacheStats()
    stats.record(
        mlx_teacache.StepDecision(
            step_idx=0, timestep=1.0, rel_l1=None, accumulated_distance=0.0, decision="forced"
        )
    )
    stats.finalize_last_generation(num_inference_steps=1, cfg_was_active=False)
    assert stats.generations == 1
    assert stats.last_generation is not None
    assert stats.last_generation.num_steps == 1
    assert stats.last_generation.decisions[0].decision == "forced"


def test_handle_exposes_stats_threshold_and_accepts_the_plugins_constructor_keywords() -> None:
    """Red if TeaCacheHandle loses `.stats` / `.rel_l1_thresh`, or its constructor keywords (`patch`, `stats`,
    `provenance`, `rel_l1_thresh`, keyword-only) change.

    Breaks: the plugin reads `handle.stats` and `handle.rel_l1_thresh`, and its test fake builds a handle with
    those keywords. The constructor and VariantPatch are non-public and pinned as test support only."""
    assert _params(TeaCacheHandle.__init__) == [
        ("self", P.POSITIONAL_OR_KEYWORD, P.empty),
        ("patch", P.KEYWORD_ONLY, P.empty),
        ("stats", P.KEYWORD_ONLY, P.empty),
        ("provenance", P.KEYWORD_ONLY, P.empty),
        ("rel_l1_thresh", P.KEYWORD_ONLY, P.empty),
    ]
    assert [f.name for f in dataclasses.fields(VariantPatch)] == ["rollbacks", "finalizers", "on_restored"]
    stats = mlx_teacache.TeaCacheStats()
    handle = TeaCacheHandle(
        patch=VariantPatch(),
        stats=stats,
        provenance=mlx_teacache.Provenance.for_user_supplied(),
        rel_l1_thresh=0.25,
    )
    assert handle.stats is stats
    assert handle.rel_l1_thresh == 0.25
    handle.restore()


@pytest.mark.parametrize(
    "name",
    ["TeaCacheUncalibratedCheckpointWarning", "TeaCacheUntestedMfluxWarning"],
)
def test_plugin_filtered_warnings_are_user_warning_subclasses(name: str) -> None:
    """Red if the warning class stops deriving from UserWarning (for example becomes a bare Warning).

    Breaks: the plugin and its users filter these by category and rely on the default UserWarning display; a
    different base changes which `-W` / `filterwarnings` rules catch them."""
    cls = getattr(mlx_teacache, name)
    assert issubclass(cls, UserWarning)


def test_version_is_a_string() -> None:
    """Red if `__version__` is removed or stops being a str.

    Breaks: the plugin prints the mlx-teacache version in its report header."""
    assert isinstance(mlx_teacache.__version__, str)
    assert mlx_teacache.__version__ != ""
