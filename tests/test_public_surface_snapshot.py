"""Exact snapshot of the public names whose shape 0.16 freezes for 1.0.

`tests/test_public_api.py` pins which names the package exports; this file pins what each of them looks like:
signatures (names, kinds, defaults, annotations), dataclass fields, the handle's and the stats' properties and which of
them can be assigned, exception bases and constructors, and the warning family. A failure means a change to the
public surface: if it is intended, update the literal here in the same change and say so in the changelog.

Every expected value is a literal, never read back from the code under test. Nothing here imports mflux, so the file
runs in the pure-core lane.
"""

import dataclasses
import inspect
import typing
from collections.abc import Sequence

import mlx_teacache
import mlx_teacache.testing
from mlx_teacache import (
    AlreadyPatchedError,
    CalibrationError,
    GenerationStats,
    IncompatibleModelError,
    InternalStateError,
    InvalidStepWindowError,
    MissingGenerationContextError,
    Provenance,
    StatsFrozenError,
    TeaCacheDisabledWarning,
    TeaCacheError,
    TeaCacheHandle,
    TeaCacheNoBenefitWarning,
    TeaCacheStats,
    TeaCacheUncalibratedCheckpointWarning,
    TeaCacheUntestedMfluxWarning,
    TeaCacheValueError,
    TeaCacheWarning,
    TransformerShapeError,
)

P = inspect.Parameter
KW = P.KEYWORD_ONLY
POK = P.POSITIONAL_OR_KEYWORD
EMPTY = P.empty


def _params(fn: object) -> list[tuple[str, object, object]]:
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]  # type: ignore[arg-type]


def _typed_params(fn: object) -> list[tuple[str, object, object, object]]:
    return [
        (p.name, p.kind, p.default, p.annotation)
        for p in inspect.signature(fn).parameters.values()  # type: ignore[arg-type]
    ]


def _properties(cls: type) -> dict[str, property]:
    return {name: value for name, value in vars(cls).items() if isinstance(value, property)}


def test_testing_all_is_exact() -> None:
    """Bug: a helper is added to or dropped from mlx_teacache.testing's public names (a private import such as
    VariantPatch exported, or reset_untested_mflux_warning renamed)."""
    assert set(mlx_teacache.testing.__all__) == {"fake_handle", "reset_untested_mflux_warning"}


def test_testing_signatures() -> None:
    """Bug: a fake_handle keyword is renamed, reordered, made positional, retyped or given another default, its
    return annotation stops being TeaCacheHandle, or reset_untested_mflux_warning starts taking an argument."""
    fake_handle = mlx_teacache.testing.fake_handle
    assert _typed_params(fake_handle) == [
        ("stats", KW, None, TeaCacheStats | None),
        ("provenance", KW, None, Provenance | None),
        ("rel_l1_thresh", KW, 0.12, float),
        ("variant_id", KW, "fake", str),
        ("coefficients", KW, None, Sequence[float] | None),
        ("decision_rule_revision", KW, None, int | None),
        ("mflux_version", KW, None, str | None),
    ]
    assert inspect.signature(fake_handle).return_annotation is TeaCacheHandle
    reset = mlx_teacache.testing.reset_untested_mflux_warning
    assert _params(reset) == []
    assert inspect.signature(reset).return_annotation is None


def test_handle_property_set_and_which_are_writable() -> None:
    """Bug: a handle property is added, removed, renamed or retyped, a read-only fact (the skip window,
    decision_rule_revision, mflux_version) gains a setter, one of the five writable ones loses its setter (writing
    coefficients was a plain attribute write in 0.15, so dropping its setter turns that write into an AttributeError),
    or a public method appears beside restore()."""
    props = _properties(TeaCacheHandle)
    assert {name: prop.fget.__annotations__["return"] for name, prop in props.items()} == {
        "stats": TeaCacheStats,
        "provenance": Provenance,
        "rel_l1_thresh": float,
        "variant_id": str,
        "coefficients": tuple[float, float, float, float, float],
        "skip_first_n_steps": int,
        "skip_last_n_steps": int,
        "decision_rule_revision": int,
        "mflux_version": str | None,
    }
    assert {name for name, prop in props.items() if prop.fset is not None} == {
        "rel_l1_thresh",
        "stats",
        "provenance",
        "variant_id",
        "coefficients",
    }
    public_callables = {
        name for name, value in vars(TeaCacheHandle).items() if callable(value) and not name.startswith("_")
    }
    assert public_callables == {"restore"}
    assert callable(vars(TeaCacheHandle)["__enter__"])
    assert callable(vars(TeaCacheHandle)["__exit__"])
    assert _params(TeaCacheHandle.restore) == [("self", POK, EMPTY)]


def test_stats_public_fields_properties_and_methods() -> None:
    """Bug: a public TeaCacheStats counter is renamed, retyped, added or dropped (the deprecated cfg_fallback_steps
    turning back into a field counts), a property is added or removed or gains or loses its setter, a public method
    appears or goes, or the constructor's public positional slots move (cfg_fallback_steps keeps its 0.15 sixth slot,
    so `TeaCacheStats(0, 0, 0, 0, 0, 0, None)` still binds as it did)."""
    public_fields = {f.name: f.type for f in dataclasses.fields(TeaCacheStats) if not f.name.startswith("_")}
    assert public_fields == {
        "generations": int,
        "computed_count": int,
        "forced_count": int,
        "skipped_count": int,
        "numerical_miss_count": int,
        "last_generation": GenerationStats | None,
    }
    props = _properties(TeaCacheStats)
    assert set(props) == {"total_active_steps", "total_steps_seen", "cfg_fallback_steps", "speedup_estimate"}
    assert {name for name, prop in props.items() if prop.fset is not None} == {"cfg_fallback_steps"}
    methods = {
        name
        for name, value in vars(TeaCacheStats).items()
        if inspect.isfunction(value) and not name.startswith("_")
    }
    assert methods == {"record", "finalize_last_generation", "discard_current_generation"}

    params = _params(TeaCacheStats.__init__)
    assert params[:8] == [
        ("self", POK, EMPTY),
        ("generations", POK, 0),
        ("computed_count", POK, 0),
        ("forced_count", POK, 0),
        ("skipped_count", POK, 0),
        ("numerical_miss_count", POK, 0),
        ("cfg_fallback_steps", POK, None),
        ("last_generation", POK, None),
    ]
    # Whatever follows the public slots is private storage, never a new public argument.
    assert [name for name, _, _ in params[8:] if not name.startswith("_")] == []


def test_stats_dataclass_shape_seen_by_dataclasses_and_match() -> None:
    """Bug: the TeaCacheStats dataclass shape changes again without a changelog line. Since 0.16,
    dataclasses.fields(), dataclasses.asdict() and positional match patterns see the private
    _cfg_fallback_steps where 0.15 had cfg_fallback_steps, so `asdict(s)["cfg_fallback_steps"]` raises KeyError
    and the field order is what `case TeaCacheStats(g, c, f, s, n, cfg, last)` binds against."""
    expected = (
        "generations",
        "computed_count",
        "forced_count",
        "skipped_count",
        "numerical_miss_count",
        "_cfg_fallback_steps",
        "last_generation",
        "_staging",
        "_frozen",
    )
    assert tuple(f.name for f in dataclasses.fields(TeaCacheStats)) == expected
    assert tuple(dataclasses.asdict(TeaCacheStats())) == expected
    assert TeaCacheStats.__match_args__ == expected


def test_provenance_fields_and_defaults() -> None:
    """Bug: a Provenance field is renamed, retyped, reordered or loses its None default (so positional or keyword
    construction breaks), the class stops being frozen, or for_user_supplied() goes away or builds another source."""
    assert dataclasses.is_dataclass(Provenance)
    assert [(f.name, f.type, f.default) for f in dataclasses.fields(Provenance)] == [
        ("source", typing.Literal["builtin", "user"], dataclasses.MISSING),
        ("revision", str | None, None),
        ("calibration_dataset", str | None, None),
        ("fit_metric", str | None, None),
        ("fit_metric_value", float | None, None),
        ("reference_url", str | None, None),
        ("default_thresh", float | None, None),
    ]
    assert Provenance.__dataclass_params__.frozen is True  # type: ignore[attr-defined]
    assert _params(Provenance.for_user_supplied) == []
    assert Provenance.for_user_supplied() == Provenance(source="user")


_OWN_CONSTRUCTORS: dict[type, list[tuple[str, object, object]]] = {
    IncompatibleModelError: [
        ("self", POK, EMPTY),
        ("actual_type", KW, EMPTY),
        ("actual_model_name", KW, EMPTY),
        ("supported", KW, EMPTY),
    ],
    AlreadyPatchedError: [
        ("self", POK, EMPTY),
        ("variant_id", KW, EMPTY),
        ("rel_l1_thresh", KW, EMPTY),
    ],
    CalibrationError: [
        ("self", POK, EMPTY),
        ("variant_id", KW, EMPTY),
        ("reason", KW, EMPTY),
    ],
    TransformerShapeError: [
        ("self", POK, EMPTY),
        ("step_idx", POK, EMPTY),
        ("expected", POK, EMPTY),
        ("actual", POK, EMPTY),
    ],
    InvalidStepWindowError: [
        ("self", POK, EMPTY),
        ("skip_first", KW, EMPTY),
        ("skip_last", KW, EMPTY),
        ("num_steps", KW, None),
        ("nominal_num_inference_steps", KW, None),
        ("active_num_steps", KW, None),
    ],
    MissingGenerationContextError: [
        ("self", POK, EMPTY),
        ("detail", POK, None),
    ],
}

# These four keep Exception's constructor: one positional message, nothing else of their own.
_EXCEPTION_CONSTRUCTORS = (TeaCacheError, TeaCacheValueError, InternalStateError, StatsFrozenError)


def test_exception_bases_and_constructor_signatures() -> None:
    """Bug: an exception changes its direct base (for example TeaCacheValueError stops being a ValueError, or one
    leaves the TeaCacheError family), a constructor keyword is renamed, made positional or loses its default
    (InvalidStepWindowError's num_steps=None alias included), or one of the plain-message exceptions gains a
    constructor of its own."""
    assert {cls: cls.__bases__ for cls in (*_EXCEPTION_CONSTRUCTORS, *_OWN_CONSTRUCTORS)} == {
        TeaCacheError: (Exception,),
        TeaCacheValueError: (TeaCacheError, ValueError),
        InternalStateError: (TeaCacheError,),
        StatsFrozenError: (TeaCacheError,),
        IncompatibleModelError: (TeaCacheError,),
        AlreadyPatchedError: (TeaCacheError,),
        CalibrationError: (TeaCacheError,),
        TransformerShapeError: (TeaCacheError,),
        InvalidStepWindowError: (TeaCacheError,),
        MissingGenerationContextError: (TeaCacheError,),
    }
    for cls, expected in _OWN_CONSTRUCTORS.items():
        assert "__init__" in vars(cls), cls.__name__
        assert _params(vars(cls)["__init__"]) == expected, cls.__name__
    for cls in _EXCEPTION_CONSTRUCTORS:
        own = [
            base.__name__
            for base in cls.__mro__
            if base.__module__.startswith("mlx_teacache") and "__init__" in vars(base)
        ]
        assert own == [], f"{cls.__name__} gets a constructor from {own}"
        assert cls("message").args == ("message",)


def test_warning_bases() -> None:
    """Bug: TeaCacheWarning stops being a UserWarning or becomes a DeprecationWarning (so the default filters and
    `-W error::DeprecationWarning` treat run-time warnings as deprecations), or one of the four run-time warnings
    leaves the family that `filterwarnings("ignore", category=TeaCacheWarning)` silences."""
    assert TeaCacheWarning.__bases__ == (UserWarning,)
    assert not issubclass(TeaCacheWarning, DeprecationWarning)
    assert {
        cls: cls.__bases__
        for cls in (
            TeaCacheDisabledWarning,
            TeaCacheNoBenefitWarning,
            TeaCacheUncalibratedCheckpointWarning,
            TeaCacheUntestedMfluxWarning,
        )
    } == {
        TeaCacheDisabledWarning: (TeaCacheWarning,),
        TeaCacheNoBenefitWarning: (TeaCacheWarning,),
        TeaCacheUncalibratedCheckpointWarning: (TeaCacheWarning,),
        TeaCacheUntestedMfluxWarning: (TeaCacheWarning,),
    }
