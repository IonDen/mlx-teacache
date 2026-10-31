"""Every typed exception must survive pickle and copy (e.g. raised in a worker process)."""

import copy
import inspect
import pickle
import warnings

import pytest

from mlx_teacache import errors as E

CASES = [
    E.TeaCacheError("x"),
    E.IncompatibleModelError(actual_type="X", actual_model_name=None, supported=["a", "b"]),
    E._StepCacheConflictError(
        actual_type="X", actual_model_name=None, supported=[], requested="step_cache_ratio=0.5"
    ),
    E.AlreadyPatchedError(variant_id="flux1-dev", rel_l1_thresh=0.2),
    E.CalibrationError(variant_id="v", reason="r"),
    E.TransformerShapeError(3, (1, 2), (1, 3)),
    E.InvalidStepWindowError(skip_first=1, skip_last=1, active_num_steps=2),
    E.InvalidStepWindowError(skip_first=2, skip_last=2, nominal_num_inference_steps=4, active_num_steps=3),
    E.MissingGenerationContextError("callback gone"),
    E.MissingGenerationContextError(),
    E.InternalStateError("x"),
    E.TeaCacheValueError("x"),
]


def _raisable_exception_classes() -> list[type]:
    # Warning classes (UserWarning subclasses) are excluded on purpose: they are emitted via
    # warnings.warn, never raised, and take only a message, so they have no keyword-only state.
    return [
        cls
        for _, cls in inspect.getmembers(E, inspect.isclass)
        if issubclass(cls, Exception) and not issubclass(cls, Warning) and cls.__module__ == E.__name__
    ]


@pytest.mark.parametrize("exc", CASES, ids=lambda e: f"{type(e).__name__}-{CASES.index(e)}")
def test_round_trips_through_pickle_and_copy(exc) -> None:
    """Bug caught: kw-only __init__ + args==(message,) -> TypeError on unpickle in a worker process."""
    for clone in (pickle.loads(pickle.dumps(exc)), copy.copy(exc)):
        assert type(clone) is type(exc)
        assert str(clone) == str(exc)


def test_every_exported_exception_class_has_a_pickling_case() -> None:
    """Bug caught: a new exception class is added without a pickling case and breaks in workers."""
    covered = {type(exc) for exc in CASES}
    missing = [cls.__name__ for cls in _raisable_exception_classes() if cls not in covered]
    assert missing == []


def test_public_attributes_survive_the_round_trip() -> None:
    """Bug caught: reduce rebuilds from the message and drops the typed attributes."""
    with pytest.warns(DeprecationWarning, match="active_num_steps"):
        window = E.InvalidStepWindowError(
            skip_first=2, skip_last=2, num_steps=3, nominal_num_inference_steps=4, active_num_steps=5
        )
    clone = pickle.loads(pickle.dumps(window))
    assert (clone.skip_first, clone.skip_last, clone.num_steps) == (2, 2, 3)
    assert (clone.nominal_num_inference_steps, clone.active_num_steps) == (4, 5)
    shape = pickle.loads(pickle.dumps(E.TransformerShapeError(3, (1, 2), (1, 3))))
    assert (shape.step_idx, shape.expected, shape.actual) == (3, (1, 2), (1, 3))
    model = pickle.loads(
        pickle.dumps(E.IncompatibleModelError(actual_type="X", actual_model_name="m", supported=["a"]))
    )
    assert (model.actual_type, model.actual_model_name, model.supported) == ("X", "m", ["a"])


def _window_attributes(exc: E.InvalidStepWindowError) -> tuple[object, ...]:
    return (
        str(exc),
        exc.skip_first,
        exc.skip_last,
        exc.num_steps,
        exc.nominal_num_inference_steps,
        exc.active_num_steps,
    )


def test_unpickling_an_alias_built_error_does_not_warn() -> None:
    """Bug caught: the unpickle path replays the deprecated num_steps= keyword, so every
    worker that receives this error warns again (or fails under an error filter)."""
    with pytest.warns(DeprecationWarning, match="active_num_steps"):
        exc = E.InvalidStepWindowError(skip_first=1, skip_last=2, num_steps=3)
    data = pickle.dumps(exc)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clones = [pickle.loads(data), copy.copy(exc), copy.deepcopy(exc)]
    for clone in clones:
        assert type(clone) is E.InvalidStepWindowError
        assert _window_attributes(clone) == _window_attributes(exc)


def test_alias_and_preferred_keyword_round_trip_with_both_values() -> None:
    with pytest.warns(DeprecationWarning, match="active_num_steps"):
        exc = E.InvalidStepWindowError(skip_first=2, skip_last=2, num_steps=3, active_num_steps=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clone = pickle.loads(pickle.dumps(exc))
    assert _window_attributes(clone) == _window_attributes(exc)
    assert (clone.num_steps, clone.active_num_steps) == (3, 5)


@pytest.mark.parametrize("exc", CASES, ids=lambda e: f"{type(e).__name__}-{CASES.index(e)}")
def test_notes_and_extra_attributes_survive_pickle_and_copy(exc) -> None:
    """Bug caught: a custom __reduce__ returns no state, so add_note() notes and
    attributes a caller or subclass set after construction vanish in a worker
    process or a copy."""
    fresh = pickle.loads(pickle.dumps(exc))
    message = str(fresh)
    fresh.add_note("while loading model A")
    fresh.request_id = "r-17"
    for clone in (pickle.loads(pickle.dumps(fresh)), copy.copy(fresh)):
        assert clone.__notes__ == ["while loading model A"]
        assert clone.request_id == "r-17"
        assert str(clone) == message


class _Legacy015Window:
    """Pickles exactly as a 0.15 InvalidStepWindowError did: _rebuild with the 0.15 constructor mapping, which
    carries num_steps= (and active_num_steps=None when only num_steps was passed), plus the instance __dict__."""

    def __init__(self, ctor: dict[str, object], state: dict[str, object]) -> None:
        self._ctor = ctor
        self._state = state

    def __reduce__(self) -> tuple[object, ...]:
        return (E._rebuild, (E.InvalidStepWindowError, dict(self._ctor)), dict(self._state))


@pytest.mark.parametrize(
    ("ctor", "message", "attributes"),
    [
        (
            {
                "skip_first": 1,
                "skip_last": 2,
                "num_steps": 3,
                "nominal_num_inference_steps": None,
                "active_num_steps": None,
            },
            "skip_first_n_steps + skip_last_n_steps must be < active denoising steps. "
            "Got skip_first=1, skip_last=2, active_num_steps=3 (sum 3 >= 3).",
            (1, 2, 3, None, 3),
        ),
        (
            {
                "skip_first": 2,
                "skip_last": 2,
                "num_steps": 3,
                "nominal_num_inference_steps": 4,
                "active_num_steps": None,
            },
            "skip_first_n_steps + skip_last_n_steps must be < active denoising steps; "
            "got skip_first=2, skip_last=2, active_num_steps=3, nominal_num_inference_steps=4 (sum 4 >= 3).",
            (2, 2, 3, 4, 3),
        ),
        (
            {
                "skip_first": 2,
                "skip_last": 2,
                "num_steps": 3,
                "nominal_num_inference_steps": None,
                "active_num_steps": 4,
            },
            "skip_first_n_steps + skip_last_n_steps must be < active denoising steps. "
            "Got skip_first=2, skip_last=2, active_num_steps=4 (sum 4 >= 4).",
            (2, 2, 3, None, 4),
        ),
    ],
    ids=["num_steps-only", "with-nominal", "both-keywords"],
)
def test_a_0_15_pickle_restores_without_a_warning(ctor, message, attributes) -> None:
    """Bug caught: a worker on this release unpickles an error a 0.15 process pickled, the stored num_steps= is
    replayed into the deprecated keyword, and the load warns (or raises under an error filter)."""
    skip_first, skip_last, num_steps, nominal, active = attributes
    state = {
        "skip_first": skip_first,
        "skip_last": skip_last,
        "num_steps": num_steps,
        "nominal_num_inference_steps": nominal,
        "active_num_steps": active,
        "_ctor": dict(ctor),
    }
    data = pickle.dumps(_Legacy015Window(ctor, state))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clone = pickle.loads(data)
        again = pickle.loads(pickle.dumps(clone))
    for restored in (clone, again):
        assert type(restored) is E.InvalidStepWindowError
        assert _window_attributes(restored) == (message, *attributes)
