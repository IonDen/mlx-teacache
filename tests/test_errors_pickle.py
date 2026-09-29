"""Every typed exception must survive pickle and copy (e.g. raised in a worker process)."""

import copy
import inspect
import pickle

import pytest

from mlx_teacache import errors as E

CASES = [
    E.TeaCacheError("x"),
    E.IncompatibleModelError(actual_type="X", actual_model_name=None, supported=["a", "b"]),
    E.AlreadyPatchedError(variant_id="flux1-dev", rel_l1_thresh=0.2),
    E.CalibrationError(variant_id="v", reason="r"),
    E.TransformerShapeError(3, (1, 2), (1, 3)),
    E.InvalidStepWindowError(skip_first=1, skip_last=1, num_steps=2),
    E.InvalidStepWindowError(
        skip_first=2, skip_last=2, num_steps=3, nominal_num_inference_steps=4, active_num_steps=3
    ),
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
    clone = pickle.loads(
        pickle.dumps(
            E.InvalidStepWindowError(
                skip_first=2, skip_last=2, num_steps=3, nominal_num_inference_steps=4, active_num_steps=5
            )
        )
    )
    assert (clone.skip_first, clone.skip_last, clone.num_steps) == (2, 2, 3)
    assert (clone.nominal_num_inference_steps, clone.active_num_steps) == (4, 5)
    shape = pickle.loads(pickle.dumps(E.TransformerShapeError(3, (1, 2), (1, 3))))
    assert (shape.step_idx, shape.expected, shape.actual) == (3, (1, 2), (1, 3))
    model = pickle.loads(
        pickle.dumps(E.IncompatibleModelError(actual_type="X", actual_model_name="m", supported=["a"]))
    )
    assert (model.actual_type, model.actual_model_name, model.supported) == ("X", "m", ["a"])


def _add_note(exc: BaseException, note: str) -> None:
    # BaseException.add_note is 3.11+; on 3.10 the note list is a plain attribute.
    if hasattr(exc, "add_note"):
        exc.add_note(note)
    else:
        exc.__notes__ = [*getattr(exc, "__notes__", []), note]  # type: ignore[attr-defined]


@pytest.mark.parametrize("exc", CASES, ids=lambda e: f"{type(e).__name__}-{CASES.index(e)}")
def test_notes_and_extra_attributes_survive_pickle_and_copy(exc) -> None:
    """Bug caught: a custom __reduce__ returns no state, so add_note() notes and
    attributes a caller or subclass set after construction vanish in a worker
    process or a copy."""
    fresh = pickle.loads(pickle.dumps(exc))
    message = str(fresh)
    _add_note(fresh, "while loading model A")
    fresh.request_id = "r-17"
    for clone in (pickle.loads(pickle.dumps(fresh)), copy.copy(fresh)):
        assert clone.__notes__ == ["while loading model A"]
        assert clone.request_id == "r-17"
        assert str(clone) == message
