"""The two introspection helpers the mflux contract pins are built on, checked
on toy classes so their behaviour is pinned independently of mflux."""

from tests._mflux_surface import assigned_attributes, ast_fingerprint


class _Toy:
    def __init__(self, flag: bool) -> None:
        self.a = 1
        if flag:
            self.b = 2
        for _ in range(1):
            self.c = 3
        other = self
        other.not_mine = 4  # not `self.`


def _f1(x):
    """doc"""
    return x + 1  # comment


def _f2(x):
    return x + 1


def _f3(x):
    return x + 2


def test_assigned_attributes_walks_nested_blocks_and_ignores_non_self() -> None:
    # bug caught: only scanning top-level statements, or matching any `<name>.attr =`
    assert assigned_attributes(_Toy) == frozenset({"a", "b", "c"})


def test_fingerprint_ignores_docstrings_and_comments_but_not_statements() -> None:
    # bug caught: hashing raw source (a comment edit would flip it) or hashing too little
    assert ast_fingerprint(_f1) == ast_fingerprint(_f2)
    assert ast_fingerprint(_f1) != ast_fingerprint(_f3)


def _pinned(a, b=1, *args, key=None, **kw):
    """The docstring is stripped before hashing."""
    if a:
        return [x + b for x in args]
    return {"k": key, **kw}


# Recorded on CPython 3.10 and checked on 3.12 and 3.13: the digest must not move
# with the interpreter, or the drift table in tests/test_mflux_forward_drift.py
# would be recorded on one Python and fail on the CI matrix's others.
_PINNED_DIGEST = "67eb2b095c5eb155"


def test_fingerprint_digest_is_pinned_across_interpreters() -> None:
    # bug caught: hashing ast.dump text (3.12 adds type_params, 3.13 drops empty
    # fields) or any other interpreter-dependent field
    assert ast_fingerprint(_pinned) == _PINNED_DIGEST


def _returns_pair(x):
    if x:
        return x, x
    return None


def test_return_tuple_arities_counts_each_literal_tuple() -> None:
    # bug caught: reading only the first return, or counting a bare value as 0
    from tests._mflux_surface import return_tuple_arities

    assert return_tuple_arities(_returns_pair) == frozenset({2, 1})
