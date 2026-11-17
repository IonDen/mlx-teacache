"""The real internal session for tests that drive the lifecycle hooks or a forward without `apply()`.

`make_session(**overrides)` builds `mlx_teacache.integrations._session.InternalHandle`, the one session type every
integration hands to the lifecycle callback and the forwards. Tests use it instead of a hand-written look-alike, so a
field added to, renamed on or dropped from the session reaches them. Imports only the session module (no mflux), so
pure-core tests can use it.
"""

from typing import Any

from mlx_teacache.integrations._session import InternalHandle

# Neutral defaults: the package's window (1 / 1) and fallback threshold (0.20). Coefficients are the identity
# polynomial, so a test that never reaches the gate does not depend on any variant's fit.
_DEFAULTS: dict[str, Any] = {
    "rel_l1_thresh": 0.20,
    "coefficients": (1.0, 0.0, 0.0, 0.0, 0.0),
    "skip_first_n_steps": 1,
    "skip_last_n_steps": 1,
    "calibrated_range": None,
}


def make_session(**overrides: Any) -> InternalHandle:
    """A fresh session with the defaults above; any constructor keyword may be overridden."""
    unknown = set(overrides) - set(_DEFAULTS)
    if unknown:
        raise TypeError(f"make_session() got unknown keyword(s): {sorted(unknown)}")
    return InternalHandle(**{**_DEFAULTS, **overrides})
