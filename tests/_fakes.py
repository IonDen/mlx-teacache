"""Test doubles that mirror real mflux contracts."""


class FaithfulCallbackRegistry:
    """Mirrors mflux.callbacks.CallbackRegistry: bare-name list attributes,
    conditional register() keyed on the callback implementing each hook."""

    def __init__(self) -> None:
        self.before_loop: list = []
        self.in_loop: list = []
        self.after_loop: list = []
        self.interrupt: list = []

    def register(self, cb) -> None:  # noqa: ANN001
        if hasattr(cb, "call_before_loop"):
            self.before_loop.append(cb)
        if hasattr(cb, "call_in_loop"):
            self.in_loop.append(cb)
        if hasattr(cb, "call_after_loop"):
            self.after_loop.append(cb)
        if hasattr(cb, "call_interrupt"):
            self.interrupt.append(cb)
