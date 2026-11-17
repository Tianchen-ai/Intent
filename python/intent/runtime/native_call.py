"""Public, invocation-owned C ABI binding for native consumers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import ctypes


@dataclass(frozen=True)
class NativeInvocation:
    """A selected native entry and its typed arguments, with retained owners.

    Consumers must call validate before submitting the entry. This is a local
    process binding, not a portable artifact or a benchmark protocol.
    """
    function: object
    arguments: tuple
    _owners: tuple = field(repr=False)
    _validate: Callable[[], None] = field(repr=False)
    _complete: Callable[[], None] | None = field(default=None, repr=False)

    @property
    def argument_types(self):
        return tuple(self.function.argtypes)

    @property
    def address(self):
        return ctypes.cast(self.function, ctypes.c_void_p).value

    def validate(self):
        self._validate()

    def launch(self):
        self.validate()
        self.function(*self.arguments)
        self.complete()

    def complete(self):
        """Report native worker failures after the synchronous invocation."""
        if self._complete is not None:
            self._complete()


@dataclass(frozen=True)
class NativeMemoryFill:
    """Contiguous CPU storage filled with one already encoded element pattern."""

    address: int
    count: int
    value: bytes
    _owners: tuple = field(repr=False)
    _validate: Callable[[], None] = field(repr=False)

    def __post_init__(self):
        if self.count < 0 or len(self.value) not in (1, 2, 4, 8):
            raise ValueError("native fill requires a nonnegative count and a 1/2/4/8-byte pattern")
        if self.count and not self.address:
            raise ValueError("nonempty native fill requires allocated storage")

    def validate(self):
        self._validate()

    def launch(self):
        self.validate()
        if not self.count:
            return
        import numpy as np
        import sys
        width = len(self.value)
        raw = (ctypes.c_ubyte * (self.count * width)).from_address(self.address)
        np.frombuffer(raw, dtype=f"u{width}")[:] = int.from_bytes(self.value, byteorder=sys.byteorder)

    def complete(self):
        pass
