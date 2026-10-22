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
