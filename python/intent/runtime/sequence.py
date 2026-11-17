"""Explicit author-owned invocation sequences, independent of measurement."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable

from .native_call import NativeInvocation, NativeMemoryFill


@dataclass(frozen=True)
class PreparedInitialization:
    """A bound workspace initialization at its declared position in a sequence."""

    _launch: Callable[[], None] = field(repr=False)
    _owners: tuple = field(repr=False)
    _native: NativeMemoryFill | None = field(default=None, repr=False)
    _enqueue: Callable[[object], None] | None = field(default=None, repr=False)

    def compile(self):
        pass

    def launch(self):
        self._launch()

    def result(self):
        return None

    def native_call(self):
        if self._native is None:
            raise NotImplementedError("this initialization has no native CPU binding")
        return self._native

    def enqueue(self, queue):
        if self._enqueue is None:
            raise NotImplementedError("this initialization cannot submit to an explicit queue")
        self._enqueue(queue)


class PreparedSequence:
    """An ordered sequence of already-bound calls and explicit initializations.

    Construction and compile never launch or select tuning winners. Outputs and
    owners retain their original allocations. No algorithm, launch partitioning,
    scheduling or implicit transfer is inferred from the sequence.
    """

    def __init__(self, calls, *, outputs, owners=()):
        self.calls = tuple(calls)
        if not self.calls:
            raise ValueError("a prepared sequence requires at least one invocation")
        self._outputs = outputs
        self._owners = tuple(owners)
        self._launch_sequence = next((call.launch_sequence for call in self.calls
                                      if callable(getattr(call, "launch_sequence", None))), None)

    def compile(self):
        for call in self.calls:
            call.compile()

    def launch(self):
        if self._launch_sequence is not None:
            self._launch_sequence(self.calls)
            return
        for call in self.calls:
            call.launch()

    def enqueue(self, queue):
        """Submit every member to a caller-owned compatible provider queue."""
        for call in self.calls:
            call.enqueue(queue)

    def result(self):
        return self._outputs

    def native_calls(self) -> tuple[NativeInvocation | NativeMemoryFill, ...]:
        """Export the complete synchronous CPU sequence after first-use setup.

        Like an individual native_call(), exporting may select configurations.
        Call launch() first when consumers depend on outputs of preceding calls.
        Consumers retain this sequence and validate/complete each exported call.
        """
        result = []
        for call in self.calls:
            if isinstance(call, PreparedSequence):
                result.extend(call.native_calls())
            else:
                result.append(call.native_call())
        return tuple(result)

    @property
    def kernel_count(self):
        return sum(call.kernel_count if isinstance(call, PreparedSequence)
                   else not isinstance(call, PreparedInitialization) for call in self.calls)

    @property
    def initialization_count(self):
        return sum(call.initialization_count if isinstance(call, PreparedSequence)
                   else isinstance(call, PreparedInitialization) for call in self.calls)

    @property
    def observations(self):
        return tuple(item for call in self.calls
                     for item in (call.observations if isinstance(call, PreparedSequence)
                                  else (getattr(call, "observation", None),)) if item is not None)
