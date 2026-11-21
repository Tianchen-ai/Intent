"""Stable typed host slots. Keep every referenced allocation alive with the frame."""

import ctypes as ct
from dataclasses import dataclass
from .views import inspect


@dataclass
class Frame:
    owners: tuple
    slots: tuple
    shape: tuple


def frame(values, abi):
    slots = []
    matrix = None
    for value, arg in zip(values, abi):
        if arg["direction"] == "Scalar":
            slots.append(ct.c_float(value))
            continue
        v = inspect(value)
        if str(v.device) != "cpu" or any(s < 0 for s in v.strides):
            raise ValueError(
                "native CPU row ABI requires host memory and representable positive strides"
            )
        if len(v.shape) == 2 and matrix is None:
            matrix = v.shape
        slots.append(ct.c_void_p(v.address))
        slots.extend(ct.c_int64(s) for s in v.strides)
    return Frame(tuple(values), tuple(slots), matrix)


class NativeBinding:
    def __init__(self, function, frame, extra, validate):
        self.function = function
        self.owners = frame.owners
        self.arguments = (*frame.slots, *(ct.c_int64(x) for x in extra))
        self.argument_types = tuple(type(x) for x in self.arguments)
        self.function.argtypes = self.argument_types
        self.function.restype = None
        self.validate = validate

    def complete(self):
        pass

    def launch(self):
        self.validate()
        self.function(*self.arguments)


class HostVariants:
    """Bind synchronous host entries with distinct parameters to the same author ABI."""

    def __init__(self, artifact, values, outputs, scope):
        self.artifact = artifact
        self.bound = frame(values, artifact.data["abi"])
        self.outputs = tuple(outputs)
        self.scope = scope
        self.entries = []
        self.selected = None

    def compile(self):
        if self.entries:
            return
        self.artifact.validate(self.scope)
        fn = self.artifact.function()
        self.entries = self.bind_rows(fn, self.bound)

    def bind_rows(self, fn, bound):
        return [
            NativeBinding(
                fn,
                bound,
                (*bound.shape, r, self.artifact.workers),
                lambda: self.artifact.validate(self.scope),
            )
            for r in self.artifact.data["rows"]
        ]

    def launch(self):
        from time import perf_counter_ns

        self.compile()
        if self.selected is None:
            shape = self.bound.shape
            key = (
                self.scope,
                shape,
                tuple(x.value for x in self.bound.slots if isinstance(x, ct.c_int64)),
            )
            with self.artifact.lock:
                if key not in self.artifact.winners:
                    from .views import adapter

                    shadow = [
                        adapter(x).trial(x) if any(x is y for y in self.outputs) else x
                        for x in self.bound.owners
                    ]
                    trials = self.bind_rows(
                        self.entries[0].function,
                        frame(shadow, self.artifact.data["abi"]),
                    )
                    timings = []
                    for call in trials:
                        call.launch()
                        samples = []
                        for _ in range(3):
                            start = perf_counter_ns()
                            call.launch()
                            samples.append(perf_counter_ns() - start)
                        timings.append(sorted(samples)[1])
                    self.artifact.winners[key] = min(
                        range(len(timings)), key=timings.__getitem__
                    )
                self.selected = self.entries[self.artifact.winners[key]]
        self.selected.launch()

    def compilation_report(self):
        self.compile()
        return {
            "candidate_count": len(self.entries),
            "rows": [{"row": r, "build": "host"} for r in self.artifact.data["rows"]],
            "builds": list(self.artifact.compilations),
            "accounting": "one native module shared by all grain rows",
        }

    def native_call(self):
        if self.selected is None:
            raise RuntimeError("launch once before requesting the native winner")
        return self.selected

    def result(self):
        return self.outputs[0] if len(self.outputs) == 1 else self.outputs
