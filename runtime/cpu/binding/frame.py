"""Stable typed host slots. Keep every referenced allocation alive with the frame."""

import ctypes as ct
from dataclasses import dataclass
from runtime.kir.binding.views import inspect


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
    def __init__(self, function, frame, extra, validate, status_type):
        self.function = function
        self.owners = frame.owners
        self.arguments = (*frame.slots, *(ct.c_int64(x) for x in extra))
        self.argument_types = tuple(type(x) for x in self.arguments)
        self.function.argtypes = self.argument_types
        self.function.restype = status_type
        self.return_type = status_type
        self.geometry_offset = len(frame.slots)
        self.validate = validate

    def complete(self):
        pass

    def launch(self):
        self.validate()
        arguments = self.arguments
        self.check_status(self.function(*arguments), arguments)

    def check_status(self, status, arguments=None):
        if status:
            actual = self.arguments if arguments is None else arguments
            geometry = tuple(x.value for x in actual[self.geometry_offset :])
            raise RuntimeError(
                f"{self.function.__name__} failed with native status {status}, "
                f"shape, grain, columns and workers={geometry}"
            )


class HostVariants:
    """Bind synchronous host entries with distinct parameters to the same author ABI."""

    def __init__(self, artifact, values, outputs, scope):
        self.artifact = artifact
        self.bound = frame(values, artifact.data["abi"])
        self.outputs = tuple(outputs)
        self.scope = scope
        self.timings = {}
        self.winner = None
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
                (*bound.shape, r["grain"], r["columns"], self.artifact.workers),
                lambda: self.artifact.validate(self.scope),
                self.artifact.status_type,
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
                    from runtime.kir.binding.trials import TrialBindings

                    shadow = TrialBindings(self.bound.owners, self.artifact.data["abi"])
                    trials = self.bind_rows(
                        self.entries[0].function,
                        frame(shadow.values, self.artifact.data["abi"]),
                    )
                    timings = []
                    for call in trials:
                        shadow.reset()
                        call.launch()
                        samples = []
                        for _ in range(3):
                            shadow.reset()
                            start = perf_counter_ns()
                            call.launch()
                            samples.append(perf_counter_ns() - start)
                        timings.append([ns / 1000 for ns in samples])
                    winner = min(
                        range(len(timings)), key=lambda i: sorted(timings[i])[1]
                    )
                    self.artifact.winners[key] = winner, dict(enumerate(timings))
                self.winner, self.timings = self.artifact.winners[key]
                self.selected = self.entries[self.winner]
        self.selected.launch()

    def tuning_report(self):
        from runtime.kir.binding.selection import selection

        return selection(self.artifact.data, self.winner, self.timings)

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
