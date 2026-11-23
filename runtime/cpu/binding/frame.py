"""Stable typed host slots. Keep every referenced allocation alive with the frame."""

import ctypes as ct
from dataclasses import dataclass
from runtime.kir.binding.views import inspect
from runtime.kir.binding.completion import PreparedLaunch


@dataclass
class Frame:
    owners: tuple
    slots: tuple
    shape: tuple


def frame(values, abi, shape):
    slots = []
    for value, arg in zip(values, abi):
        if arg["direction"] == "Scalar":
            slots.append(
                ct.c_uint64(value)
                if arg["dtype"] == "u64"
                else ct.c_int32(value) if arg["dtype"] == "i32" else ct.c_float(value)
            )
            continue
        v = inspect(value)
        if str(v.device) != "cpu" or any(s < 0 for s in v.strides):
            raise ValueError(
                "native CPU row ABI requires host memory and representable positive strides"
            )
        slots.append(ct.c_void_p(v.address))
        if len(v.shape)>2:slots.extend(ct.c_int64(n) for n in v.shape[:-1])
        slots.extend(ct.c_int64(s) for s in v.strides)
    return Frame(tuple(values), tuple(slots), shape)


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


class HostVariants(PreparedLaunch):
    """Bind synchronous host entries with distinct parameters to the same author ABI."""

    device = "cpu"

    def __init__(self, artifact, values, outputs, scope, shape):
        self.artifact = artifact
        self.bound = frame(values, artifact.data["abi"], shape)
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

    def enqueue(self):
        from time import perf_counter_ns

        self.compile()
        if self.selected is None and len(self.entries)==1:
            self.winner,self.selected=0,self.entries[0]
            self.timings={0: []}
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
                        frame(shadow.values, self.artifact.data["abi"], shape),
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

    def bind_row(self, row_id):
        """Bind a declared native row to the same complete author invocation."""
        from copy import copy
        self.compile()
        if not 0 <= row_id < len(self.entries):
            raise ValueError("unknown CPU row")
        row=copy(self)
        row._forced = True
        row.selected=row.entries[row_id]
        row.winner=row_id
        return row

    def tuning_report(self):
        from runtime.kir.binding.selection import selection
        self.wait_previous()
        return selection(self, self.winner, self.timings)

    def compilation_report(self):
        self.compile()
        return {
            "candidate_count": len(self.entries),
            "rows": [{"row": r, "build": "host"} for r in self.artifact.data["rows"]],
            "builds": list(self.artifact.compilations),
            "accounting": "one native module shared by all grain rows",
        }

    def native_call(self):
        self.wait_previous()
        if self.selected is None:
            raise RuntimeError("launch once before requesting the native winner")
        return self.selected

    def result(self):
        self.wait_previous()
        return self.outputs[0] if len(self.outputs) == 1 else self.outputs
