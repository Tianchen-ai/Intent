"""Bound author sequence. Members retain their allocations for the entire launch."""

from dataclasses import dataclass
from math import prod
from . import views


@dataclass(frozen=True)
class NativeFill:
    address: int
    count: int
    value: bytes
    owner: object

    def validate(self):
        pass

    def complete(self):
        pass


class Fill:
    def __init__(self, destination, value):
        desc = views.inspect(destination)
        views.Signature.require_writable(desc)
        if desc.dtype == "i32" and (type(value) is not int or not -(1<<31) <= value < 1<<31):
            raise ValueError("i32 fill requires a representable integer")
        self.destination = destination
        self.value = value
        self.desc = desc

    def compile(self):
        pass

    def launch(self):
        views.adapter(self.destination).fill(self.destination, self.value)

    def native_calls(self):
        import numpy as np

        if str(self.desc.device) != "cpu":
            raise ValueError("native host fills require a CPU allocation")
        # A dense view may permute axes; writing one scalar remains contiguous.
        span = 1
        for stride, n in sorted(zip(self.desc.strides, self.desc.shape)):
            if n > 1 and stride != span:
                raise ValueError("strided fill cannot be exported as a contiguous native fill")
            span *= n
        dtype = self.desc.dtype
        if dtype == "bf16":
            import ml_dtypes
            dtype = ml_dtypes.bfloat16
        else:
            dtype = {"f16": "float16", "f32":"float32", "i32":"int32"}[dtype]
        return (NativeFill(self.desc.address, prod(self.desc.shape),
                           np.asarray(self.value, dtype=dtype).tobytes(), self.destination),)


class PreparedSequence:
    def __init__(self, steps, labels, outputs):
        self.steps = tuple(steps)
        self.labels = tuple(labels)
        self.outputs = outputs

    def compile(self):
        for step in self.steps:
            step.compile()

    def launch(self):
        for step in self.steps:
            step.launch()

    def result(self):
        return self.outputs

    def native_calls(self):
        return tuple(binding for step in self.steps for binding in
                     (step.native_calls() if isinstance(step, Fill) else (step.native_call(),)))

    def tuning_report(self):
        return {"steps": [
            {"op": "fill", "slot": name, "value": step.value, "dtype": step.desc.dtype,
             "shape": tuple(step.desc.shape), "strides": tuple(step.desc.strides)}
            if isinstance(step, Fill) else {"op": "call", "kernel": name, **step.tuning_report()}
            for name, step in zip(self.labels, self.steps, strict=True)]}

    def compilation_report(self):
        return {"steps": [{"op":"fill"} if isinstance(step, Fill) else
                          {"op":"call", **step.compilation_report()} for step in self.steps]}
