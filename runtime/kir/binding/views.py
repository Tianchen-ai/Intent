"""Bind logical views while leaving allocation and ownership with the container."""

from dataclasses import dataclass
from math import gcd, prod
from . import numpy_views, torch_views

ADAPTERS = (numpy_views, torch_views)


def adapter(value):
    for implementation in ADAPTERS:
        if implementation.accepts(value):
            return implementation
    raise TypeError(f"unsupported tensor container: {type(value).__module__}")


@dataclass(frozen=True)
class View:
    dtype: str
    shape: tuple
    strides: tuple
    device: object
    allocation: object
    address: int
    writable: bool


def inspect(value):
    return View(*adapter(value).describe(value))


class Signature:
    def __init__(self, arguments):
        self.args = arguments

    def bind(self, inputs, outputs):
        arguments = [arg for arg in self.args if arg["direction"] != "Out"]
        if len(inputs) != len(arguments):
            raise TypeError("input count differs from signature")
        shapes, bound = {}, {}
        device = allocator = None

        def check(arg, value):
            view = inspect(value)
            if view.dtype != arg["dtype"] or len(view.shape) != len(arg["dims"]):
                raise ValueError(f"invalid view for {arg['name']}")
            for axis, n in zip(arg["dims"], view.shape):
                if shapes.setdefault(axis, n) != n:
                    raise ValueError(f"inconsistent dimension {axis}")
            return view

        for arg, value in zip(arguments, inputs):
            if arg["direction"] == "Scalar":
                if not isinstance(value, (int, float)):
                    raise TypeError("expected a host scalar")
                value = float(value)
            else:
                view = check(arg, value)
                if device is not None and device != view.device:
                    raise ValueError("arguments span devices")
                allocator = allocator or adapter(value)
                device = view.device
            bound[arg["name"]] = value
        outs = [a for a in self.args if a["direction"] == "Out"]
        supplied = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
        if outputs is not None and len(supplied) != len(outs):
            raise ValueError("output count differs")
        result = []
        for i, arg in enumerate(outs):
            y = (
                supplied[i]
                if outputs is not None
                else allocator.allocate(
                    tuple(shapes[n] for n in arg["dims"]), arg["dtype"], device
                )
            )
            view = check(arg, y)
            if view.device != device or not view.writable:
                raise ValueError("output device or write permissions differ")
            dims = [(n, abs(s)) for n, s in zip(view.shape, view.strides) if n > 1]
            conflict = any(s == 0 for _, s in dims)
            if len(dims) == 2 and not conflict:
                (m, a), (n, b) = dims
                g = gcd(a, b)
                conflict = m > b // g and n > a // g
            if prod(view.shape) and (
                conflict
                or any(
                    adapter(y).overlaps(y, bound[a["name"]])
                    for a in self.args
                    if a["name"] in bound and a["direction"] != "Scalar"
                )
            ):
                raise ValueError("row kernel needs nonoverlapping writable storage")
            result.append(y)
            bound[arg["name"]] = y
        return tuple(bound[a["name"]] for a in self.args), result, device
