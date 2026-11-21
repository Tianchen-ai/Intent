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


def touches(a, first, b, second):
    if not prod(first.shape) or not prod(second.shape):
        return False
    if first.dtype is None or second.dtype is None:
        raise ValueError("unsupported view element type")
    if str(first.device) != str(second.device):
        return False

    def span(v):
        width = {"f32": 4, "i32": 4, "bf16": 2, "f16": 2}[v.dtype]
        delta = [width * (n - 1) * s for n, s in zip(v.shape, v.strides)]
        return v.address + sum(min(d, 0) for d in delta), v.address + width + sum(
            max(d, 0) for d in delta
        )

    x, y = span(first)
    u, v = span(second)
    if y <= u or v <= x:
        return False
    if str(first.device) == "cpu":
        import numpy as np

        def raw_view(value, descriptor):
            if adapter(value) is numpy_views:
                return value
            # Dtype-independent strided bytes at each logical element.
            import ctypes

            begin, end = span(descriptor)
            width = value.element_size()
            storage = np.ctypeslib.as_array(
                (ctypes.c_ubyte * (end - begin)).from_address(begin)
            )
            return np.ndarray(
                (*descriptor.shape, width),
                dtype=np.uint8,
                buffer=storage,
                offset=descriptor.address - begin,
                strides=(*(s * width for s in descriptor.strides), 1),
            )

        try:
            return np.shares_memory(
                raw_view(a, first), raw_view(b, second), max_work=1000
            )
        except getattr(np, "exceptions", np).TooHardError:
            return True
    # On accelerators, overlapping byte spans may alias even when DLPack or
    # another framework constructed a new storage wrapper.
    return True


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
                if (isinstance(axis, int) and n != axis) or shapes.setdefault(
                    axis, n
                ) != n:
                    raise ValueError(f"inconsistent dimension {axis}")
            return view

        views = []
        for arg, value in zip(arguments, inputs):
            if arg["direction"] == "Scalar":
                if not isinstance(value, (int, float)):
                    raise TypeError("expected a host scalar")
                if arg["dtype"] == "u64":
                    if type(value) is not int or not 0 <= value < (1 << 64):
                        raise ValueError("scalar outside u64")
                elif arg["dtype"] == "i32":
                    if not isinstance(value, int) or not -(1 << 31) <= value < (
                        1 << 31
                    ):
                        raise ValueError("scalar outside i32")
                else:
                    value = float(value)
            else:
                view = check(arg, value)
                if device is not None and device != view.device:
                    raise ValueError("arguments span devices")
                if arg["direction"] == "InOut":
                    self.require_writable(view)
                for other, slot, peer in views:
                    if "InOut" not in (arg["direction"], slot["direction"]):
                        continue
                    if (
                        view.address == peer.address
                        and view.dtype == peer.dtype
                        and view.shape == peer.shape
                        and view.strides == peer.strides
                    ):
                        continue
                    if touches(value, view, other, peer):
                        raise ValueError(
                            f"parallel members of {arg['name']} and {slot['name']} may overlap"
                        )
                views.append((value, arg, view))
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
                    tuple(n if isinstance(n, int) else shapes[n] for n in arg["dims"]), arg["dtype"], device
                )
            )
            view = check(arg, y)
            if view.device != device or not view.writable:
                raise ValueError("output device or write permissions differ")
            self.require_writable(view)
            if prod(view.shape) and any(
                touches(y, view, bound[a["name"]], inspect(bound[a["name"]]))
                for a in self.args
                if a["name"] in bound and a["direction"] != "Scalar"
            ):
                raise ValueError("independent Out overlaps a bound argument")
            bound[arg["name"]] = y
        result = [
            bound[a["name"]] for a in self.args if a["direction"] in ("Out", "InOut")
        ]
        return tuple(bound[a["name"]] for a in self.args), result, device

    @staticmethod
    def require_writable(view):
        dims = [(n, abs(s)) for n, s in zip(view.shape, view.strides) if n > 1]
        conflict = any(s == 0 for _, s in dims)
        if len(dims) == 2 and not conflict:
            (m, a), (n, b) = dims
            g = gcd(a, b)
            conflict = m > b // g and n > a // g
        if len(dims) > 2:
            span = 1
            for n, s in sorted(dims, key=lambda t: t[1]):
                if s < span:
                    conflict = True
                span += (n - 1) * s
        if prod(view.shape) and (conflict or not view.writable):
            raise ValueError(
                "parallel writable view has overlapping or unverified members"
            )
