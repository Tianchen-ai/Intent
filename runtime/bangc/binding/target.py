"""Local CNRT device facts for the BANG C exit."""

from math import prod
import ctypes as C
from functools import lru_cache
import torch
from runtime.bangc.launch.call import Artifact


@lru_cache(maxsize=None)
def facts(index):
    lib = C.CDLL("libcnrt.so")
    get = lib.cnrtDeviceGetAttribute
    get.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int]
    values = {}
    # Stable cnrtDeviceAttr_t identifiers; sizes are bytes per physical owner.
    for name, field in (
        ("isa", 0x20A),
        ("clusters", 0x205),
        ("cores", 0x206),
        ("nram_bytes", 0x302),
    ):
        value = C.c_int()
        status = get(C.byref(value), field, index)
        if status:
            raise RuntimeError(f"CNRT query {name} on device {index} returned {status}")
        values[name] = value.value
    return values


class Target:
    def specialize(self, device, values):
        if device.type != "mlu":
            raise ValueError("BANG C requires inputs on a local MLU")
        x = next(v for v in values if isinstance(v, torch.Tensor))
        if not 1 <= x.ndim <= 4:
            raise ValueError("MLU member domain supports ranks one through four")
        return device, (prod(x.shape[:-1]), x.shape[-1])

    def options(self, device, shape):
        return dict(**facts(device.index), observed_width=shape[1])

    def materialize(self, data):
        return Artifact(data)
