from __future__ import annotations

import ctypes
from math import prod
from itertools import product
from pathlib import Path

from intent.language.dtypes import DTYPES


_runtimes: dict[str, Runtime] = {}
ELEMENT_BYTES = {name: (dtype.bits + 7) // 8 for name, dtype in DTYPES.items()}


class Runtime:
    def __init__(self, neuware: str) -> None:
        self.library = ctypes.CDLL(str(Path(neuware) / "lib64/libcnrt.so"))
        signatures = {
            "cnrtGetDeviceCount": [ctypes.POINTER(ctypes.c_uint)],
            "cnrtSetDevice": [ctypes.c_int],
            "cnrtMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
            "cnrtFree": [ctypes.c_void_p],
            "cnrtMemcpy": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
            "cnrtQueueCreate": [ctypes.POINTER(ctypes.c_void_p)],
            "cnrtQueueSync": [ctypes.c_void_p],
            "cnrtQueueDestroy": [ctypes.c_void_p],
            "cnrtNotifierCreate": [ctypes.POINTER(ctypes.c_void_p)],
            "cnrtNotifierDestroy": [ctypes.c_void_p],
            "cnrtPlaceNotifier": [ctypes.c_void_p, ctypes.c_void_p],
            "cnrtNotifierDuration": [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)],
        }
        for name, signature in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = signature
            function.restype = ctypes.c_int

    def invoke(self, name: str, *arguments) -> None:
        status = getattr(self.library, name)(*arguments)
        if status:
            raise RuntimeError(f"{name} failed with CNRT status {status}")

    def select(self, device: int) -> None:
        count = ctypes.c_uint()
        self.invoke("cnrtGetDeviceCount", ctypes.byref(count))
        if device < 0 or device >= count.value:
            raise ValueError(f"MLU device {device} is unavailable; CNRT reports {count.value} devices")
        self.invoke("cnrtSetDevice", device)


def runtime(neuware: str = "/usr/local/neuware") -> Runtime:
    key = str(Path(neuware).resolve())
    if key not in _runtimes:
        _runtimes[key] = Runtime(key)
    return _runtimes[key]


class DeviceBuffer:
    """An owned contiguous MLU allocation with an explicit shape and storage dtype."""

    def __init__(self, shape: tuple[int, ...], dtype: str, *, device: int = 0,
                 neuware: str = "/usr/local/neuware") -> None:
        self.shape = tuple(shape)
        if any(not isinstance(size, int) or size < 0 for size in self.shape):
            raise ValueError("MLU extents must be nonnegative integers")
        if dtype not in ELEMENT_BYTES:
            raise NotImplementedError(f"MLU storage dtype {dtype} is not implemented")
        self.dtype = dtype
        self.device = device
        self.strides = tuple(prod(self.shape[i + 1:]) for i in range(len(self.shape)))
        self.nbytes = prod(self.shape) * ELEMENT_BYTES[dtype]
        self.runtime = runtime(neuware)
        self.pointer = 0
        self.runtime.select(device)
        pointer = ctypes.c_void_p()
        self.runtime.invoke("cnrtMalloc", ctypes.byref(pointer), max(1, self.nbytes))
        self.pointer = pointer.value

    @classmethod
    def from_host(cls, storage, *, shape: tuple[int, ...], dtype: str,
                  device: int = 0, neuware: str = "/usr/local/neuware") -> DeviceBuffer:
        result = cls(shape, dtype, device=device, neuware=neuware)
        result.copy_from_host(storage)
        return result

    def copy_from_host(self, storage) -> None:
        view = memoryview(storage)
        if not view.c_contiguous or view.nbytes != self.nbytes:
            raise ValueError("host storage must be contiguous and match the MLU allocation size")
        if not self.pointer:
            raise ValueError("MLU allocation is closed")
        self.runtime.select(self.device)
        if self.nbytes:
            owner = ctypes.create_string_buffer(view.tobytes(), self.nbytes)
            self.runtime.invoke("cnrtMemcpy", self.pointer, ctypes.addressof(owner), self.nbytes, 0)

    def to_host(self) -> bytearray:
        if not self.pointer:
            raise ValueError("MLU allocation is closed")
        output = bytearray(self.nbytes)
        self.runtime.select(self.device)
        if self.nbytes:
            owner = (ctypes.c_ubyte * self.nbytes).from_buffer(output)
            self.runtime.invoke("cnrtMemcpy", ctypes.addressof(owner), self.pointer, self.nbytes, 2)
        return output

    def close(self) -> None:
        if self.pointer:
            self.runtime.invoke("cnrtSetDevice", self.device)
            self.runtime.invoke("cnrtFree", self.pointer)
            self.pointer = 0

    @property
    def allocation_pointer(self) -> int:
        return self.pointer

    @property
    def byte_bounds(self) -> tuple[int, int]:
        return self.pointer, self.pointer + self.nbytes

    def view(self, shape: tuple[int, ...], strides: tuple[int, ...], *, offset: int = 0) -> DeviceView:
        return DeviceView(self, shape, strides, offset=offset)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        pointer = getattr(self, "pointer", 0)
        if pointer:
            self.runtime.library.cnrtSetDevice(self.device)
            self.runtime.library.cnrtFree(pointer)
            self.pointer = 0


class DeviceView:
    """A shape/stride-preserving view retaining its owning device allocation."""

    def __init__(self, owner: DeviceBuffer | DeviceView, shape: tuple[int, ...],
                 strides: tuple[int, ...], *, offset: int = 0) -> None:
        self.shape, self.strides = tuple(shape), tuple(strides)
        if len(self.shape) != len(self.strides) or any(not isinstance(size, int) or size < 0 for size in self.shape):
            raise ValueError("MLU view shape and strides must have matching ranks and nonnegative extents")
        if not isinstance(offset, int) or any(not isinstance(stride, int) for stride in self.strides):
            raise TypeError("MLU view offset and strides must be integer element offsets")
        if not owner.pointer:
            raise ValueError("cannot view a closed MLU allocation")
        self.owner = owner.owner if isinstance(owner, DeviceView) else owner
        self.dtype, self.device, self.runtime = owner.dtype, owner.device, owner.runtime
        width = ELEMENT_BYTES[self.dtype]
        self.byte_offset = (owner.byte_offset if isinstance(owner, DeviceView) else 0) + offset * width
        self.nbytes = prod(self.shape) * width
        deltas = [(size - 1) * stride * width for size, stride in zip(self.shape, self.strides)]
        self._lower = sum(min(0, delta) for delta in deltas) if self.nbytes else 0
        self._upper = sum(max(0, delta) for delta in deltas) + width if self.nbytes else 0
        if self.byte_offset + self._lower < 0 or self.byte_offset + self._upper > self.owner.nbytes:
            raise ValueError("MLU view reaches outside its allocation")

    @property
    def pointer(self) -> int:
        return self.owner.pointer + self.byte_offset if self.owner.pointer else 0

    @property
    def allocation_pointer(self) -> int:
        return self.owner.pointer

    @property
    def byte_bounds(self) -> tuple[int, int]:
        return self.pointer + self._lower, self.pointer + self._upper

    def view(self, shape: tuple[int, ...], strides: tuple[int, ...], *, offset: int = 0) -> DeviceView:
        return DeviceView(self, shape, strides, offset=offset)

    def to_host(self) -> bytearray:
        if not self.pointer:
            raise ValueError("MLU view refers to a closed allocation")
        self.runtime.select(self.device)
        lower, upper = self.byte_bounds
        span = bytearray(upper - lower)
        if span:
            staging = (ctypes.c_ubyte * len(span)).from_buffer(span)
            self.runtime.invoke("cnrtMemcpy", ctypes.addressof(staging), lower, len(span), 2)
        width = ELEMENT_BYTES[self.dtype]
        output = bytearray(self.nbytes)
        for linear, coordinates in enumerate(product(*(range(size) for size in self.shape))):
            source = sum(index * stride for index, stride in zip(coordinates, self.strides)) * width - self._lower
            output[linear * width:(linear + 1) * width] = span[source:source + width]
        return output
