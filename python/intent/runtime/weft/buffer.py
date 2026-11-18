from __future__ import annotations

import ctypes
from math import prod

from intent.language.dtypes import DTYPES


# This describes raw storage, not the external compiler's arithmetic support.
ELEMENT_BYTES = {name: (dtype.bits + 7) // 8 for name, dtype in DTYPES.items()}


class Buffer:
    """A borrowed CPU view. Strides and offsets count storage elements."""

    def __init__(self, storage, *, shape: tuple[int, ...], dtype: str,
                 strides=None, offset=0) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.element_bytes = ELEMENT_BYTES[dtype]
        if any(type(extent) is not int or extent < 0 for extent in self.shape):
            raise ValueError("buffer extents must be nonnegative integers")
        view = memoryview(storage)
        if view.readonly or not view.c_contiguous:
            raise NotImplementedError("native Weft buffers require writable contiguous storage")
        backing = view.cast("B")
        dense = tuple(prod(self.shape[axis + 1:]) for axis in range(len(self.shape)))
        self.strides = dense if strides is None else tuple(strides)
        if len(self.strides) != len(self.shape) or any(type(s) is not int or s < 0 for s in self.strides):
            raise ValueError("Weft requires one nonnegative element stride per axis")
        if type(offset) is not int or offset < 0:
            raise ValueError("Weft buffer offset must be a nonnegative element offset")
        self.canonical = self.strides == dense
        self.nbytes = prod(self.shape) * self.element_bytes
        if strides is None and offset == 0 and backing.nbytes != self.nbytes:
            raise ValueError("buffer storage size disagrees with its shape and dtype")
        span = (1 + sum((n - 1) * s for n, s in zip(self.shape, self.strides))) if self.nbytes else 0
        first = offset * self.element_bytes
        last = first + span * self.element_bytes
        if last > backing.nbytes:
            raise ValueError("Weft view exceeds its backing storage")
        self.storage_span = backing[first:last]
        address = ctypes.c_char * 0
        self.pointer = ctypes.addressof(address.from_buffer(self.storage_span))
        owner = memoryview(view.obj).cast("B")
        self.allocation = ctypes.addressof(address.from_buffer(owner))
        self.allocation_end = self.allocation + owner.nbytes
        self.allocation_identity = self.allocation if owner.nbytes else id(view.obj)
        delta = self.pointer - self.allocation
        if delta % self.element_bytes:
            raise ValueError("Weft view offset must span whole storage elements")
        self.offset = delta // self.element_bytes

    @property
    def storage(self):
        if not self.canonical:
            raise ValueError("strided Weft views have no flat logical byte buffer")
        return self.storage_span

    def _elements(self):
        import numpy as np
        return np.ndarray(self.shape, dtype=f"V{self.element_bytes}", buffer=self.storage_span,
                          strides=tuple(s * self.element_bytes for s in self.strides))

    @classmethod
    def empty(cls, shape: tuple[int, ...], dtype: str) -> Buffer:
        return cls(bytearray(prod(shape) * ELEMENT_BYTES[dtype]), shape=shape, dtype=dtype)

    def to_host(self):
        return bytearray(self.storage_span if self.canonical else self._elements().tobytes())

    def copy_from_host(self, storage):
        value = memoryview(storage).cast("B")
        if value.nbytes != self.nbytes:
            raise ValueError("host byte count disagrees with the native buffer")
        if self.canonical:
            self.storage_span[:] = value
        else:
            import numpy as np
            occupied = 1
            for stride, size in sorted((s, n) for n, s in zip(self.shape, self.strides) if n > 1):
                if self.nbytes and stride < occupied:
                    raise ValueError("cannot write host bytes into an overlapping Weft view")
                occupied += (size - 1) * stride
            elements = self._elements()
            elements[...] = np.frombuffer(value, dtype=elements.dtype).reshape(self.shape)
