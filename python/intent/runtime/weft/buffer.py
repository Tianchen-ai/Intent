from __future__ import annotations

import ctypes
from math import prod

from intent.language.dtypes import DTYPES


# This describes raw storage, not the external compiler's arithmetic support.
ELEMENT_BYTES = {name: (dtype.bits + 7) // 8 for name, dtype in DTYPES.items()}


class Buffer:
    """A contiguous CPU view with explicit Intent element type and shape."""

    def __init__(self, storage, *, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.element_bytes = ELEMENT_BYTES[dtype]
        if any(not isinstance(extent, int) or extent < 0 for extent in self.shape):
            raise ValueError("buffer extents must be nonnegative integers")
        view = memoryview(storage)
        if view.readonly or not view.c_contiguous:
            raise NotImplementedError("native Weft buffers require writable contiguous storage")
        self.storage = view.cast("B")
        self.nbytes = prod(self.shape) * self.element_bytes
        if self.storage.nbytes != self.nbytes:
            raise ValueError("buffer storage size disagrees with its shape and dtype")
        self.pointer = ctypes.addressof(ctypes.c_char.from_buffer(self.storage)) if self.nbytes else 0
        owner = memoryview(view.obj).cast("B")
        self.allocation = ctypes.addressof(ctypes.c_char.from_buffer(owner)) if owner.nbytes else id(view.obj)
        self.allocation_end = self.allocation + owner.nbytes
        self.strides = tuple(prod(self.shape[axis + 1:]) for axis in range(len(self.shape)))

    @classmethod
    def empty(cls, shape: tuple[int, ...], dtype: str) -> Buffer:
        return cls(bytearray(prod(shape) * ELEMENT_BYTES[dtype]), shape=shape, dtype=dtype)
