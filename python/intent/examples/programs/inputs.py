"""Portable host input storage for the ordinary examples.

Low-precision arrays keep their actual storage format. Providers adapt these
bytes at the invocation boundary; host preparation does not require Torch.
"""

from dataclasses import dataclass
from contextvars import ContextVar
from math import prod

import numpy as np


_NUMPY_TYPES = {
    "bool": np.bool_, "u8": np.uint8, "i8": np.int8,
    "u16": np.uint16, "i16": np.int16,
    "u32": np.uint32, "i32": np.int32,
    "u64": np.uint64, "i64": np.int64,
    "f16": np.float16, "f32": np.float32, "f64": np.float64,
}
_LOW_PRECISION_TYPES = {
    "bf16": "bfloat16", "f8e4m3fn": "float8_e4m3fn", "f8e5m2": "float8_e5m2",
}
_RANDOM = ContextVar("example_random", default=None)


def seed(value):
    """Give this program's host preparation its own reproducible input stream."""
    _RANDOM.set(np.random.default_rng(value))


def _random():
    generator = _RANDOM.get()
    if generator is None:
        generator = np.random.default_rng()
        _RANDOM.set(generator)
    return generator


def numpy_dtype(element):
    if element in _NUMPY_TYPES:
        return np.dtype(_NUMPY_TYPES[element])
    if element in _LOW_PRECISION_TYPES:
        import ml_dtypes
        return np.dtype(getattr(ml_dtypes, _LOW_PRECISION_TYPES[element]))
    raise NotImplementedError(f"example storage does not support dtype {element}")


@dataclass(eq=False)
class Array:
    data: np.ndarray | None
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self):
        self.shape = tuple(self.shape)
        if self.dtype not in _NUMPY_TYPES and self.dtype not in _LOW_PRECISION_TYPES:
            raise NotImplementedError(f"example storage does not support dtype {self.dtype}")
        if any(not isinstance(size, (int, np.integer)) or size < 0 for size in self.shape):
            raise ValueError("example array dimensions must be nonnegative integers")
        if self.data is not None:
            if self.data.shape != self.shape or self.data.dtype != numpy_dtype(self.dtype):
                raise ValueError("example array storage must match its declared shape and dtype")

    @classmethod
    def metadata(cls, shape, dtype):
        return cls(None, dtype, tuple(shape))

    @property
    def strides(self):
        if self.data is None:
            return tuple(prod(self.shape[axis + 1:]) for axis in range(len(self.shape)))
        return tuple(stride // self.data.itemsize for stride in self.data.strides)

    @property
    def nbytes(self):
        if self.data is not None:
            return self.data.nbytes
        bits = {"bool": 8, "bf16": 16, "f8e4m3fn": 8, "f8e5m2": 8}
        width = bits[self.dtype] if self.dtype in bits else int(self.dtype[1:])
        return prod(self.shape) * (width // 8)

    def to_numpy(self, dtype=None):
        if self.data is None:
            raise ValueError("a source-only array has no host contents")
        return self.data if dtype is None else self.data.astype(numpy_dtype(dtype))


def array(values, dtype):
    data = np.asarray(values, dtype=numpy_dtype(dtype))
    if not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)
    return Array(data, dtype, data.shape)


def from_bytes(storage, shape, dtype):
    shape = tuple(shape)
    data = np.frombuffer(storage, dtype=numpy_dtype(dtype), count=prod(shape)).reshape(shape)
    return Array(data, dtype, shape)


def empty(shape, dtype):
    data = np.empty(shape, dtype=numpy_dtype(dtype))
    return Array(data, dtype, data.shape)


def full(shape, value, dtype):
    data = np.full(shape, value, dtype=numpy_dtype(dtype))
    return Array(data, dtype, data.shape)


def zeros(shape, dtype):
    return full(shape, 0, dtype)


def ones(shape, dtype):
    return full(shape, 1, dtype)


def normal(shape, dtype="f32", *, scale=None):
    value = array(_random().standard_normal(shape, dtype=np.float32), dtype)
    if scale is not None:
        # Preserve the examples' low-precision sample -> scaled sample rounding.
        value = array(value.to_numpy("f32") * np.float32(scale), dtype)
    return value


def uniform(shape, dtype="f32"):
    return array(_random().random(shape, dtype=np.float32), dtype)


def integers(low, high, shape, dtype):
    return array(_random().integers(low, high, size=shape, dtype=numpy_dtype(dtype)), dtype)


def arange(start, stop=None, step=1, *, dtype="i32"):
    if stop is None:
        start, stop = 0, start
    return array(np.arange(start, stop, step, dtype=numpy_dtype(dtype)), dtype)
