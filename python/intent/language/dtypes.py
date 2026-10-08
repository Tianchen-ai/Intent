from __future__ import annotations

import builtins as python_builtins
from dataclasses import dataclass
from enum import Enum


class DTypeCategory(Enum):
    BOOL = "bool"
    SIGNED_INTEGER = "signed_integer"
    UNSIGNED_INTEGER = "unsigned_integer"
    INDEX = "index"
    FLOAT = "float"
    BFLOAT = "bfloat"


@dataclass(frozen=True, slots=True)
class DType:
    name: str
    category: DTypeCategory
    bits: int | None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dtype name must not be empty")
        if not isinstance(self.category, DTypeCategory):
            raise TypeError("dtype category must be a DTypeCategory")
        if isinstance(self.bits, python_builtins.bool) or (
            self.bits is not None and not isinstance(self.bits, python_builtins.int)
        ):
            raise TypeError("dtype bit width must be an integer or None")
        if self.bits is not None and self.bits <= 0:
            raise ValueError("dtype bit width must be positive")
        if self.category is DTypeCategory.INDEX and self.bits != 64:
            raise ValueError("logical index dtype must be signed 64-bit")

    def __repr__(self) -> str:
        return f"I.{self.name}"


bool = DType("bool", DTypeCategory.BOOL, 1)
index = DType("index", DTypeCategory.INDEX, 64)

i8 = DType("i8", DTypeCategory.SIGNED_INTEGER, 8)
i16 = DType("i16", DTypeCategory.SIGNED_INTEGER, 16)
i32 = DType("i32", DTypeCategory.SIGNED_INTEGER, 32)
i64 = DType("i64", DTypeCategory.SIGNED_INTEGER, 64)

u8 = DType("u8", DTypeCategory.UNSIGNED_INTEGER, 8)
u16 = DType("u16", DTypeCategory.UNSIGNED_INTEGER, 16)
u32 = DType("u32", DTypeCategory.UNSIGNED_INTEGER, 32)
u64 = DType("u64", DTypeCategory.UNSIGNED_INTEGER, 64)

f8e4m3fn = DType("f8e4m3fn", DTypeCategory.FLOAT, 8)
f8e5m2 = DType("f8e5m2", DTypeCategory.FLOAT, 8)
f16 = DType("f16", DTypeCategory.FLOAT, 16)
bf16 = DType("bf16", DTypeCategory.BFLOAT, 16)
f32 = DType("f32", DTypeCategory.FLOAT, 32)
f64 = DType("f64", DTypeCategory.FLOAT, 64)


DTYPES = {
    value.name: value
    for value in (
        bool,
        index,
        i8,
        i16,
        i32,
        i64,
        u8,
        u16,
        u32,
        u64,
        f8e4m3fn,
        f8e5m2,
        f16,
        bf16,
        f32,
        f64,
    )
}


def dtype(name: str) -> DType:
    return DTYPES[name]
