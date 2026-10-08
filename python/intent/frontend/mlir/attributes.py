from __future__ import annotations

import struct
import re
from dataclasses import dataclass
from enum import Enum
from enum import IntEnum

from intent.language import DType

from ..semantics.operations import IndexRelation
from ..semantics.operations import IndexTerm
from ..semantics.operations import AtomicOrdering
from ..semantics.operations import AtomicRMWKind
from ..semantics.operations import BinaryOperator
from ..semantics.operations import ComparePredicate
from ..semantics.operations import ScaledFormatKind
from ..semantics.operations import QuantFormatKind
from ..semantics.operations import ShapeExpr
from ..semantics.operations import ShapeRelation
from ..semantics.operations import UnaryOperator
from .types import quote


@dataclass(frozen=True, slots=True)
class DenseI32Array:
    values: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SymbolRef:
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not self.name.isidentifier():
            raise ValueError("MLIR symbol reference requires an identifier")


@dataclass(frozen=True, slots=True)
class ParameterAttribute:
    name: str
    origin: int


@dataclass(frozen=True, slots=True)
class FunctionKindAttribute:
    kind: int


@dataclass(frozen=True, slots=True)
class SparseFormatAttribute:
    kind: int
    compression_axis: int


def emit_attribute(value: object) -> str:
    if isinstance(value, DenseI32Array):
        return "array<i32" + (
            ": " + ", ".join(str(element) for element in value.values)
            if value.values else ""
        ) + ">"
    if isinstance(value, SymbolRef):
        return f"@{value.name}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, UnaryOperator):
        return f"#intent.unary_operator<{value.name.lower()}>"
    if isinstance(value, BinaryOperator):
        return f"#intent.binary_operator<{value.name.lower()}>"
    if isinstance(value, ComparePredicate):
        return f"#intent.compare_predicate<{value.name.lower()}>"
    if isinstance(value, AtomicOrdering):
        return f"#intent.atomic_ordering<{value.name.lower()}>"
    if isinstance(value, AtomicRMWKind):
        spelling = {
            AtomicRMWKind.MAX: "maximum",
            AtomicRMWKind.MIN: "minimum",
            AtomicRMWKind.AND: "bitwise_and",
            AtomicRMWKind.OR: "bitwise_or",
            AtomicRMWKind.XOR: "bitwise_xor",
        }.get(value, value.name.lower())
        return f"#intent.atomic_rmw_kind<{spelling}>"
    if isinstance(value, ScaledFormatKind):
        return f"#intent.scaled_format<{value.name.lower()}>"
    if isinstance(value, QuantFormatKind):
        return f"#intent.quant_format<{value.name.lower()}>"
    if isinstance(value, IntEnum):
        return f"{int(value)} : i64"
    if isinstance(value, int):
        return f"{value} : i64"
    if isinstance(value, float):
        bits = int.from_bytes(struct.pack(">d", value), byteorder="big")
        return f"0x{bits:016X} : f64"
    if isinstance(value, str):
        return quote(value)
    if value is None:
        return "unit"
    if isinstance(value, DType):
        from .types import emit_dtype

        return f"{emit_dtype(value)}"
    if isinstance(value, Enum):
        return quote(str(value.value))
    if isinstance(value, IndexRelation):
        terms = ", ".join(_emit_index_term(term) for term in value.terms)
        return (
            f"#intent.index_relation<{value.source_rank}, {value.result_rank}, "
            f"{_emit_dense_i64(value.result_dimensions)}, [{terms}]>"
        )
    if isinstance(value, ShapeExpr):
        return f"#intent.shape_expr<{int(value.kind)}, {value.dimension}, {value.payload}>"
    if isinstance(value, ShapeRelation):
        return "#intent.shape_relation<[" + ", ".join(
            emit_attribute(axis) for axis in value.axes
        ) + "]>"
    if isinstance(value, ParameterAttribute):
        return f"#intent.parameter<{quote(value.name)}, {value.origin}>"
    if isinstance(value, FunctionKindAttribute):
        return f"#intent.function_kind<{value.kind}>"
    if isinstance(value, SparseFormatAttribute):
        return f"#intent.sparse_format<{value.kind}, {value.compression_axis}>"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(emit_attribute(element) for element in value) + "]"
    if isinstance(value, dict):
        return emit_dictionary(value)
    raise TypeError(f"unsupported MLIR attribute value {value!r}")


def emit_dictionary(values: dict[str, object]) -> str:
    return "{" + ", ".join(
        f"{_emit_key(key)} = {emit_attribute(value)}"
        for key, value in sorted(values.items())
    ) + "}"


def _emit_key(key: str) -> str:
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$.]*", key):
        return key
    return quote(key)


def _emit_index_term(term: IndexTerm) -> str:
    absent = -(1 << 63)
    operands = tuple(
        -1 if position is None else position for position in term.operand_positions
    )
    static = tuple(absent if value is None else value for value in term.static_values)
    return (
        f"#intent.index_term<{int(term.kind)}, {_emit_dense_i64(operands)}, "
        f"{_emit_dense_i64(static)}>"
    )


def _emit_dense_i64(values: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"
