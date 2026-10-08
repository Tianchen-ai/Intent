from __future__ import annotations

from collections.abc import Callable

from intent.language import DType
from intent.language.annotations import ViewConstraints
from intent.language.annotations import ViewKind

from ..semantics.types import BufferType
from ..semantics.types import ConstexprType
from ..semantics.types import DimExpr
from ..semantics.types import DomainType
from ..semantics.types import EnumType
from ..semantics.types import LogicalIndexType
from ..semantics.types import RecordType
from ..semantics.types import RegionType
from ..semantics.types import ScalarType
from ..semantics.types import StaticDim
from ..semantics.types import TensorType
from ..semantics.types import TupleType
from ..semantics.types import ValueType


_DTYPE_TYPES = {
    "bool": "i1",
    "index": "index",
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
    "u8": "ui8",
    "u16": "ui16",
    "u32": "ui32",
    "u64": "ui64",
    "f8e4m3fn": "f8E4M3FN",
    "f8e5m2": "f8E5M2",
    "f16": "f16",
    "bf16": "bf16",
    "f32": "f32",
    "f64": "f64",
}


DimensionID = Callable[[DimExpr], int]


def emit_type(value_type: ValueType, dimension_id: DimensionID | None = None) -> str:
    if isinstance(value_type, ScalarType):
        return emit_dtype(value_type.dtype)
    if isinstance(value_type, TensorType):
        dimensions = "x".join(
            str(dimension.value) if isinstance(dimension, StaticDim) else "?"
            for dimension in value_type.shape
        )
        element = emit_dtype(value_type.dtype)
        encoding = ""
        if dimension_id is None:
            if any(not isinstance(dimension, StaticDim) for dimension in value_type.shape):
                raise ValueError(
                    "dynamic tensor type emission requires a dimension identity resolver"
                )
        else:
            ids = [dimension_id(dimension) for dimension in value_type.shape]
            encoding = ", #intent.tensor_shape<[" + ", ".join(
                str(identity) for identity in ids
            ) + "]>"
        return f"tensor<{dimensions + 'x' if dimensions else ''}{element}{encoding}>"
    if isinstance(value_type, LogicalIndexType):
        return f"!intent.logical_index<{value_type.source_id}, {value_type.axis}>"
    if isinstance(value_type, DomainType):
        if value_type.origin_id is None:
            raise ValueError("domain type emission requires a bound origin ID")
        return (
            f"!intent.domain<{int(value_type.flavor)}, {value_type.rank}, "
            f"{value_type.origin_id}>"
        )
    if isinstance(value_type, RegionType):
        if value_type.origin_id is None:
            raise ValueError("region type emission requires a bound origin ID")
        return (
            f"!intent.region<{value_type.source_id}, {value_type.rank}, "
            f"{value_type.origin_id}>"
        )
    if isinstance(value_type, BufferType):
        if value_type.origin_id is None:
            raise ValueError("buffer type emission requires a bound origin ID")
        tensor = emit_type(TensorType(value_type.dtype, value_type.shape), dimension_id)
        return f"!intent.buffer<{tensor}, {value_type.origin_id}>"
    if isinstance(value_type, TupleType):
        components = "[" + ", ".join(
            emit_type(component, dimension_id) for component in value_type.components
        ) + "]"
        return f"!intent.tuple<{components}>"
    if isinstance(value_type, RecordType):
        names = "[" + ", ".join(quote(name) for name, _ in value_type.fields) + "]"
        types = "[" + ", ".join(
            emit_type(field_type, dimension_id) for _, field_type in value_type.fields
        ) + "]"
        return f"!intent.record<{names}, {types}>"
    if isinstance(value_type, ConstexprType):
        return f"!intent.constexpr<{emit_type(value_type.value_type, dimension_id)}>"
    if isinstance(value_type, EnumType):
        names = "[" + ", ".join(quote(name) for name, _ in value_type.members) + "]"
        values = "[" + ", ".join(
            str(value) for _, value in value_type.members
        ) + "]"
        return f"!intent.enum<{quote(value_type.name)}, {names}, {values}>"
    raise NotImplementedError(
        f"Kernel IR type {type(value_type).__name__} has no Intent MLIR type"
    )


def emit_dtype(dtype: DType) -> str:
    try:
        return _DTYPE_TYPES[dtype.name]
    except KeyError:
        raise ValueError(f"dtype {dtype.name!r} has no MLIR builtin spelling") from None


def emit_view_type(
    value_type: TensorType,
    kind: ViewKind,
    constraints: ViewConstraints | None,
    dimension_id: DimensionID,
) -> str:
    if constraints is None:
        raise ValueError("view type emission requires canonical constraints")
    access_code = int(kind)
    stride_values = constraints.strides or ()
    stride_text = "[" + ", ".join(
        "unit"
        if stride is None
        else quote(stride)
        if isinstance(stride, str)
        else f"{stride} : i64"
        for stride in stride_values
    ) + "]"
    constraint_text = (
        f"#intent.view_constraints<"
        f"{'true' if constraints.strides is not None else 'false'}, "
        f"{stride_text}, {quote(constraints.alias or '')}, "
        f"{'true' if constraints.noalias else 'false'}>"
    )
    return (
        f"!intent.view<{emit_type(value_type, dimension_id)}, {access_code}, "
        f"{constraint_text}>"
    )


def quote(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'
