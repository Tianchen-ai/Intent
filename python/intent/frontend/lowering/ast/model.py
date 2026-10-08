from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TypeAlias

from intent.frontend.semantics import DimExpr
from intent.frontend.semantics import OperationKind
from intent.frontend.mlir import MlirValue


@dataclass(frozen=True, slots=True)
class Literal:
    value: bool | int | float


@dataclass(frozen=True, slots=True)
class ConstexprBinding:
    python_value: object
    ir_value: MlirValue


@dataclass(frozen=True, slots=True)
class ShapeDimension:
    dimension: DimExpr
    source: MlirValue
    axis: int


@dataclass(frozen=True, slots=True)
class ShapeValue:
    dimensions: tuple[ShapeDimension, ...]


@dataclass(frozen=True, slots=True)
class IterationSpec:
    opcode: OperationKind
    source: MlirValue


@dataclass(frozen=True, slots=True)
class RaggedSpec:
    outer: MlirValue
    members: MlirValue
    offsets: MlirValue
    mapping: MlirValue | None = None


@dataclass(frozen=True, slots=True)
class SparseFormatSpec:
    kind: int
    compression_axis: int
    logical_extent: MlirValue


@dataclass(frozen=True, slots=True)
class StaticTuple:
    elements: tuple[Expression, ...]


Expression: TypeAlias = (
    MlirValue
    | Literal
    | ConstexprBinding
    | ShapeDimension
    | ShapeValue
    | IterationSpec
    | RaggedSpec
    | SparseFormatSpec
    | StaticTuple
    | object
)


@dataclass(slots=True)
class LoopContext:
    opcode: OperationKind
    carried_names: tuple[str, ...]


__all__ = [
    "ConstexprBinding",
    "Expression",
    "IterationSpec",
    "Literal",
    "LoopContext",
    "RaggedSpec",
    "ShapeDimension",
    "ShapeValue",
    "StaticTuple",
    "SparseFormatSpec",
]
