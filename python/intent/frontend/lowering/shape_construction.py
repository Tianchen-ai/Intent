from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from intent.frontend.semantics import DimExpr, ShapeExpr, ShapeExprKind, ShapeRelation, StaticDim

if TYPE_CHECKING:
    from intent.frontend.mlir import MlirValue


@dataclass(frozen=True, slots=True)
class LoweredShape:
    dimensions: tuple[DimExpr, ...]
    operands: tuple[MlirValue, ...]
    relation: ShapeRelation


class ShapeBuilder:
    def __init__(self, dimension_id: Callable[[DimExpr], int], first_operand_position: int):
        self._dimension_id = dimension_id
        self._first_operand_position = first_operand_position
        self._dimensions: list[DimExpr] = []
        self._operands: list[MlirValue] = []
        self._axes: list[ShapeExpr] = []

    def append_extent(self, dimension: DimExpr, value: MlirValue | None = None) -> None:
        if isinstance(dimension, StaticDim):
            kind, payload = ShapeExprKind.STATIC, dimension.value
        else:
            if value is None:
                raise TypeError("dynamic shape extent requires its canonical SSA value")
            kind = ShapeExprKind.SSA_EXTENT
            payload = self._first_operand_position + len(self._operands)
            self._operands.append(value)
        self._dimensions.append(dimension)
        self._axes.append(ShapeExpr(kind, self._dimension_id(dimension), payload))

    def append_inferred(self, dimension: DimExpr) -> None:
        self._dimensions.append(dimension)
        self._axes.append(ShapeExpr(ShapeExprKind.INFERRED, self._dimension_id(dimension), -1))

    def finish(self) -> LoweredShape:
        return LoweredShape(tuple(self._dimensions), tuple(self._operands), ShapeRelation(tuple(self._axes)))
