from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from intent.frontend.semantics import DomainType
from intent.frontend.semantics import RegionType
from intent.frontend.semantics import StaticDim
from intent.frontend.semantics import SymbolDim
from intent.frontend.semantics import is_integer
from intent.frontend.mlir import MlirValue
from intent.language import DType
from intent.language.signatures import INTRINSIC_SIGNATURES

from ..ast.expressions import compile_time_value
from ..ast.model import ShapeDimension
from ..ast.model import ShapeValue
from ..ast.model import StaticTuple
from ..shape_construction import ShapeBuilder

if TYPE_CHECKING:
    from ..ast.context import FunctionLowerer
    from ..shape_construction import LoweredShape


def bind_call(
    lowerer: FunctionLowerer,
    node: ast.Call,
    positional: tuple[str, ...],
    *,
    required: tuple[str, ...] = (),
) -> dict[str, ast.AST]:
    if len(node.args) > len(positional):
        lowerer.error(node, "too many positional intrinsic arguments")
    bound = {name: value for name, value in zip(positional, node.args)}
    for keyword in node.keywords:
        if keyword.arg is None:
            lowerer.error(keyword, "**kwargs expansion is not supported in Intent source")
        if keyword.arg not in positional:
            lowerer.error(keyword, f"unknown intrinsic argument {keyword.arg!r}")
        if keyword.arg in bound:
            lowerer.error(keyword, f"duplicate intrinsic argument {keyword.arg!r}")
        bound[keyword.arg] = keyword.value
    missing = [name for name in required if name not in bound]
    if missing:
        lowerer.error(node, f"missing intrinsic argument {missing[0]!r}")
    return bound


def require_dtype(lowerer: FunctionLowerer, node: ast.AST) -> DType:
    value = lowerer.lower_expression(node)
    if not isinstance(value, DType):
        lowerer.error(node, "dtype argument must be an Intent dtype such as I.f32")
    return value


def bind_declared_call(
    lowerer: FunctionLowerer, node: ast.Call, name: str
) -> dict[str, ast.AST]:
    keywords: dict[str, ast.AST] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            lowerer.error(keyword, "**kwargs expansion is not supported in Intent source")
        if keyword.arg in keywords:
            lowerer.error(keyword, f"duplicate I.{name} argument {keyword.arg!r}")
        keywords[keyword.arg] = keyword.value
    signature = INTRINSIC_SIGNATURES[name]
    try:
        arguments = signature.bind(*node.args, **keywords)
    except TypeError as error:
        lowerer.error(node, f"I.{name}{signature}: {error}")
    arguments.apply_defaults()
    return {
        parameter: value
        if isinstance(value, ast.AST)
        else ast.copy_location(ast.Constant(value=value), node)
        for parameter, value in arguments.arguments.items()
        if value is not None
    }


def optional_dtype(
    lowerer: FunctionLowerer, node: ast.AST | None
) -> DType | None:
    if node is None:
        return None
    value = lowerer.lower_expression(node)
    if value is not None and not isinstance(value, DType):
        lowerer.error(node, "acc_dtype must be an Intent dtype or None")
    return value


def require_static_bool(lowerer: FunctionLowerer, node: ast.AST) -> bool:
    expression = lowerer.lower_expression(node)
    known, value = compile_time_value(expression)
    if not known or not isinstance(value, bool):
        lowerer.error(node, "argument must be a compile-time bool")
    return value


def require_static_int(lowerer: FunctionLowerer, node: ast.AST) -> int:
    expression = lowerer.lower_expression(node)
    known, value = compile_time_value(expression)
    if not known or isinstance(value, bool) or not isinstance(value, int):
        lowerer.error(node, "argument must be a compile-time integer")
    return value


def require_axes(lowerer: FunctionLowerer, node: ast.AST) -> tuple[int, ...]:
    expression = lowerer.lower_expression(node)
    known, value = compile_time_value(expression)
    if known and isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if isinstance(expression, StaticTuple):
        axes: list[int] = []
        for element in expression.elements:
            known, value = compile_time_value(element)
            if not known or isinstance(value, bool) or not isinstance(value, int):
                lowerer.error(node, "axes must contain compile-time integers")
            axes.append(value)
        if axes:
            return tuple(axes)
    lowerer.error(node, "axis must be a compile-time integer or non-empty tuple of compile-time integers")


def normalize_axes(
    lowerer: FunctionLowerer,
    axes: tuple[int, ...],
    rank: int,
    node: ast.AST,
) -> tuple[int, ...]:
    normalized: list[int] = []
    for axis in axes:
        if not -rank <= axis < rank:
            lowerer.error(node, f"axis {axis} is outside rank {rank}")
        value = axis % rank
        if value in normalized:
            lowerer.error(node, "axes must be unique")
        normalized.append(value)
    return tuple(normalized)


def lower_shape(
    lowerer: FunctionLowerer,
    node: ast.AST,
    *,
    first_operand_position: int,
) -> LoweredShape:
    expression = lowerer.lower_expression(node)
    elements = (
        tuple(expression.dimensions)
        if isinstance(expression, ShapeValue)
        else expression.elements
        if isinstance(expression, StaticTuple)
        else (expression,)
    )
    shape = ShapeBuilder(lowerer.compiler.builder.dimension_id, first_operand_position)

    for element in elements:
        if isinstance(element, ShapeDimension):
            shape.append_extent(
                element.dimension,
                None
                if isinstance(element.dimension, StaticDim)
                else lowerer.materialize_dimension(element, node),
            )
        elif isinstance(element, MlirValue) and isinstance(
            element.type, (DomainType, RegionType)
        ):
            for axis, dimension in enumerate(lowerer.dynamic_shape_for_region(element)):
                shape.append_extent(
                    dimension,
                    None
                    if isinstance(dimension, StaticDim)
                    else lowerer.materialize_dimension(
                        ShapeDimension(dimension, element, axis), node
                    ),
                )
        elif isinstance(element, MlirValue) and is_integer(element.type):
            # Shape operands are nonnegative.  Under that contract, the extent
            # of an existing unit-step domain(0, element) is exactly element.
            dimension = lowerer.integer_shape_dimension(element)
            shape.append_extent(
                dimension,
                None if isinstance(dimension, StaticDim) else element,
            )
        else:
            known, value = compile_time_value(element)
            if known and isinstance(value, int) and not isinstance(value, bool):
                if value < 0:
                    lowerer.error(node, "shape dimensions must be non-negative")
                shape.append_extent(StaticDim(value))
            elif isinstance(element, str):
                dimension = SymbolDim(element)
                shape.append_extent(
                    dimension,
                    lowerer.materialize_shape_extent(dimension, node),
                )
            elif element is Ellipsis:
                shape.append_inferred(lowerer.fresh_dynamic_dimension("reshape_inferred"))
            else:
                lowerer.error(
                    node,
                    "shape elements must be static, symbolic, integer SSA, shape-derived, or region",
                )
    return shape.finish()


def optional_node(bound: dict[str, ast.AST], name: str) -> ast.AST | None:
    return bound.get(name)
