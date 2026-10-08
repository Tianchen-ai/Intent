from __future__ import annotations

from ..products import build_product

import ast
from typing import TYPE_CHECKING

from intent.frontend.semantics import BinaryOperator
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import RecordType
from intent.frontend.semantics import LogicalIndexType
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import TensorType
from intent.frontend.semantics import UnaryOperator
from intent.frontend.mlir import MlirValue
from intent.frontend.semantics import broadcast_shape
from intent.frontend.semantics import static_numel
from intent.language import DType
from intent.language import DTypeCategory
from intent.language import bool as intent_bool
from intent.language import f32 as intent_f32

from ..ast.model import Literal
from .common import bind_call
from .common import bind_declared_call
from .common import lower_shape
from .common import normalize_axes
from .common import require_axes
from .common import require_dtype
from .common import require_static_bool

if TYPE_CHECKING:
    from ..ast.context import FunctionLowerer


def lower_tensor_intrinsic(
    lowerer: FunctionLowerer,
    name: str,
    node: ast.Call,
) -> object:
    handlers = {
        "reshape": _reshape,
        "join": _join,
        "transpose": _transpose,
        "full": _full,
        "zeros": _zeros,
        "record": _record,
        "select": _select,
        "cast": _cast,
        "bitcast": _bitcast,
        "mask": _mask,
        "exp": lambda context, call: _unary(context, call, UnaryOperator.EXP),
        "exp2": lambda context, call: _unary(context, call, UnaryOperator.EXP2),
        "log": lambda context, call: _unary(context, call, UnaryOperator.LOG),
        "log1p": lambda context, call: _unary(context, call, UnaryOperator.LOG1P),
        "lgamma": lambda context, call: _unary(context, call, UnaryOperator.LGAMMA),
        "sin": lambda context, call: _unary(context, call, UnaryOperator.SIN),
        "asin": lambda context, call: _unary(context, call, UnaryOperator.ASIN),
        "cos": lambda context, call: _unary(context, call, UnaryOperator.COS),
        "floor": lambda context, call: _unary(context, call, UnaryOperator.FLOOR),
        "erf": lambda context, call: _unary(context, call, UnaryOperator.ERF),
        "erfc": lambda context, call: _unary(context, call, UnaryOperator.ERFC),
        "i0": lambda context, call: _unary(context, call, UnaryOperator.I0),
        "rsqrt": lambda context, call: _unary(context, call, UnaryOperator.RSQRT),
        "sqrt": lambda context, call: _unary(context, call, UnaryOperator.SQRT),
        "sigmoid": lambda context, call: _unary(context, call, UnaryOperator.SIGMOID),
        "tanh": lambda context, call: _unary(context, call, UnaryOperator.TANH),
        "abs": lambda context, call: _unary(context, call, UnaryOperator.ABS),
        "maximum": lambda context, call: _binary(context, call, BinaryOperator.MAXIMUM),
        "minimum": lambda context, call: _binary(context, call, BinaryOperator.MINIMUM),
        "maximum_num": lambda context, call: _binary(context, call, BinaryOperator.MAXIMUM_NUM),
        "minimum_num": lambda context, call: _binary(context, call, BinaryOperator.MINIMUM_NUM),
        "add": lambda context, call: _binary(context, call, BinaryOperator.ADD),
        "fdiv": lambda context, call: _binary(context, call, BinaryOperator.TRUE_DIVIDE),
    }
    handler = handlers.get(name)
    if handler is None:
        return NotImplemented
    return handler(lowerer, node)


def _reshape(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("value", "shape"), required=("value", "shape"))
    source = lowerer.read_value(lowerer.lower_expression(bound["value"]), bound["value"])
    if not isinstance(source.type, TensorType):
        lowerer.error(node, "I.reshape input must be a tensor")
    shape = lower_shape(lowerer, bound["shape"], first_operand_position=1)
    source_numel = static_numel(source.type.shape)
    result_numel = static_numel(shape.dimensions)
    if source_numel is not None and result_numel is not None and source_numel != result_numel:
        lowerer.error(node, f"I.reshape must preserve element count: {source_numel} to {result_numel}")
    operation = lowerer.emit(
        OperationKind.RESHAPE,
        lowerer.location(node),
        operands=(source, *shape.operands),
        result_types=(TensorType(source.type.dtype, shape.dimensions),),
        attributes={"shape": shape.relation},
    )
    return operation.results[0]


def _join(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("lhs", "rhs"), required=("lhs", "rhs"))
    lhs_expression = lowerer.lower_expression(bound["lhs"])
    rhs_expression = lowerer.lower_expression(bound["rhs"])
    lhs, rhs = lowerer.coerce_pair(lhs_expression, rhs_expression, node)
    lhs_dtype, lhs_shape = lowerer.dtype_and_shape(lhs.type, node)
    rhs_dtype, rhs_shape = lowerer.dtype_and_shape(rhs.type, node)
    if lhs_dtype != rhs_dtype:
        lowerer.error(node, "I.join inputs must have the same dtype")
    try:
        shape = broadcast_shape(lhs_shape, rhs_shape)
    except ValueError as error:
        lowerer.error(node, str(error))
    lhs = lowerer.broadcast_value(lhs, shape, node)
    rhs = lowerer.broadcast_value(rhs, shape, node)
    operation = lowerer.emit(
        OperationKind.JOIN,
        lowerer.location(node),
        operands=(lhs, rhs),
        result_types=(TensorType(lhs_dtype, shape + (2,)),),
    )
    return operation.results[0]


def _transpose(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("value", "permutation"), required=("value",))
    source = lowerer.read_value(lowerer.lower_expression(bound["value"]), bound["value"])
    if not isinstance(source.type, TensorType):
        lowerer.error(node, "I.transpose input must be a tensor")
    if "permutation" in bound:
        permutation = require_axes(lowerer, bound["permutation"])
    else:
        permutation = tuple(reversed(range(source.type.rank)))
    if sorted(permutation) != list(range(source.type.rank)):
        lowerer.error(node, "transpose permutation must cover every axis")
    result_shape = tuple(source.type.shape[axis] for axis in permutation)
    operation = lowerer.emit(
        OperationKind.TRANSPOSE,
        lowerer.location(node),
        operands=(source,),
        result_types=(TensorType(source.type.dtype, result_shape),),
        attributes={"permutation": permutation},
    )
    return operation.results[0]


def _full(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("shape", "fill", "dtype"),
        required=("shape", "fill", "dtype"),
    )
    dtype = require_dtype(lowerer, bound["dtype"])
    shape = lower_shape(lowerer, bound["shape"], first_operand_position=1)
    fill_expression = lowerer.lower_expression(bound["fill"])
    if isinstance(fill_expression, MlirValue) and isinstance(
        fill_expression.type, LogicalIndexType
    ) and dtype.category is DTypeCategory.INDEX:
        fill = fill_expression
    else:
        fill = lowerer.materialize(
            fill_expression,
            bound["fill"],
            ScalarType(dtype),
        )
    operation = lowerer.emit(
        OperationKind.FULL,
        lowerer.location(node),
        operands=(fill, *shape.operands),
        result_types=(TensorType(dtype, shape.dimensions),),
        attributes={"shape": shape.relation},
    )
    return operation.results[0]


def _zeros(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("shape", "dtype"), required=("shape", "dtype"))
    dtype = require_dtype(lowerer, bound["dtype"])
    shape = lower_shape(lowerer, bound["shape"], first_operand_position=1)
    fill = lowerer.emit_literal(0, node, ScalarType(dtype))
    operation = lowerer.emit(
        OperationKind.FULL,
        lowerer.location(node),
        operands=(fill, *shape.operands),
        result_types=(TensorType(dtype, shape.dimensions),),
        attributes={"shape": shape.relation},
    )
    return operation.results[0]


def _record(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    if node.args or not node.keywords:
        lowerer.error(node, "I.record requires one or more named fields")
    names: list[str] = []
    values: list[MlirValue] = []
    for keyword in node.keywords:
        if keyword.arg is None:
            lowerer.error(keyword, "record does not support **field expansion")
        names.append(keyword.arg)
        values.append(lowerer.read_value(lowerer.lower_expression(keyword.value), keyword.value))
    result_type = RecordType(tuple((name, value.type) for name, value in zip(names, values)))
    return build_product(lowerer, values, result_type, node)


def _cast(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("value", "dtype"), required=("value", "dtype"))
    source = lowerer.read_value(lowerer.lower_expression(bound["value"]), bound["value"])
    dtype = require_dtype(lowerer, bound["dtype"])
    source_dtype, shape = lowerer.dtype_and_shape(source.type, node)
    attributes = {}
    if source_dtype.category in (DTypeCategory.FLOAT, DTypeCategory.BFLOAT) and (
        dtype.category
        in (DTypeCategory.SIGNED_INTEGER, DTypeCategory.UNSIGNED_INTEGER)
    ):
        attributes["rounding"] = 0
    operation = lowerer.emit(
        OperationKind.CAST,
        lowerer.location(node),
        operands=(source,),
        result_types=(lowerer.value_result_type(dtype, shape, ranked=isinstance(source.type, TensorType)),),
        attributes=attributes,
    )
    return operation.results[0]


def _bitcast(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("value", "dtype"), required=("value", "dtype"))
    source = lowerer.read_value(lowerer.lower_expression(bound["value"]), bound["value"])
    dtype = require_dtype(lowerer, bound["dtype"])
    source_dtype, shape = lowerer.dtype_and_shape(source.type, node)
    if source_dtype.category in (DTypeCategory.BOOL, DTypeCategory.INDEX) or (
        dtype.category in (DTypeCategory.BOOL, DTypeCategory.INDEX)
    ):
        lowerer.error(node, "I.bitcast does not accept bool or index dtypes")
    if source_dtype.bits is None or source_dtype.bits != dtype.bits:
        lowerer.error(node, "I.bitcast requires equal source and result bit widths")
    operation = lowerer.emit(
        OperationKind.BITCAST,
        lowerer.location(node),
        operands=(source,),
        result_types=(lowerer.value_result_type(dtype, shape, ranked=isinstance(source.type, TensorType)),),
    )
    return operation.results[0]


def _mask(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("value", "valid", "fill"),
        required=("value", "valid", "fill"),
    )
    value_expression = lowerer.lower_expression(bound["value"])
    value = lowerer.read_value(value_expression, bound["value"])
    dtype, _ = lowerer.dtype_and_shape(value.type, node)
    predicate = lowerer.read_value(lowerer.lower_expression(bound["valid"]), bound["valid"])
    fill_expression = lowerer.lower_expression(bound["fill"])
    fill = lowerer.materialize(
        fill_expression,
        bound["fill"],
        ScalarType(dtype) if isinstance(fill_expression, Literal) else None,
    )
    fill = lowerer.read_value(fill, bound["fill"])
    _, value_shape = lowerer.dtype_and_shape(value.type, node)
    fill_dtype, fill_shape = lowerer.dtype_and_shape(fill.type, node)
    if fill_dtype != dtype:
        lowerer.error(node, "mask fill dtype must match value dtype")
    predicate_dtype, predicate_shape = lowerer.dtype_and_shape(predicate.type, node)
    if predicate_dtype != intent_bool:
        lowerer.error(node, "mask valid predicate must have bool dtype")
    try:
        result_shape = broadcast_shape(broadcast_shape(value_shape, fill_shape), predicate_shape)
    except ValueError as error:
        lowerer.error(node, str(error))
    ranked = any(isinstance(operand.type, TensorType) for operand in (value, predicate, fill))
    result_type = lowerer.value_result_type(dtype, result_shape, ranked=ranked)
    value = lowerer.broadcast_value(value, result_shape, node, ranked=ranked)
    predicate = lowerer.broadcast_value(predicate, result_shape, node, ranked=ranked)
    fill = lowerer.broadcast_value(fill, result_shape, node, ranked=ranked)
    operation = lowerer.emit(
        OperationKind.MASK,
        lowerer.location(node),
        operands=(value, predicate, fill),
        result_types=(result_type,),
    )
    return operation.results[0]


def _select(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("condition", "true_value", "false_value"),
        required=("condition", "true_value", "false_value"),
    )
    condition = lowerer.read_value(
        lowerer.lower_expression(bound["condition"]), bound["condition"]
    )
    lhs, rhs = lowerer.coerce_pair(
        lowerer.lower_expression(bound["true_value"]),
        lowerer.lower_expression(bound["false_value"]),
        node,
    )
    result_type = lowerer.broadcast_result_type(lhs.type, rhs.type, node)
    dtype, value_shape = lowerer.dtype_and_shape(result_type, node)
    condition_dtype, condition_shape = lowerer.dtype_and_shape(condition.type, node)
    if condition_dtype != intent_bool:
        lowerer.error(bound["condition"], "I.select condition must have bool dtype")
    try:
        result_shape = broadcast_shape(condition_shape, value_shape)
    except ValueError as error:
        lowerer.error(node, str(error))
    ranked = isinstance(result_type, TensorType) or isinstance(condition.type, TensorType)
    condition = lowerer.broadcast_value(condition, result_shape, node, ranked=ranked)
    lhs = lowerer.broadcast_value(lhs, result_shape, node, ranked=ranked)
    rhs = lowerer.broadcast_value(rhs, result_shape, node, ranked=ranked)
    return lowerer.emit(
        OperationKind.SELECT,
        lowerer.location(node),
        operands=(condition, lhs, rhs),
        result_types=(lowerer.value_result_type(dtype, result_shape, ranked=ranked),),
    ).results[0]


def _math_attributes(
    lowerer: FunctionLowerer,
    node: ast.Call,
    bound: dict[str, ast.AST],
    dtype: DType,
) -> dict[str, bool]:
    attributes = {
        name: require_static_bool(lowerer, bound[name])
        for name in ("approximate", "flush_to_zero")
        if name in bound
    }
    if attributes.get("flush_to_zero", False) and not attributes["approximate"]:
        lowerer.error(node, "flush_to_zero requires approximate=True")
    if attributes.get("approximate", False) and dtype != intent_f32:
        lowerer.error(node, "explicit approximate math requires f32 operands")
    return {name: value for name, value in attributes.items() if value}


def _unary(
    lowerer: FunctionLowerer,
    node: ast.Call,
    operator: UnaryOperator,
) -> MlirValue:
    if operator in (UnaryOperator.EXP2, UnaryOperator.TANH):
        bound = bind_declared_call(
            lowerer, node, "exp2" if operator == UnaryOperator.EXP2 else "tanh"
        )
    else:
        bound = bind_call(lowerer, node, ("value",), required=("value",))
    source = lowerer.read_value(lowerer.lower_expression(bound["value"]), bound["value"])
    dtype, _ = lowerer.dtype_and_shape(source.type, node)
    operation = lowerer.emit(
        OperationKind.UNARY,
        lowerer.location(node),
        operands=(source,),
        result_types=(source.type,),
        attributes={
            "operator_kind": operator,
            **_math_attributes(lowerer, node, bound, dtype),
        },
    )
    return operation.results[0]


def _binary(
    lowerer: FunctionLowerer,
    node: ast.Call,
    operator: BinaryOperator,
) -> MlirValue:
    bound = (
        bind_declared_call(lowerer, node, "fdiv")
        if operator == BinaryOperator.TRUE_DIVIDE
        else bind_call(lowerer, node, ("lhs", "rhs"), required=("lhs", "rhs"))
    )
    lhs_expression = lowerer.lower_expression(bound["lhs"])
    rhs_expression = lowerer.lower_expression(bound["rhs"])
    lhs, rhs = lowerer.coerce_pair(lhs_expression, rhs_expression, node)
    result_type = lowerer.broadcast_result_type(lhs.type, rhs.type, node)
    dtype, result_shape = lowerer.dtype_and_shape(result_type, node)
    lhs = lowerer.broadcast_value(lhs, result_shape, node, ranked=isinstance(result_type, TensorType))
    rhs = lowerer.broadcast_value(rhs, result_shape, node, ranked=isinstance(result_type, TensorType))
    operation = lowerer.emit(
        OperationKind.BINARY,
        lowerer.location(node),
        operands=(lhs, rhs),
        result_types=(result_type,),
        attributes={
            "operator_kind": operator,
            **_math_attributes(lowerer, node, bound, dtype),
        },
    )
    return operation.results[0]
