from __future__ import annotations

import ast

from intent.frontend.semantics import BinaryOperator, OperationKind
from intent.frontend.semantics import StaticDim, TensorType, broadcast_shape
from intent.language import DTypeCategory

from ..ast.model import ShapeDimension, SparseFormatSpec
from ..shape_construction import ShapeBuilder
from .common import bind_declared_call, require_dtype, require_static_bool, require_static_int
from .contractions import parse_scaled_format, emit_contract, emit_scaled_contract, emit_sparse_contract


def lower_matrix_intrinsic(lowerer, name: str, node: ast.Call):
    if name not in (
        "dot", "matvec", "vecmat", "matmul", "outer", "scaled_matmul", "sparse_matmul"
    ):
        return NotImplemented
    bound = bind_declared_call(lowerer, node, name)
    if name == "scaled_matmul":
        values = tuple(_tensor(lowerer, bound[key]) for key in ("lhs", "lhs_scale", "rhs", "rhs_scale"))
        group = require_static_int(lowerer, bound["group_size"])
        return emit_scaled_contract(
            lowerer, *values, parse_scaled_format(lowerer, bound["lhs_format"]),
            parse_scaled_format(lowerer, bound["rhs_format"]), group, group,
            ((1, 0), (2, 1)), (), require_dtype(lowerer, bound["acc_dtype"]), node,
        )
    if name == "sparse_matmul":
        compressed = _tensor(lowerer, bound["compressed"])
        rhs = _tensor(lowerer, bound["rhs"])
        metadata = lowerer.read_value(lowerer.lower_expression(bound["metadata"]), bound["metadata"])
        format_value = lowerer.lower_expression(bound["format"])
        if not isinstance(format_value, SparseFormatSpec):
            lowerer.error(node, "I.sparse_matmul requires a typed I.sparse format")
        if compressed.type.rank != 2 or rhs.type.rank != 2 or format_value.compression_axis != 1:
            lowerer.error(node, "I.sparse_matmul requires rank-two matrices compressed along lhs axis 1")
        return emit_sparse_contract(
            lowerer, compressed, metadata, rhs, format_value, ((1, 0),), (),
            require_dtype(lowerer, bound["acc_dtype"]), node,
        )

    keys = ("matrix", "vector") if name == "matvec" else ("vector", "matrix") if name == "vecmat" else ("lhs", "rhs")
    lhs, rhs = (_tensor(lowerer, bound[key]) for key in keys)
    if lhs.type.dtype != rhs.type.dtype or lhs.type.dtype.category is DTypeCategory.BOOL:
        lowerer.error(node, f"I.{name} requires matching numeric dtypes; cast explicitly")
    if name in ("dot", "outer"):
        if lhs.type.rank != 1 or rhs.type.rank != 1:
            lowerer.error(node, f"I.{name} requires two rank-one tensors")
        if name == "outer":
            return _outer(lowerer, lhs, rhs, node)
        return emit_contract(
            lowerer, lhs, rhs, ((0, 0),), (), require_dtype(lowerer, bound["acc_dtype"]), node
        )

    lhs_core = 1 if name == "vecmat" else 2
    rhs_core = 1 if name == "matvec" else 2
    if lhs.type.rank < lhs_core or rhs.type.rank < rhs_core:
        lowerer.error(node, f"I.{name} needs core ranks {lhs_core} and {rhs_core}")
    lhs_transpose = "transpose" if name == "matvec" else "transpose_lhs"
    rhs_transpose = "transpose" if name == "vecmat" else "transpose_rhs"
    if lhs_core == 2 and require_static_bool(lowerer, bound[lhs_transpose]):
        lhs = _transpose_matrix(lowerer, lhs, node)
    if rhs_core == 2 and require_static_bool(lowerer, bound[rhs_transpose]):
        rhs = _transpose_matrix(lowerer, rhs, node)
    try:
        batch_shape = broadcast_shape(lhs.type.shape[:-lhs_core], rhs.type.shape[:-rhs_core])
    except ValueError as error:
        lowerer.error(node, f"I.{name} batch shapes: {error}")
    lhs = lowerer.broadcast_value(lhs, batch_shape + lhs.type.shape[-lhs_core:], node)
    rhs = lowerer.broadcast_value(rhs, batch_shape + rhs.type.shape[-rhs_core:], node)
    return emit_contract(
        lowerer, lhs, rhs,
        ((lhs.type.rank - 1, rhs.type.rank - rhs_core),),
        tuple((axis, axis) for axis in range(len(batch_shape))),
        require_dtype(lowerer, bound["acc_dtype"]), node,
    )


def _tensor(lowerer, node):
    value = lowerer.read_value(lowerer.lower_expression(node), node)
    if not isinstance(value.type, TensorType):
        lowerer.error(node, "matrix operation operand must be a tensor")
    return value


def _transpose_matrix(lowerer, value, node):
    permutation = tuple(range(value.type.rank - 2)) + (value.type.rank - 1, value.type.rank - 2)
    return lowerer.emit(
        OperationKind.TRANSPOSE, lowerer.location(node), operands=(value,),
        result_types=(TensorType(value.type.dtype, tuple(value.type.shape[axis] for axis in permutation)),),
        attributes={"permutation": permutation},
    ).results[0]


def _outer(lowerer, lhs, rhs, node):
    dimension = lhs.type.shape[0]
    shape = ShapeBuilder(lowerer.compiler.builder.dimension_id, first_operand_position=1)
    shape.append_extent(
        dimension,
        None if isinstance(dimension, StaticDim) else lowerer.materialize_dimension(ShapeDimension(dimension, lhs, 0), node),
    )
    shape.append_extent(StaticDim(1))
    lowered = shape.finish()
    column = lowerer.emit(
        OperationKind.RESHAPE, lowerer.location(node), operands=(lhs, *lowered.operands),
        result_types=(TensorType(lhs.type.dtype, lowered.dimensions),),
        attributes={"shape": lowered.relation},
    ).results[0]
    result_shape = (dimension, rhs.type.shape[0])
    return lowerer.emit(
        OperationKind.BINARY, lowerer.location(node),
        operands=(lowerer.broadcast_value(column, result_shape, node), lowerer.broadcast_value(rhs, result_shape, node)),
        result_types=(TensorType(lhs.type.dtype, result_shape),),
        attributes={"operator_kind": BinaryOperator.MULTIPLY},
    ).results[0]
