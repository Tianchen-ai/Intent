"""Canonical dense, scaled and sparse contraction construction."""
from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from intent.frontend.mlir import MlirValue
from intent.frontend.mlir.attributes import SparseFormatAttribute
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import ScaledFormatKind
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import StaticDim
from intent.frontend.semantics import TensorType
from intent.frontend.semantics.types import dims_compatible
from intent.language import bool as intent_bool
from intent.language import index as intent_index
from intent.language.builtins import ScaledFormat
from ..ast.expressions import compile_time_value
from ..ast.model import ShapeDimension
from ..ast.model import SparseFormatSpec
from ..ast.model import StaticTuple
from .common import bind_call
from .common import require_dtype
from .common import require_static_int
from .common import normalize_axes

if TYPE_CHECKING:
    from ..ast.context import FunctionLowerer

def lower_contract(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("lhs", "rhs", "reduce", "acc_dtype", "batch"),
        required=("lhs", "rhs", "reduce", "acc_dtype"),
    )
    lhs = lowerer.read_value(lowerer.lower_expression(bound["lhs"]), bound["lhs"])
    rhs = lowerer.read_value(lowerer.lower_expression(bound["rhs"]), bound["rhs"])
    if not isinstance(lhs.type, TensorType) or not isinstance(rhs.type, TensorType):
        lowerer.error(node, "I.contract operands must be tensors")
    return emit_contract(
        lowerer, lhs, rhs,
        _axis_pairs(lowerer, bound["reduce"], "reduction"),
        _axis_pairs(lowerer, bound["batch"], "batch") if "batch" in bound else (),
        require_dtype(lowerer, bound["acc_dtype"]), node,
    )


def emit_contract(lowerer, lhs, rhs, reduce, batch, acc_dtype, node):
    if lhs.type.dtype != rhs.type.dtype or lhs.type.dtype == intent_bool:
        lowerer.error(node, "contraction requires matching numeric input dtypes; cast explicitly")
    reduce, batch, result_shape = _paired_relations(lowerer, lhs, rhs, reduce, batch, node)
    return lowerer.emit(
        OperationKind.CONTRACT,
        lowerer.location(node),
        operands=(lhs, rhs),
        result_types=(TensorType(acc_dtype, result_shape),),
        attributes={"reduce": reduce, "batch": batch},
    ).results[0]


def lower_scaled_contract(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        (
            "lhs",
            "lhs_scale",
            "rhs",
            "rhs_scale",
            "lhs_format",
            "rhs_format",
            "lhs_group_size",
            "rhs_group_size",
            "reduce",
            "batch",
            "acc_dtype",
        ),
        required=(
            "lhs",
            "lhs_scale",
            "rhs",
            "rhs_scale",
            "lhs_format",
            "rhs_format",
            "lhs_group_size",
            "rhs_group_size",
            "reduce",
            "acc_dtype",
        ),
    )
    lhs = lowerer.read_value(lowerer.lower_expression(bound["lhs"]), bound["lhs"])
    lhs_scale = lowerer.read_value(
        lowerer.lower_expression(bound["lhs_scale"]), bound["lhs_scale"]
    )
    rhs = lowerer.read_value(lowerer.lower_expression(bound["rhs"]), bound["rhs"])
    rhs_scale = lowerer.read_value(
        lowerer.lower_expression(bound["rhs_scale"]), bound["rhs_scale"]
    )
    if any(
        not isinstance(value.type, TensorType)
        for value in (lhs, lhs_scale, rhs, rhs_scale)
    ):
        lowerer.error(node, "I.scaled_contract operands and scales must be tensors")
    lhs_format = parse_scaled_format(lowerer, bound["lhs_format"])
    rhs_format = parse_scaled_format(lowerer, bound["rhs_format"])
    lhs_group = require_static_int(lowerer, bound["lhs_group_size"])
    rhs_group = require_static_int(lowerer, bound["rhs_group_size"])
    reduce = _axis_pairs(lowerer, bound["reduce"], "reduction")
    batch = _axis_pairs(lowerer, bound["batch"], "batch") if "batch" in bound else ()
    return emit_scaled_contract(
        lowerer, lhs, lhs_scale, rhs, rhs_scale, lhs_format, rhs_format,
        lhs_group, rhs_group, reduce, batch, require_dtype(lowerer, bound["acc_dtype"]), node,
    )


def emit_scaled_contract(
    lowerer, lhs, lhs_scale, rhs, rhs_scale, lhs_format, rhs_format,
    lhs_group, rhs_group, reduce, batch, acc_dtype, node,
):
    if lhs_group <= 0 or lhs_group != rhs_group:
        lowerer.error(node, "scaled-contract requires one equal positive group size")
    reduce = tuple(
        (
            normalize_axes(lowerer, (left,), lhs.type.rank, node)[0],
            normalize_axes(lowerer, (right,), rhs.type.rank, node)[0],
        )
        for left, right in reduce
    )
    if (
        lhs.type.rank != 3
        or lhs_scale.type.rank != 2
        or rhs.type.rank != 3
        or rhs_scale.type.rank != 2
        or reduce != ((1, 0), (2, 1))
        or batch
    ):
        lowerer.error(
            node,
            "scaled-contract requires lhs/lhs-scale [M,G,C]/[M,G], "
            "rhs/rhs-scale [G,C,N]/[N,G], reduce=((1,0),(2,1)), and no batch axes",
        )
    lhs_carrier = lhs_group // (2 if lhs_format is ScaledFormatKind.E2M1 else 1)
    rhs_carrier = rhs_group // (2 if rhs_format is ScaledFormatKind.E2M1 else 1)
    if (
        lhs_group % (2 if lhs_format is ScaledFormatKind.E2M1 else 1)
        or rhs_group % (2 if rhs_format is ScaledFormatKind.E2M1 else 1)
        or lhs.type.shape[2] != StaticDim(lhs_carrier)
        or rhs.type.shape[1] != StaticDim(rhs_carrier)
        or not dims_compatible(lhs.type.shape[0], lhs_scale.type.shape[0])
        or not dims_compatible(lhs.type.shape[1], lhs_scale.type.shape[1])
        or not dims_compatible(lhs.type.shape[1], rhs.type.shape[0])
        or not dims_compatible(lhs.type.shape[1], rhs_scale.type.shape[1])
        or not dims_compatible(rhs.type.shape[2], rhs_scale.type.shape[0])
    ):
        lowerer.error(node, "scaled-contract operands violate the closed scale-axis relation")
    result_shape = (lhs.type.shape[0], rhs.type.shape[2])
    return lowerer.emit(
        OperationKind.SCALED_CONTRACT,
        lowerer.location(node),
        operands=(lhs, lhs_scale, rhs, rhs_scale),
        result_types=(TensorType(acc_dtype, result_shape),),
        attributes={
            "reduce": reduce,
            "batch": batch,
            "lhs_format": lhs_format,
            "rhs_format": rhs_format,
            "lhs_group_size": lhs_group,
            "rhs_group_size": rhs_group,
        },
    ).results[0]


def lower_sparse_format(
    lowerer: FunctionLowerer,
    name: str,
    node: ast.Call,
) -> SparseFormatSpec:
    bound = bind_call(
        lowerer,
        node,
        ("compression_axis", "logical_extent"),
        required=("compression_axis", "logical_extent"),
    )
    axis = require_static_int(lowerer, bound["compression_axis"])
    extent = lowerer.materialize(
        lowerer.lower_expression(bound["logical_extent"]),
        bound["logical_extent"],
        ScalarType(intent_index),
    )
    kind = {"sparse.one_of_two": 0, "sparse.two_of_four": 1}[name]
    return SparseFormatSpec(kind, axis, extent)


def lower_sparse_contract(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("compressed", "metadata", "dense_rhs", "format", "reduce", "batch", "acc_dtype"),
        required=("compressed", "metadata", "dense_rhs", "format", "reduce", "acc_dtype"),
    )
    compressed = lowerer.read_value(
        lowerer.lower_expression(bound["compressed"]), bound["compressed"]
    )
    metadata = lowerer.read_value(
        lowerer.lower_expression(bound["metadata"]), bound["metadata"]
    )
    rhs = lowerer.read_value(
        lowerer.lower_expression(bound["dense_rhs"]), bound["dense_rhs"]
    )
    format_value = lowerer.lower_expression(bound["format"])
    if not isinstance(format_value, SparseFormatSpec):
        lowerer.error(bound["format"], "sparse contract format must be a typed I.sparse schema")
    if not isinstance(compressed.type, TensorType) or not isinstance(rhs.type, TensorType):
        lowerer.error(node, "sparse contract data operands must be tensors")
    return emit_sparse_contract(
        lowerer, compressed, metadata, rhs, format_value,
        _axis_pairs(lowerer, bound["reduce"], "reduction"),
        _axis_pairs(lowerer, bound["batch"], "batch") if "batch" in bound else (),
        require_dtype(lowerer, bound["acc_dtype"]), node,
    )


def emit_sparse_contract(lowerer, compressed, metadata, rhs, format_value, reduce, batch, acc_dtype, node):
    reduce, batch, result_shape = _paired_relations(
        lowerer, compressed, rhs, reduce, batch, node,
        logical_lhs_axis=format_value.compression_axis,
    )
    return lowerer.emit(
        OperationKind.SPARSE_CONTRACT,
        lowerer.location(node),
        operands=(compressed, metadata, rhs, format_value.logical_extent),
        result_types=(TensorType(acc_dtype, result_shape),),
        attributes={
            "format": SparseFormatAttribute(
                format_value.kind, format_value.compression_axis
            ),
            "reduce": reduce,
            "batch": batch,
        },
    ).results[0]


def lower_sparse_contract_2to4(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("compressed_lhs", "metadata", "rhs", "acc_dtype"),
        required=("compressed_lhs", "metadata", "rhs", "acc_dtype"),
    )
    compressed = lowerer.read_value(
        lowerer.lower_expression(bound["compressed_lhs"]),
        bound["compressed_lhs"],
    )
    metadata = lowerer.read_value(
        lowerer.lower_expression(bound["metadata"]), bound["metadata"]
    )
    rhs = lowerer.read_value(lowerer.lower_expression(bound["rhs"]), bound["rhs"])
    if (
        not isinstance(compressed.type, TensorType)
        or not isinstance(rhs.type, TensorType)
        or compressed.type.rank != 2
        or rhs.type.rank != 2
    ):
        lowerer.error(node, "I.sparse_contract_2to4 requires rank-two data operands")
    logical_extent = lowerer.materialize_dimension(
        ShapeDimension(rhs.type.shape[0], rhs, 0), bound["rhs"]
    )
    return lowerer.emit(
        OperationKind.SPARSE_CONTRACT,
        lowerer.location(node),
        operands=(compressed, metadata, rhs, logical_extent),
        result_types=(
            TensorType(
                require_dtype(lowerer, bound["acc_dtype"]),
                (compressed.type.shape[0], rhs.type.shape[1]),
            ),
        ),
        attributes={
            "format": SparseFormatAttribute(1, 1),
            "reduce": ((1, 0),),
            "batch": (),
        },
    ).results[0]


def _axis_pairs(
    lowerer: FunctionLowerer,
    node: ast.AST,
    purpose: str,
) -> tuple[tuple[int, int], ...]:
    expression = lowerer.lower_expression(node)
    if not isinstance(expression, StaticTuple):
        lowerer.error(node, f"contract {purpose}= must be a tuple of axis pairs")
    pairs: list[tuple[int, int]] = []
    for element in expression.elements:
        if not isinstance(element, StaticTuple) or len(element.elements) != 2:
            lowerer.error(node, f"contract {purpose} entry must be an axis pair")
        pair: list[int] = []
        for axis in element.elements:
            known, value = compile_time_value(axis)
            if not known or isinstance(value, bool) or not isinstance(value, int):
                lowerer.error(node, f"contract {purpose} axes must be compile-time integers")
            pair.append(value)
        pairs.append((pair[0], pair[1]))
    return tuple(pairs)


def _paired_relations(
    lowerer: FunctionLowerer,
    lhs: MlirValue,
    rhs: MlirValue,
    reduce: tuple[tuple[int, int], ...],
    batch: tuple[tuple[int, int], ...],
    node: ast.AST,
    *,
    logical_lhs_axis: int | None = None,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[object, ...]]:
    if not reduce:
        lowerer.error(node, "contract requires at least one reduction pair")
    lhs_reduce: set[int] = set()
    rhs_reduce: set[int] = set()
    lhs_batch: set[int] = set()
    rhs_batch: set[int] = set()
    normalized_reduce: list[tuple[int, int]] = []
    normalized_batch: list[tuple[int, int]] = []
    for left, right in reduce:
        left = normalize_axes(lowerer, (left,), lhs.type.rank, node)[0]
        right = normalize_axes(lowerer, (right,), rhs.type.rank, node)[0]
        if left in lhs_reduce or right in rhs_reduce:
            lowerer.error(node, "contract reduction axes must be unique")
        lhs_reduce.add(left)
        rhs_reduce.add(right)
        normalized_reduce.append((left, right))
    for left, right in batch:
        left = normalize_axes(lowerer, (left,), lhs.type.rank, node)[0]
        right = normalize_axes(lowerer, (right,), rhs.type.rank, node)[0]
        if left in lhs_reduce or right in rhs_reduce or left in lhs_batch or right in rhs_batch:
            lowerer.error(node, "contract batch axes must be unique and disjoint")
        if not dims_compatible(lhs.type.shape[left], rhs.type.shape[right]):
            lowerer.error(node, "contract paired batch extents are incompatible")
        lhs_batch.add(left)
        rhs_batch.add(right)
        normalized_batch.append((left, right))
    for left, right in normalized_reduce:
        if left != logical_lhs_axis and not dims_compatible(
            lhs.type.shape[left], rhs.type.shape[right]
        ):
            lowerer.error(node, "contract paired reduction extents are incompatible")
    result_shape = tuple(
        dimension
        for axis, dimension in enumerate(lhs.type.shape)
        if axis not in lhs_reduce
    ) + tuple(
        dimension
        for axis, dimension in enumerate(rhs.type.shape)
        if axis not in rhs_reduce and axis not in rhs_batch
    )
    return tuple(normalized_reduce), tuple(normalized_batch), result_shape


def parse_scaled_format(lowerer: FunctionLowerer, node: ast.AST) -> ScaledFormatKind:
    value = lowerer.lower_expression(node)
    if not isinstance(value, ScaledFormat):
        lowerer.error(node, "scaled-contract format must be I.e2m1/I.e4m3/I.e8m0")
    mapping = {
        "e2m1": ScaledFormatKind.E2M1,
        "e4m3": ScaledFormatKind.E4M3,
        "e8m0": ScaledFormatKind.E8M0,
    }
    try:
        return mapping[value.name]
    except KeyError:
        lowerer.error(node, "unknown scaled-contract format")
