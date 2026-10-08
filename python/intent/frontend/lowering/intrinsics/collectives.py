"""Reduce, scan, arg-reduce and histogram lowering to canonical operations."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from intent.frontend.mlir import MlirValue, RegionState
from intent.frontend.semantics import BinaryOperator
from intent.frontend.semantics import ComparePredicate
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import StaticDim
from intent.frontend.semantics import TensorType
from intent.frontend.semantics import ValueType
from intent.frontend.semantics.types import dims_compatible
from intent.language import DTypeCategory
from intent.language import bool as intent_bool
from intent.language import i32
from intent.language import index as intent_index
from intent.language import u32
from ..regions import pure_region
from ..products import ProductSchema
from ..ast.expressions import compile_time_value
from ..ast.model import ShapeDimension
from ..ast.model import StaticTuple
from .common import bind_call
from .common import bind_declared_call
from .common import optional_dtype
from .common import require_axes
from .common import require_dtype
from .common import require_static_bool
from .common import normalize_axes
from .callbacks import (
    builtin_combine_region,
    combine_region,
    lower_component_identity,
    rebuild_components,
    region_captures,
    source_components,
)

if TYPE_CHECKING:
    from ..ast.context import FunctionLowerer

@dataclass(frozen=True, slots=True)
class ReductionInputs:
    sources: tuple[MlirValue, ...]
    schema: ProductSchema
    axes: tuple[int, ...]
    identities: tuple[MlirValue, ...]
    captures: tuple[MlirValue, ...]
    combine: RegionState
    reduced_types: tuple[ValueType, ...]


def _prepare_reduction(lowerer: FunctionLowerer, name: str, node: ast.Call, bound, *, scan: bool) -> ReductionInputs:
    purpose = "scan" if scan else "reduce"
    generic = name == purpose
    source = lowerer.read_value(lowerer.lower_expression(bound["value"]), bound["value"])
    components, schema = source_components(lowerer, source, node)
    if not components or any(not isinstance(value.type, TensorType) for value in components):
        lowerer.error(node, f"I.{purpose} source components must be tensors")
    source_axes = require_axes(lowerer, bound["axis"])
    axes = normalize_axes(lowerer, source_axes, components[0].type.rank, node)
    if scan:
        if len(axes) != 1:
            lowerer.error(node, "I.scan requires exactly one axis")
    for component in components[1:]:
        component_axes = normalize_axes(lowerer, source_axes, component.type.rank, node)
        if component_axes != axes:
            lowerer.error(node, f"I.{purpose} axes must identify the same positions in every source component; "
                          "use non-negative axes when component ranks differ")
        if any(not dims_compatible(components[0].type.shape[axis], component.type.shape[axis])
               for axis in axes):
            lowerer.error(node, f"I.{purpose} source components must share each reduction or scan extent")
    if not generic and schema.kind != "scalar":
        lowerer.error(node, f"I.{name} requires one tensor")
    if name in ("reduce.any", "reduce.all") and components[0].type.dtype != intent_bool:
        lowerer.error(node, f"I.{name} requires a bool tensor")

    acc_dtype = optional_dtype(lowerer, bound.get("acc_dtype"))
    if acc_dtype is not None and len(components) != 1:
        lowerer.error(node, f"record {purpose} uses explicit field dtypes")
    if acc_dtype is None and name in ("reduce.sum", "cumsum"):
        acc_dtype = _sum_accumulator_dtype(components[0].type.dtype)
    if acc_dtype is not None and components[0].type.dtype != acc_dtype:
        component = components[0]
        components = (lowerer.emit(
            OperationKind.CAST, lowerer.location(node), operands=(component,),
            result_types=(TensorType(acc_dtype, component.type.shape),),
        ).results[0],)
    if name in ("reduce.sum", "reduce.max", "cumsum", "cummax") and components[0].type.dtype == intent_bool:
        lowerer.error(node, f"I.{name} requires a numeric accumulator dtype")
    reduced = set(axes)
    reduced_types = tuple(lowerer.value_result_type(
        component.type.dtype,
        tuple(dimension for axis, dimension in enumerate(component.type.shape) if axis not in reduced),
    ) for component in components)
    operator = {
        "reduce.max": BinaryOperator.MAXIMUM, "reduce.sum": BinaryOperator.ADD,
        "reduce.any": BinaryOperator.LOGICAL_OR, "reduce.all": BinaryOperator.LOGICAL_AND,
        "cumsum": BinaryOperator.ADD, "cummax": BinaryOperator.MAXIMUM,
    }.get(name)
    identity = (_builtin_identity(lowerer, components[0].type.dtype, operator, node)
                if operator is not None else
                lower_component_identity(lowerer, bound["identity"], components, schema, node))
    identities, identity_schema = source_components(lowerer, identity, node)
    if schema != identity_schema or len(components) != len(identities):
        lowerer.error(node, f"{purpose} source and identity schemas must match")
    for component, expected, identity in zip(components, reduced_types, identities):
        if identity.type not in (ScalarType(component.type.dtype), expected):
            lowerer.error(node, f"{purpose} identity must be an element scalar or the complete axis slice")
    captures, capture_bindings = region_captures(lowerer, bound.get("combine_operands"), node)
    accumulator_types = tuple(value.type for value in identities)
    if operator is not None:
        combine = builtin_combine_region(lowerer, accumulator_types, operator, node)
    else:
        combine, results = combine_region(
            lowerer, bound["combine"], accumulator_types, captures, capture_bindings, schema, node,
        )
        if results != accumulator_types:
            lowerer.error(node, f"{purpose} combine result schema must match identity")
    return ReductionInputs(components, schema, axes, identities, captures, combine, reduced_types)


def _emit_collective(lowerer, node, inputs: ReductionInputs, kind, result_types, attributes):
    operation = lowerer.emit(
        kind, lowerer.location(node),
        operand_groups=(inputs.sources, inputs.identities, inputs.captures),
        result_types=result_types,
        attributes=attributes,
        regions=(inputs.combine,),
    )
    return rebuild_components(lowerer, operation.results, inputs.schema, node)


def lower_reduce(lowerer: FunctionLowerer, name: str, node: ast.Call) -> MlirValue:
    bound = (bind_call(
        lowerer, node, ("value", "axis", "identity", "combine", "combine_operands", "acc_dtype"),
        required=("value", "axis", "identity", "combine"),
    ) if name == "reduce" else bind_declared_call(lowerer, node, name))
    inputs = _prepare_reduction(lowerer, name, node, bound, scan=False)
    return _emit_collective(lowerer, node, inputs, OperationKind.REDUCE,
                            inputs.reduced_types, {"axes": inputs.axes})


def lower_arg_reduce_max(lowerer: FunctionLowerer, node: ast.Call) -> StaticTuple:
    bound = bind_call(
        lowerer,
        node,
        ("value", "axis", "acc_dtype"),
        required=("value", "axis"),
    )
    source = lowerer.read_value(
        lowerer.lower_expression(bound["value"]), bound["value"]
    )
    if not isinstance(source.type, TensorType):
        lowerer.error(node, "I.arg_reduce.max input must be a tensor")
    axes = normalize_axes(
        lowerer,
        require_axes(lowerer, bound["axis"]),
        source.type.rank,
        node,
    )
    if len(axes) != 1:
        lowerer.error(node, "I.arg_reduce.max requires exactly one axis")
    acc_dtype = optional_dtype(lowerer, bound.get("acc_dtype")) or source.type.dtype
    if source.type.dtype != acc_dtype:
        source = lowerer.emit(
            OperationKind.CAST,
            lowerer.location(node),
            operands=(source,),
            result_types=(TensorType(acc_dtype, source.type.shape),),
        ).results[0]
    identity = _builtin_identity(lowerer, acc_dtype, BinaryOperator.MAXIMUM, node)
    indices = lowerer.emit(
        OperationKind.INDICES,
        lowerer.location(node),
        operands=(source,),
        result_types=(TensorType(intent_index, source.type.shape),),
        attributes={"tensor_axis": axes[0]},
    ).results[0]
    index_identity = lowerer.emit_literal((1 << 63) - 1, node, ScalarType(intent_index))
    combine = _argmax_combine_region(lowerer, ScalarType(acc_dtype), node)
    result_shape = tuple(
        dimension
        for axis, dimension in enumerate(source.type.shape)
        if axis != axes[0]
    )
    operation = lowerer.emit(
        OperationKind.REDUCE,
        lowerer.location(node),
        operand_groups=((source, indices), (identity, index_identity), ()),
        result_types=(
            lowerer.value_result_type(acc_dtype, result_shape),
            lowerer.value_result_type(intent_index, result_shape),
        ),
        attributes={
            "axes": axes,
        },
        regions=(combine,),
    )
    return StaticTuple(operation.results)


def lower_scan(lowerer: FunctionLowerer, node: ast.Call, name: str) -> MlirValue:
    bound = (bind_call(
        lowerer, node,
        ("value", "axis", "identity", "combine", "combine_operands", "inclusive", "reverse", "acc_dtype"),
        required=("value", "axis", "identity", "combine", "inclusive"),
    ) if name == "scan" else bind_declared_call(lowerer, node, name))
    inputs = _prepare_reduction(lowerer, name, node, bound, scan=True)
    return _emit_collective(
        lowerer, node, inputs, OperationKind.SCAN, tuple(value.type for value in inputs.sources),
        {"axis": inputs.axes[0], "inclusive": require_static_bool(lowerer, bound["inclusive"]),
         "reverse": require_static_bool(lowerer, bound["reverse"]) if "reverse" in bound else False},
    )


def lower_histogram(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("values", "bins", "valid", "count_dtype"),
        required=("values", "bins", "valid", "count_dtype"),
    )
    values = lowerer.read_value(
        lowerer.lower_expression(bound["values"]), bound["values"]
    )
    if not isinstance(values.type, TensorType) or values.type.dtype.category not in (
        DTypeCategory.SIGNED_INTEGER,
        DTypeCategory.UNSIGNED_INTEGER,
        DTypeCategory.INDEX,
    ):
        lowerer.error(node, "I.histogram values must be an integer tensor")
    bins_expression = lowerer.lower_expression(bound["bins"])
    bins = lowerer.materialize(
        bins_expression,
        bound["bins"],
        ScalarType(intent_index),
    )
    valid = lowerer.materialize(
        lowerer.lower_expression(bound["valid"]), bound["valid"]
    )
    valid_dtype, _ = lowerer.dtype_and_shape(valid.type, node)
    if valid_dtype != intent_bool:
        lowerer.error(node, "I.histogram valid must be bool")
    valid = lowerer.broadcast_value(valid, values.type.shape, node)
    count_dtype = require_dtype(lowerer, bound["count_dtype"])
    if count_dtype.category not in (
        DTypeCategory.SIGNED_INTEGER,
        DTypeCategory.UNSIGNED_INTEGER,
    ):
        lowerer.error(node, "I.histogram count dtype must be fixed-width integer")
    result_extent = (
        bins_expression.dimension
        if isinstance(bins_expression, ShapeDimension)
        else lowerer.fresh_dynamic_dimension("histogram_bins")
    )
    known, static_bins = compile_time_value(bins_expression)
    if known:
        if isinstance(static_bins, bool) or not isinstance(static_bins, int) or static_bins <= 0:
            lowerer.error(bound["bins"], "histogram bins must be positive")
        result_extent = StaticDim(static_bins)
    return lowerer.emit(
        OperationKind.HISTOGRAM,
        lowerer.location(node),
        operands=(values, bins, valid),
        result_types=(TensorType(count_dtype, (result_extent,)),),
        attributes={
            "result_dimension": lowerer.compiler.builder.dimension_id(result_extent)
        },
    ).results[0]


def _argmax_combine_region(
    lowerer: FunctionLowerer,
    value_type: ValueType,
    node: ast.AST,
):
    types = (value_type, ScalarType(intent_index))
    def combine(arguments):
        lhs_value, lhs_index, rhs_value, rhs_index = arguments
        greater = lowerer.emit(
            OperationKind.COMPARE,
            lowerer.location(node),
            operands=(lhs_value, rhs_value),
            result_types=(ScalarType(intent_bool),),
            attributes={"predicate": ComparePredicate.GT},
        ).results[0]
        equal = lowerer.emit(
            OperationKind.COMPARE,
            lowerer.location(node),
            operands=(lhs_value, rhs_value),
            result_types=(ScalarType(intent_bool),),
            attributes={"predicate": ComparePredicate.EQ},
        ).results[0]
        lower = lowerer.emit(
            OperationKind.COMPARE,
            lowerer.location(node),
            operands=(lhs_index, rhs_index),
            result_types=(ScalarType(intent_bool),),
            attributes={"predicate": ComparePredicate.LE},
        ).results[0]
        tied = lowerer.emit(
            OperationKind.BINARY,
            lowerer.location(node),
            operands=(equal, lower),
            result_types=(ScalarType(intent_bool),),
            attributes={"operator_kind": BinaryOperator.LOGICAL_AND},
        ).results[0]
        choose = lowerer.emit(
            OperationKind.BINARY,
            lowerer.location(node),
            operands=(greater, tied),
            result_types=(ScalarType(intent_bool),),
            attributes={"operator_kind": BinaryOperator.LOGICAL_OR},
        ).results[0]
        if value_type.dtype.category in (DTypeCategory.FLOAT, DTypeCategory.BFLOAT):
            lhs_nan, rhs_nan = (
                lowerer.emit(
                    OperationKind.COMPARE,
                    lowerer.location(node),
                    operands=(value, value),
                    result_types=(ScalarType(intent_bool),),
                    attributes={"predicate": ComparePredicate.NE},
                ).results[0]
                for value in (lhs_value, rhs_value)
            )
            nan_choice = lowerer.emit(
                OperationKind.SELECT,
                lowerer.location(node),
                operands=(rhs_nan, lower, lowerer.emit_literal(True, node)),
                result_types=(ScalarType(intent_bool),),
            ).results[0]
            choose = lowerer.emit(
                OperationKind.SELECT,
                lowerer.location(node),
                operands=(lhs_nan, nan_choice, choose),
                result_types=(ScalarType(intent_bool),),
            ).results[0]
        selected = tuple(
            lowerer.emit(
                OperationKind.SELECT,
                lowerer.location(node),
                operands=(choose, lhs, rhs),
                result_types=(result_type,),
            ).results[0]
            for lhs, rhs, result_type in zip(
                (lhs_value, lhs_index), (rhs_value, rhs_index), types
            )
        )
        return selected
    return pure_region(lowerer, types + types, node, combine, purpose="argmax combine")[0]


def _sum_accumulator_dtype(dtype):
    from intent.language import f32

    if dtype.name in ("i8", "i16"):
        return i32
    if dtype.name in ("u8", "u16"):
        return u32
    if dtype.name in ("f8e4m3fn", "f8e5m2", "f16", "bf16"):
        return f32
    return dtype


def _builtin_identity(lowerer, dtype, operator, node):
    if operator is BinaryOperator.LOGICAL_OR:
        value = False
    elif operator is BinaryOperator.LOGICAL_AND:
        value = True
    elif operator is BinaryOperator.ADD:
        value = 0
    elif operator is BinaryOperator.MAXIMUM:
        if dtype.category in (DTypeCategory.FLOAT, DTypeCategory.BFLOAT):
            value = -448.0 if dtype.name == "f8e4m3fn" else -float("inf")
        elif dtype.category in (DTypeCategory.SIGNED_INTEGER, DTypeCategory.INDEX):
            value = -(1 << (dtype.bits - 1))
        elif dtype.category is DTypeCategory.UNSIGNED_INTEGER:
            value = 0
        else:
            lowerer.error(node, "maximum identity requires a numeric dtype")
    else:
        raise NotImplementedError(f"builtin identity for {operator}")
    return lowerer.emit_literal(value, node, ScalarType(dtype))
