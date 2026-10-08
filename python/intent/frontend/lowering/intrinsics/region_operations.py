"""Region fold and scan sources, summaries and slice result relations."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from intent.frontend.mlir import MlirValue, RegionState
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import TensorType
from intent.frontend.semantics import ValueType
from intent.frontend.semantics.types import dims_compatible
from ..products import map_product_type, project_product
from ..ast.model import ConstexprBinding
from ..ast.model import StaticTuple
from .common import bind_call
from .common import require_static_int
from .callbacks import helper_region, region_captures

if TYPE_CHECKING:
    from ..ast.context import FunctionLowerer

@dataclass(frozen=True, slots=True)
class RegionSummary:
    sources: tuple[MlirValue, ...]
    axis: int
    captures: tuple[MlirValue, ...]
    capture_bindings: tuple[ConstexprBinding | None, ...]
    slice_types: tuple[TensorType, ...]
    summarize: RegionState
    summary_types: tuple[ValueType, ...]
    identity: tuple[MlirValue, ...]
    combine: RegionState


def _prepare_region_summary(lowerer, bound, node, purpose: str) -> RegionSummary:
    sources = _tensor_sources(lowerer, bound["source"], node)
    axis = _shared_source_axis(lowerer, sources, bound["axis"], node)
    captures, capture_bindings = region_captures(lowerer, bound.get("operands"), node)
    slice_types = _slice_types(lowerer, sources, axis, purpose)
    summarize, summary_types = helper_region(
        lowerer, bound["summarize"], slice_types + tuple(value.type for value in captures),
        (None,) * len(slice_types) + capture_bindings, node,
    )
    values = _values(lowerer, bound["identity"], node)
    if len(values) != len(summary_types):
        lowerer.error(node, f"{purpose} identity arity must match summary schema")
    identity = tuple(_align_value_schema(lowerer, value, target, bound["identity"])
                     for value, target in zip(values, summary_types))
    identity_types = tuple(value.type for value in identity)
    if identity_types != summary_types:
        lowerer.error(node, f"{purpose} summarize result {summary_types} must match identity schema {identity_types}")
    combine, combine_types = helper_region(
        lowerer, bound["combine"], summary_types + summary_types,
        (None,) * (2 * len(summary_types)), node,
    )
    if combine_types != summary_types:
        lowerer.error(node, f"{purpose} combine result must match summary schema")
    return RegionSummary(sources, axis, captures, capture_bindings, slice_types,
                         summarize, summary_types, identity, combine)


def lower_region_fold(lowerer: FunctionLowerer, node: ast.Call) -> object:
    bound = bind_call(
        lowerer, node, ("source", "axis", "summarize", "combine", "identity", "operands"),
        required=("source", "axis", "summarize", "combine", "identity"),
    )
    summary = _prepare_region_summary(lowerer, bound, node, "region_fold")
    operation = lowerer.emit(
        OperationKind.REGION_FOLD, lowerer.location(node),
        operand_groups=(summary.sources, summary.identity, summary.captures),
        result_types=summary.summary_types,
        attributes={"axis": summary.axis},
        regions=(summary.summarize, summary.combine),
    )
    return operation.results[0] if len(operation.results) == 1 else StaticTuple(operation.results)


def lower_region_scan(lowerer: FunctionLowerer, node: ast.Call) -> StaticTuple:
    bound = bind_call(
        lowerer, node,
        ("source", "axis", "summarize", "combine", "identity", "initial_state", "apply", "emit", "operands"),
        required=("source", "axis", "summarize", "combine", "identity", "initial_state", "apply", "emit"),
    )
    summary = _prepare_region_summary(lowerer, bound, node, "region_scan")
    initial = _values(lowerer, bound["initial_state"], node)
    initial_types = tuple(value.type for value in initial)
    apply, state_types = helper_region(
        lowerer, bound["apply"], summary.summary_types + initial_types,
        (None,) * (len(summary.summary_types) + len(initial_types)), node,
    )
    if state_types != initial_types:
        lowerer.error(node, "region_scan apply result must match initial-state schema")
    emit, slice_output_types = helper_region(
        lowerer, bound["emit"], summary.slice_types + state_types + tuple(value.type for value in summary.captures),
        (None,) * (len(summary.slice_types) + len(state_types)) + summary.capture_bindings, node,
    )
    source_extent = summary.sources[0].type.shape[summary.axis]
    output_types = tuple(_replace_slice_extent(value_type, summary.slice_types[0].shape[summary.axis], source_extent)
                         for value_type in slice_output_types)
    operation = lowerer.emit(
        OperationKind.REGION_SCAN, lowerer.location(node),
        operand_groups=(summary.sources, summary.identity, initial, summary.captures),
        result_type_groups=(output_types, state_types),
        attributes={"axis": summary.axis},
        regions=(summary.summarize, summary.combine, apply, emit),
    )
    outputs = operation.results[:len(output_types)]
    states = operation.results[len(output_types):]
    output = outputs[0] if len(outputs) == 1 else StaticTuple(outputs)
    state = states[0] if len(states) == 1 else StaticTuple(states)
    return StaticTuple((output, state))


def _values(
    lowerer: FunctionLowerer,
    node: ast.AST,
    call_node: ast.AST,
) -> tuple[MlirValue, ...]:
    expression = lowerer.lower_expression(node)
    elements = expression.elements if isinstance(expression, StaticTuple) else (expression,)
    return tuple(lowerer.materialize(element, call_node) for element in elements)


def _align_value_schema(lowerer: FunctionLowerer, value: MlirValue,
                        target: ValueType, node: ast.AST) -> MlirValue:
    def align_leaf(component, expected):
        if isinstance(component.type, TensorType) and isinstance(expected, TensorType):
            if (component.type.dtype != expected.dtype or component.type.rank != expected.rank
                    or any(not dims_compatible(source, destination)
                           for source, destination in zip(component.type.shape, expected.shape))):
                lowerer.error(node, "structured identity tensor does not match its summary schema")
            return lowerer.broadcast_value(component, expected.shape, node)
        if component.type == expected:
            return component
        lowerer.error(node, f"structured identity value {component.type} does not match summary schema {expected}")
    return project_product(lowerer, value, target, node, align_leaf)


def _tensor_sources(
    lowerer: FunctionLowerer,
    node: ast.AST,
    call_node: ast.AST,
) -> tuple[MlirValue, ...]:
    expression = lowerer.lower_expression(node)
    elements = expression.elements if isinstance(expression, StaticTuple) else (expression,)
    values = tuple(lowerer.read_value(element, call_node) for element in elements)
    if not values or any(not isinstance(value.type, TensorType) for value in values):
        lowerer.error(call_node, "region source must contain one or more tensors")
    return values


def _shared_source_axis(
    lowerer: FunctionLowerer,
    sources: tuple[MlirValue, ...],
    axis_node: ast.AST,
    call_node: ast.AST,
) -> int:
    axis = require_static_int(lowerer, axis_node)
    rank = sources[0].type.rank
    if not -rank <= axis < rank:
        lowerer.error(axis_node, "region source axis is outside rank")
    axis %= rank
    extent = sources[0].type.shape[axis]
    for source in sources[1:]:
        if source.type.rank <= axis or not dims_compatible(source.type.shape[axis], extent):
            lowerer.error(call_node, "region source components must share the source extent")
    return axis


def _slice_types(
    lowerer: FunctionLowerer,
    sources: tuple[MlirValue, ...],
    axis: int,
    role: str,
) -> tuple[TensorType, ...]:
    extent = lowerer.fresh_dynamic_dimension(f"{role}_slice")
    return tuple(
        TensorType(
            source.type.dtype,
            tuple(
                extent if position == axis else dimension
                for position, dimension in enumerate(source.type.shape)
            ),
        )
        for source in sources
    )


def _replace_slice_extent(value_type: ValueType, slice_extent: object, full_extent: object) -> ValueType:
    def replace_leaf(leaf):
        if not isinstance(leaf, TensorType):
            raise ValueError("region_scan emit result must be a tensor or tensor record")
        matches = [axis for axis, dim in enumerate(leaf.shape) if dim == slice_extent]
        if len(matches) != 1:
            raise ValueError("region_scan emit result must contain its source slice extent exactly once")
        shape = list(leaf.shape)
        shape[matches[0]] = full_extent
        return TensorType(leaf.dtype, tuple(shape))
    return map_product_type(value_type, replace_leaf)
