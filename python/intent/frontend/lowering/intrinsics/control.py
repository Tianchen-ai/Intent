from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from intent.frontend.mlir import MlirValue
from intent.frontend.semantics import BufferType
from intent.frontend.semantics import BinaryOperator
from intent.frontend.semantics import DomainFlavor
from intent.frontend.semantics import DomainType
from intent.frontend.semantics import LogicalIndexType
from intent.frontend.semantics import IndexRelation
from intent.frontend.semantics import IndexTerm
from intent.frontend.semantics import IndexTermKind
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import RegionType
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import StaticDim
from intent.frontend.semantics import TensorType
from intent.frontend.semantics.types import is_integer
from intent.language import index as intent_index
from intent.language.dtypes import DTypeCategory

from ..ast.expressions import compile_time_value
from ..ast.model import IterationSpec
from ..ast.model import Literal
from ..ast.model import RaggedSpec
from ..ast.model import ShapeDimension
from ..ast.model import StaticTuple
from .common import bind_call
from .common import require_static_int

if TYPE_CHECKING:
    from ..ast.context import FunctionLowerer


def lower_control_intrinsic(
    lowerer: FunctionLowerer,
    name: str,
    node: ast.Call,
) -> object:
    handlers = {
        "domain": _domain,
        "parallel": _parallel,
        "indices": _indices,
        "end": _region_end,
        "assume_in_bounds": _assume_in_bounds,
        "ragged": _ragged,
        "members": _members,
    }
    handler = handlers.get(name)
    if handler is None:
        return NotImplemented
    return handler(lowerer, node)


def _domain(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("start", "stop", "step"),
        required=("start", "stop"),
    )
    expressions = [
        lowerer.lower_expression(bound["start"]),
        lowerer.lower_expression(bound["stop"]),
    ]
    if "step" in bound:
        step = lowerer.lower_expression(bound["step"])
        known, value = compile_time_value(step)
        if not known or isinstance(value, bool) or not isinstance(value, int) or value != 1:
            lowerer.error(bound["step"], "I.domain requires a compile-time unit step")
        expressions.append(step)
    runtime = any(not compile_time_value(expression)[0] for expression in expressions)
    operands = tuple(_integer_value(lowerer, expression, node) for expression in expressions)
    flavor = DomainFlavor.RUNTIME if runtime else (
        DomainFlavor.STRIDED if len(operands) == 3 else DomainFlavor.DENSE
    )
    start_known, start_value = compile_time_value(expressions[0])
    stop_known, stop_value = compile_time_value(expressions[1])
    if (
        start_known
        and stop_known
        and isinstance(start_value, int)
        and not isinstance(start_value, bool)
        and isinstance(stop_value, int)
        and not isinstance(stop_value, bool)
    ):
        extent = StaticDim(max(0, stop_value - start_value))
    elif start_known and start_value == 0 and isinstance(expressions[1], ShapeDimension):
        extent = expressions[1].dimension
    elif start_known and start_value == 0 and isinstance(expressions[1], MlirValue):
        stop = operands[1]
        extent = lowerer.integer_shape_dimension(
            stop, lowerer.fresh_dynamic_dimension("domain_extent")
        )
    else:
        extent = lowerer.fresh_dynamic_dimension("domain_extent")
        if all(lowerer.dtype_and_shape(value.type, node)[0] == intent_index
               for value in operands[:2]):
            length = lowerer.emit(
                OperationKind.BINARY,
                lowerer.location(node),
                operands=(operands[1], operands[0]),
                result_types=(ScalarType(intent_index),),
                attributes={"operator_kind": BinaryOperator.SUBTRACT},
            ).results[0]
            bounds = lowerer.integer_expression_bounds(length)
            if bounds is not None and bounds[0] >= 0:
                extent = lowerer.integer_shape_dimension(length, extent)
    operation = lowerer.emit(
        OperationKind.DOMAIN,
        lowerer.location(node),
        operands=operands,
        result_types=(DomainType(flavor, 1),),
        attributes={
            "extent_dimensions": (
                lowerer.compiler.builder.dimension_id(extent),
            )
        },
    )
    result = operation.results[0]
    return result


def _parallel(lowerer: FunctionLowerer, node: ast.Call) -> IterationSpec:
    return IterationSpec(OperationKind.PARALLEL, _iteration_source(lowerer, node))


def _iteration_source(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("source",), required=("source",))
    expression = lowerer.lower_expression(bound["source"])
    if isinstance(expression, StaticTuple):
        domains = tuple(
            lowerer.materialize(element, bound["source"])
            for element in expression.elements
        )
        if not domains or any(not isinstance(value.type, DomainType) for value in domains):
            lowerer.error(node, "domain product requires one or more domains")
        extent_shape = tuple(
            dimension
            for domain in domains
            for dimension in lowerer.dynamic_shape_for_region(domain)
        )
        operation = lowerer.emit(
            OperationKind.DOMAIN_PRODUCT,
            lowerer.location(node),
            operands=domains,
            result_types=(
                DomainType(
                    DomainFlavor.PRODUCT,
                    sum(value.type.rank for value in domains),
                ),
            ),
            attributes={
                "extent_dimensions": tuple(
                    lowerer.compiler.builder.dimension_id(dimension)
                    for dimension in extent_shape
                )
            },
        )
        result = operation.results[0]
        return result
    source = lowerer.materialize(expression, bound["source"])
    if not isinstance(source.type, (DomainType, RegionType)):
        lowerer.error(node, "iteration source must be a domain or source-derived subregion")
    return source


def _indices(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("region",), required=("region",))
    source = lowerer.materialize(
        lowerer.lower_expression(bound["region"]), bound["region"]
    )
    if not isinstance(source.type, (DomainType, RegionType)):
        lowerer.error(node, "I.indices requires a logical domain or subregion")
    return lowerer.emit(
        OperationKind.INDICES,
        lowerer.location(node),
        operands=(source,),
        result_types=(
            TensorType(intent_index, lowerer.dynamic_shape_for_region(source)),
        ),
    ).results[0]


def _region_end(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("region",), required=("region",))
    source = lowerer.materialize(
        lowerer.lower_expression(bound["region"]), bound["region"]
    )
    if not isinstance(source.type, (DomainType, RegionType)) or source.type.rank != 1:
        lowerer.error(node, "I.end requires a rank-one logical domain or subregion")
    return lowerer.emit(
        OperationKind.REGION_END,
        lowerer.location(node),
        operands=(source,),
        result_types=(ScalarType(intent_index),),
    ).results[0]


def _assume_in_bounds(lowerer: FunctionLowerer, node: ast.Call) -> StaticTuple:
    bound = bind_call(
        lowerer,
        node,
        ("index", "view", "axis"),
        required=("index", "view", "axis"),
    )
    index = lowerer.materialize(lowerer.lower_expression(bound["index"]), bound["index"])
    view = lowerer.materialize(lowerer.lower_expression(bound["view"]), bound["view"])
    scalar_index = isinstance(index.type, (ScalarType, LogicalIndexType)) and is_integer(
        index.type
    )
    tensor_index = isinstance(index.type, TensorType) and index.type.dtype.category in (
        DTypeCategory.SIGNED_INTEGER,
        DTypeCategory.UNSIGNED_INTEGER,
        DTypeCategory.INDEX,
    )
    if not scalar_index and not tensor_index:
        lowerer.error(bound["index"], "assumed index must be an integer scalar or tensor")
    if view.view_kind is not None and isinstance(view.type, TensorType):
        shape = view.type.shape
    elif isinstance(view.type, BufferType):
        shape = view.type.shape
    else:
        lowerer.error(bound["view"], "assumed bound must reference an ABI view or logical buffer")
    axis = require_static_int(lowerer, bound["axis"])
    if not -len(shape) <= axis < len(shape):
        lowerer.error(bound["axis"], "assumed bound axis is outside the target rank")
    lowerer.emit(
        OperationKind.ASSUME_IN_BOUNDS,
        lowerer.location(node),
        operands=(index, view),
        attributes={"axis": axis % len(shape)},
    )
    return StaticTuple(())


def _ragged(lowerer: FunctionLowerer, node: ast.Call) -> RaggedSpec:
    bound = bind_call(
        lowerer,
        node,
        ("outer", "members", "offsets", "indices"),
        required=("outer", "members", "offsets"),
    )
    outer = lowerer.materialize(lowerer.lower_expression(bound["outer"]), bound["outer"])
    members = lowerer.materialize(
        lowerer.lower_expression(bound["members"]), bound["members"]
    )
    offsets = lowerer.read_value(
        lowerer.lower_expression(bound["offsets"]), bound["offsets"]
    )
    if not isinstance(outer.type, DomainType):
        lowerer.error(node, "ragged outer must be a domain")
    if not isinstance(members.type, DomainType) or members.type.rank != 1:
        lowerer.error(node, "ragged members must be a rank-one domain")
    mapping = None
    if "indices" in bound:
        mapping = lowerer.read_value(
            lowerer.lower_expression(bound["indices"]), bound["indices"]
        )
    return RaggedSpec(outer, members, offsets, mapping)


def _members(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(lowerer, node, ("region",), required=("region",))
    region = lowerer.materialize(
        lowerer.lower_expression(bound["region"]), bound["region"]
    )
    if not isinstance(region.type, RegionType):
        lowerer.error(node, "I.members requires an offsets-derived subregion")
    coordinates = _indices(lowerer, _single_argument_call(node, bound["region"]))
    mapping = lowerer.ragged_mappings.get(region)
    if mapping is None:
        return coordinates
    return lowerer.emit(
        OperationKind.GATHER,
        lowerer.location(node),
        operand_groups=((mapping,), (coordinates,), (), ()),
        result_types=(TensorType(mapping.type.dtype, coordinates.type.shape),),
        attributes={
            "index": IndexRelation(
                1,
                len(coordinates.type.shape),
                tuple(
                    lowerer.compiler.builder.dimension_id(dimension)
                    for dimension in coordinates.type.shape
                ),
                (IndexTerm(IndexTermKind.VALUE_INDEX, (0,)),),
            )
        },
    ).results[0]


def _single_argument_call(template: ast.Call, argument: ast.AST) -> ast.Call:
    call = ast.Call(func=template.func, args=[argument], keywords=[])
    ast.copy_location(call, template)
    return call


def _integer_value(
    lowerer: FunctionLowerer,
    expression: object,
    node: ast.AST,
) -> MlirValue:
    known, static_value = compile_time_value(expression)
    if known:
        if isinstance(static_value, bool) or not isinstance(static_value, int):
            lowerer.error(node, "domain extent must be an integer constexpr")
        return lowerer.materialize(Literal(static_value), node, ScalarType(intent_index))
    value = lowerer.materialize(expression, node)
    if not is_integer(value.type):
        lowerer.error(node, "domain extent must be integer/index")
    return value
