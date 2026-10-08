from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from intent.frontend.semantics import AtomicOrdering
from intent.frontend.semantics import AtomicRMWKind
from intent.frontend.semantics import BinaryOperator
from intent.frontend.semantics import BufferType
from intent.frontend.semantics import Effect
from intent.frontend.semantics import EffectKind
from intent.frontend.semantics import LogicalIndexType
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import ResourceKind
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import RecordType
from intent.frontend.semantics import TensorType
from intent.frontend.mlir import MlirValue
from intent.frontend.semantics import broadcast_shape
from intent.frontend.semantics import dims_compatible
from intent.frontend.semantics.types import is_integer
from intent.language import f32
from intent.language import u32
from intent.language import u64
from intent.language import index as intent_index
from intent.language import bool as intent_bool

from ..ast.indexing import lower_index
from ..ast.indexing import validate_indexed_value
from ..ast.model import Literal
from ..ast.model import StaticTuple
from .common import bind_call
from .common import lower_shape
from .common import require_dtype
from .callbacks import builtin_combine_region

if TYPE_CHECKING:
    from ..ast.context import FunctionLowerer


def lower_memory_intrinsic(
    lowerer: FunctionLowerer,
    name: str,
    node: ast.Call,
) -> object:
    handlers = {
        "gather": _gather,
        "scatter_unique": lambda context, call: _scatter(context, call, reduce=False),
        "scatter_reduce": lambda context, call: _scatter(context, call, reduce=True),
        "buffer": _buffer,
        "store": _store,
        "mutable_load": _mutable_load,
        "atomic.load": _atomic_load,
        "atomic.store": _atomic_store,
        "atomic.exchange": lambda context, call: _atomic_rmw(context, call, AtomicRMWKind.EXCHANGE),
        "atomic.add": lambda context, call: _atomic_rmw(context, call, AtomicRMWKind.ADD),
        "atomic.max": lambda context, call: _atomic_rmw(context, call, AtomicRMWKind.MAX),
        "atomic.min": lambda context, call: _atomic_rmw(context, call, AtomicRMWKind.MIN),
        "atomic.and_": lambda context, call: _atomic_rmw(context, call, AtomicRMWKind.AND),
        "atomic.or_": lambda context, call: _atomic_rmw(context, call, AtomicRMWKind.OR),
        "atomic.xor": lambda context, call: _atomic_rmw(context, call, AtomicRMWKind.XOR),
        "atomic.compare_exchange": _atomic_compare_exchange,
        "random.bits": _random_bits,
        "random.uniform": _random_uniform,
    }
    handler = handlers.get(name)
    if handler is None:
        return NotImplemented
    return handler(lowerer, node)


def _gather(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("source", "index", "valid", "fill"),
        required=("source", "index"),
    )
    source = lowerer.materialize(lowerer.lower_expression(bound["source"]), bound["source"])
    if not isinstance(source.type, TensorType):
        lowerer.error(node, "I.gather source must be a tensor/view")
    lowered = lower_index(lowerer, source, bound["index"])
    if "valid" in bound:
        valid = lowerer.materialize(lowerer.lower_expression(bound["valid"]), bound["valid"])
    else:
        valid = lowerer.emit_literal(True, node, ScalarType(intent_bool))
    if "fill" in bound:
        fill_expression = lowerer.lower_expression(bound["fill"])
        fill = lowerer.materialize(
            fill_expression,
            bound["fill"],
            ScalarType(source.type.dtype) if isinstance(fill_expression, Literal) else None,
        )
    else:
        fill = lowerer.emit_literal(
            False if source.type.dtype == intent_bool else 0,
            node,
            ScalarType(source.type.dtype),
        )
    valid_dtype, valid_shape = lowerer.dtype_and_shape(valid.type, node)
    if valid_dtype.name != "bool":
        lowerer.error(node, "gather valid predicate must have bool dtype")
    fill_dtype, fill_shape = lowerer.dtype_and_shape(fill.type, node)
    if fill_dtype != source.type.dtype:
        lowerer.error(node, "gather fill dtype must match source dtype")
    result_shape = tuple(lowered.result_shape)
    for shape, subject in ((valid_shape, "valid"), (fill_shape, "fill")):
        try:
            broadcasted = tuple(broadcast_shape(shape, result_shape))
        except ValueError as error:
            lowerer.error(node, str(error))
        if len(broadcasted) != len(result_shape) or not all(
            dims_compatible(source, destination)
            for source, destination in zip(broadcasted, result_shape)
        ):
            lowerer.error(node, f"gather {subject} cannot broadcast to indexed shape")
    valid = lowerer.broadcast_value(valid, result_shape, node)
    fill = lowerer.broadcast_value(fill, result_shape, node)
    effects = ()
    if source.view_kind is not None:
        lowerer.require_readable_view(source, node)
        effects = (Effect(EffectKind.READ, ResourceKind.EXTERNAL_VIEW, source),)
    attributes: dict[str, object] = {"index": lowered.relation}
    operation = lowerer.emit(
        OperationKind.VIEW_LOAD if source.view_kind is not None else OperationKind.GATHER,
        lowerer.location(node),
        operand_groups=((source,), lowered.operands, (valid,), (fill,)),
        result_types=(lowerer.value_result_type(source.type.dtype, result_shape),),
        attributes=attributes,
        effects=effects,
    )
    return operation.results[0]


def _scatter(lowerer: FunctionLowerer, node: ast.Call, *, reduce: bool) -> StaticTuple:
    positional = ("destination", "index", "value", "combine")
    required = ("destination", "index", "value", "combine") if reduce else (
        "destination",
        "index",
        "value",
    )
    bound = bind_call(lowerer, node, positional, required=required)
    destination = lowerer.materialize(
        lowerer.lower_expression(bound["destination"]), bound["destination"]
    )
    if not isinstance(destination.type, TensorType) or destination.view_kind is None:
        lowerer.error(node, "scatter destination must be an external output/InOut view")
    lowerer.require_writable_view(destination, node)
    lowered = lower_index(lowerer, destination, bound["index"])
    value_expression = lowerer.lower_expression(bound["value"])
    value = lowerer.materialize(
        value_expression,
        bound["value"],
        ScalarType(destination.type.dtype) if isinstance(value_expression, Literal) else None,
    )
    value = validate_indexed_value(
        lowerer, value, lowered.result_shape, node, expected_dtype=destination.type.dtype
    )
    operands = (destination, *lowered.operands, value)
    attributes: dict[str, object] = {"index": lowered.relation}
    regions = ()
    if reduce:
        combine = lowerer.lower_expression(bound["combine"])
        if getattr(combine, "name", None) != "add":
            lowerer.error(bound["combine"], "scatter_reduce currently requires the typed I.add combine")
        regions = (
            builtin_combine_region(
                lowerer,
                (ScalarType(destination.type.dtype),),
                BinaryOperator.ADD,
                node,
            ),
        )
    lowerer.emit(
        OperationKind.SCATTER_REDUCE if reduce else OperationKind.SCATTER_UNIQUE,
        lowerer.location(node),
        operands=operands,
        attributes=attributes,
        regions=regions,
        effects=(Effect(EffectKind.WRITE, ResourceKind.EXTERNAL_VIEW, destination),),
    )
    return StaticTuple(())


def _buffer(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("shape", "dtype", "init"),
        required=("shape", "dtype"),
    )
    dtype = require_dtype(lowerer, bound["dtype"])
    initializer: MlirValue | None = None
    if "init" in bound:
        expression = lowerer.lower_expression(bound["init"])
        if expression is not None:
            initializer = lowerer.materialize(
                expression,
                bound["init"],
                ScalarType(dtype) if isinstance(expression, Literal) else None,
            )
            initializer = lowerer.read_value(initializer, bound["init"])
    shape = lower_shape(lowerer, bound["shape"], first_operand_position=0)
    attributes: dict[str, object] = {"shape": shape.relation}
    operation = lowerer.emit(
        OperationKind.BUFFER,
        lowerer.location(node),
        operand_groups=(shape.operands, (initializer,) if initializer is not None else ()),
        result_types=(BufferType(dtype, shape.dimensions),),
        attributes=attributes,
    )
    return operation.results[0]


def _store(lowerer: FunctionLowerer, node: ast.Call) -> StaticTuple:
    bound = bind_call(
        lowerer,
        node,
        ("target", "index", "value"),
        required=("target", "index", "value"),
    )
    target = lowerer.materialize(lowerer.lower_expression(bound["target"]), bound["target"])
    if not isinstance(target.type, (TensorType, BufferType)):
        lowerer.error(node, "I.store target must be a view or logical buffer")
    lowered = lower_index(lowerer, target, bound["index"])
    value_expression = lowerer.lower_expression(bound["value"])
    value = lowerer.materialize(
        value_expression,
        bound["value"],
        ScalarType(target.type.dtype) if isinstance(value_expression, Literal) else None,
    )
    value = validate_indexed_value(
        lowerer, value, lowered.result_shape, node, expected_dtype=target.type.dtype
    )
    operands = (target, *lowered.operands, value)
    attributes: dict[str, object] = {"index": lowered.relation}
    if isinstance(target.type, BufferType):
        opcode = OperationKind.BUFFER_STORE
        effect = Effect(EffectKind.WRITE, ResourceKind.LOGICAL_BUFFER, target)
    else:
        if target.view_kind is None:
            lowerer.error(node, "pure tensor SSA cannot be mutated")
        lowerer.require_writable_view(target, node)
        opcode = OperationKind.VIEW_STORE
        effect = Effect(EffectKind.WRITE, ResourceKind.EXTERNAL_VIEW, target)
    lowerer.emit(
        opcode,
        lowerer.location(node),
        operands=operands,
        attributes=attributes,
        effects=(effect,),
    )
    return StaticTuple(())


def _mutable_load(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("target", "index"),
        required=("target", "index"),
    )
    target = lowerer.materialize(lowerer.lower_expression(bound["target"]), bound["target"])
    if not isinstance(target.type, (TensorType, BufferType)):
        lowerer.error(node, "I.mutable_load target must be a view or logical buffer")
    lowered = lower_index(lowerer, target, bound["index"])
    attributes: dict[str, object] = {"index": lowered.relation}
    if isinstance(target.type, BufferType):
        opcode = OperationKind.BUFFER_LOAD
        resource = ResourceKind.LOGICAL_BUFFER
    else:
        if target.view_kind is None:
            lowerer.error(node, "mutable_load tensor target must be an external view")
        lowerer.require_readable_view(target, node)
        opcode = OperationKind.VIEW_LOAD
        resource = ResourceKind.EXTERNAL_VIEW
    layout = (
        {"operands": (target, *lowered.operands)}
        if isinstance(target.type, BufferType)
        else {"operand_groups": ((target,), lowered.operands, (), ())}
    )
    operation = lowerer.emit(
        opcode,
        lowerer.location(node),
        **layout,
        result_types=(lowerer.value_result_type(target.type.dtype, lowered.result_shape),),
        attributes=attributes,
        effects=(Effect(EffectKind.READ, resource, target),),
    )
    return operation.results[0]


def _atomic_load(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("target", "index", "order"),
        required=("target", "index", "order"),
    )
    target, lowered = _atomic_address(lowerer, bound, node)
    order = _ordering(lowerer, bound["order"])
    if order not in (AtomicOrdering.RELAXED, AtomicOrdering.ACQUIRE):
        lowerer.error(bound["order"], "atomic load allows relaxed or acquire order")
    return lowerer.emit(
        OperationKind.ATOMIC_LOAD,
        lowerer.location(node),
        operands=(target, *lowered.operands),
        result_types=(lowerer.value_result_type(target.type.dtype, lowered.result_shape),),
        attributes={"index": lowered.relation, "ordering": order},
        effects=(_atomic_effect(target),),
    ).results[0]


def _atomic_store(lowerer: FunctionLowerer, node: ast.Call) -> StaticTuple:
    bound = bind_call(
        lowerer,
        node,
        ("target", "index", "value", "order"),
        required=("target", "index", "value", "order"),
    )
    target, lowered, value = _atomic_value_inputs(lowerer, bound, node)
    order = _ordering(lowerer, bound["order"])
    if order not in (AtomicOrdering.RELAXED, AtomicOrdering.RELEASE):
        lowerer.error(bound["order"], "atomic store allows relaxed or release order")
    lowerer.emit(
        OperationKind.ATOMIC_STORE,
        lowerer.location(node),
        operands=(target, *lowered.operands, value),
        attributes={
            "index": lowered.relation,
            "ordering": order,
        },
        effects=(_atomic_effect(target),),
    )
    return StaticTuple(())


def _atomic_rmw(
    lowerer: FunctionLowerer,
    node: ast.Call,
    kind: AtomicRMWKind,
) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("target", "index", "value", "order"),
        required=("target", "index", "value", "order"),
    )
    target, lowered, value = _atomic_value_inputs(lowerer, bound, node)
    return lowerer.emit(
        OperationKind.ATOMIC_RMW,
        lowerer.location(node),
        operands=(target, *lowered.operands, value),
        result_types=(value.type,),
        attributes={
            "index": lowered.relation,
            "ordering": _ordering(lowerer, bound["order"]),
            "kind": kind,
        },
        effects=(_atomic_effect(target),),
    ).results[0]


def _atomic_compare_exchange(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("target", "index", "expected", "desired", "order"),
        required=("target", "index", "expected", "desired", "order"),
    )
    target, lowered = _atomic_address(lowerer, bound, node)
    expected_expression = lowerer.lower_expression(bound["expected"])
    desired_expression = lowerer.lower_expression(bound["desired"])
    expected = lowerer.materialize(
        expected_expression,
        bound["expected"],
        ScalarType(target.type.dtype) if isinstance(expected_expression, Literal) else None,
    )
    desired = lowerer.materialize(
        desired_expression,
        bound["desired"],
        ScalarType(target.type.dtype) if isinstance(desired_expression, Literal) else None,
    )
    expected = validate_indexed_value(
        lowerer, expected, lowered.result_shape, node, expected_dtype=target.type.dtype
    )
    desired = validate_indexed_value(
        lowerer, desired, lowered.result_shape, node, expected_dtype=target.type.dtype
    )
    result_type = RecordType(
        (
            ("old_value", desired.type),
            ("success", lowerer.value_result_type(intent_bool, lowered.result_shape)),
        )
    )
    return lowerer.emit(
        OperationKind.ATOMIC_COMPARE_EXCHANGE,
        lowerer.location(node),
        operands=(target, *lowered.operands, expected, desired),
        result_types=(result_type,),
        attributes={
            "index": lowered.relation,
            "ordering": _ordering(lowerer, bound["order"]),
        },
        effects=(_atomic_effect(target),),
    ).results[0]


def _random_bits(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("seed", "logical_counter"),
        required=("seed", "logical_counter"),
    )
    seed = lowerer.materialize(lowerer.lower_expression(bound["seed"]), bound["seed"])
    counter = lowerer.materialize(
        lowerer.lower_expression(bound["logical_counter"]), bound["logical_counter"]
    )
    if seed.type != ScalarType(u64):
        lowerer.error(bound["seed"], "I.random.bits seed must be u64")
    if isinstance(counter.type, ScalarType):
        if counter.type.dtype not in (u64, intent_index):
            lowerer.error(bound["logical_counter"], "random counter must be u64/index")
        shape = ()
    elif isinstance(counter.type, TensorType):
        if counter.type.dtype not in (u64, intent_index):
            lowerer.error(bound["logical_counter"], "random counter must be u64/index")
        shape = counter.type.shape
    else:
        lowerer.error(bound["logical_counter"], "random counter must be u64/index")
    return lowerer.emit(
        OperationKind.RANDOM_BITS,
        lowerer.location(node),
        operands=(seed, counter),
        result_types=(lowerer.value_result_type(u32, shape),),
    ).results[0]


def _random_uniform(lowerer: FunctionLowerer, node: ast.Call) -> MlirValue:
    bound = bind_call(
        lowerer,
        node,
        ("seed", "logical_counter", "dtype"),
        required=("seed", "logical_counter", "dtype"),
    )
    dtype = require_dtype(lowerer, bound["dtype"])
    if dtype != f32:
        lowerer.error(bound["dtype"], "I.random.uniform canonical result dtype is f32")
    bits_call = ast.Call(
        func=node.func,
        args=[bound["seed"], bound["logical_counter"]],
        keywords=[],
    )
    ast.copy_location(bits_call, node)
    bits = _random_bits(lowerer, bits_call)
    shift = lowerer.emit_literal(8, node, ScalarType(u32))
    _, bits_shape = lowerer.dtype_and_shape(bits.type, node)
    shift = lowerer.broadcast_value(shift, bits_shape, node)
    shifted = lowerer.emit(
        OperationKind.BINARY,
        lowerer.location(node),
        operands=(bits, shift),
        result_types=(bits.type,),
        attributes={"operator_kind": BinaryOperator.RIGHT_SHIFT},
    ).results[0]
    converted = lowerer.emit(
        OperationKind.CAST,
        lowerer.location(node),
        operands=(shifted,),
        result_types=(lowerer.value_result_type(f32, getattr(bits.type, "shape", ())),),
    ).results[0]
    scale = lowerer.emit_literal(2.0 ** -24, node, ScalarType(f32))
    scale = lowerer.broadcast_value(scale, bits_shape, node)
    return lowerer.emit(
        OperationKind.BINARY,
        lowerer.location(node),
        operands=(converted, scale),
        result_types=(converted.type,),
        attributes={"operator_kind": BinaryOperator.MULTIPLY},
    ).results[0]


def _atomic_address(lowerer: FunctionLowerer, bound: dict[str, ast.AST], node: ast.Call):
    target = lowerer.materialize(lowerer.lower_expression(bound["target"]), bound["target"])
    if not isinstance(target.type, (TensorType, BufferType)):
        lowerer.error(node, "atomic target must be an external view or logical buffer")
    if isinstance(target.type, TensorType):
        lowerer.require_atomic_view(target, node)
    lowered = lower_index(lowerer, target, bound["index"])
    return target, lowered


def _atomic_value_inputs(lowerer: FunctionLowerer, bound: dict[str, ast.AST], node: ast.Call):
    target, lowered = _atomic_address(lowerer, bound, node)
    expression = lowerer.lower_expression(bound["value"])
    value = lowerer.materialize(
        expression,
        bound["value"],
        ScalarType(target.type.dtype) if isinstance(expression, Literal) else None,
    )
    value = validate_indexed_value(
        lowerer, value, lowered.result_shape, node, expected_dtype=target.type.dtype
    )
    return target, lowered, value


def _atomic_effect(target: MlirValue) -> Effect:
    resource = (
        ResourceKind.LOGICAL_BUFFER
        if isinstance(target.type, BufferType)
        else ResourceKind.EXTERNAL_VIEW
    )
    return Effect(EffectKind.ATOMIC, resource, target)


def _ordering(lowerer: FunctionLowerer, node: ast.AST) -> AtomicOrdering:
    expression = lowerer.lower_expression(node)
    value = expression.value if isinstance(expression, Literal) else expression
    if isinstance(value, AtomicOrdering):
        return value
    if isinstance(value, str):
        mapping = {
            "relaxed": AtomicOrdering.RELAXED,
            "acquire": AtomicOrdering.ACQUIRE,
            "release": AtomicOrdering.RELEASE,
            "acq_rel": AtomicOrdering.ACQ_REL,
        }
        if value in mapping:
            return mapping[value]
    lowerer.error(node, "invalid canonical atomic ordering")
