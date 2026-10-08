from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from intent.api import Definition
from intent.frontend.mlir import BlockState
from intent.frontend.mlir.attributes import DenseI32Array
from intent.frontend.semantics import ConstexprType
from intent.frontend.semantics import DynamicDim
from intent.frontend.semantics import Effect
from intent.frontend.mlir import FunctionState
from intent.frontend.semantics import ValueType
from intent.frontend.diagnostics import Location
from intent.frontend.semantics import LogicalIndexType
from intent.frontend.semantics import IndexRelation
from intent.frontend.semantics import IndexTerm
from intent.frontend.semantics import IndexTermKind
from intent.frontend.semantics import OperationKind
from intent.frontend.mlir import EmittedOperation
from intent.frontend.mlir import RegionState
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import StaticDim
from intent.frontend.semantics import TensorType
from intent.frontend.semantics import TupleType
from intent.frontend.mlir import MlirValue
from intent.frontend.semantics import broadcast_shape
from intent.frontend.semantics import dims_compatible
from intent.language import ViewKind
from intent.language import DType
from intent.language import DTypeCategory
from intent.language import bool as intent_bool
from intent.language import f64
from intent.language import i64
from intent.language import index as intent_index
from intent.frontend.semantics import EffectKind
from intent.frontend.semantics import ResourceKind
from intent.frontend.semantics import RegionType

from ...diagnostics.errors import FrontendError
from .model import ConstexprBinding
from .model import Expression
from .model import Literal
from .model import LoopContext
from .model import ShapeDimension
from .model import ShapeValue
from .model import StaticTuple
from ...source.unit import SourceUnit
from ..scope import lowering_scope
from ..shape_construction import ShapeBuilder
from ..products import build_product, project_product, same_product_type
from ..shapes import ShapeAnalysis

if TYPE_CHECKING:
    from ...compilation.compiler import FrontendCompiler


@dataclass(slots=True)
class InlineHelperFrame:
    entry_block: BlockState
    returned: tuple[MlirValue, ...] | None = None


class FunctionLowerer:
    def __init__(
        self,
        *,
        compiler: FrontendCompiler,
        definition: Definition[object, object],
        source: SourceUnit,
        function: FunctionState,
        constexpr_values: dict[str, object],
    ) -> None:
        self.compiler = compiler
        self.definition = definition
        self.source = source
        self.function = function
        self.environment: dict[str, Expression] = {}
        self.call_arguments: dict[ast.AST, Expression] = {}
        self.current_block = function.body.blocks[0]
        self.loop_stack: list[LoopContext] = []
        self.inline_helpers: list[InlineHelperFrame] = []
        self.ragged_mappings: dict[MlirValue, MlirValue] = {}
        self.shapes = ShapeAnalysis(compiler.builder)
        self._initialize_parameters(constexpr_values)

    def _initialize_parameters(self, constexpr_values: dict[str, object]) -> None:
        for parameter in self.function.parameters:
            name = parameter.spec.name
            value = parameter.value
            if isinstance(parameter.spec.type, ConstexprType):
                self.environment[name] = ConstexprBinding(constexpr_values[name], value)
            else:
                self.environment[name] = value

    def lower(self) -> None:
        self.lower_statements(self.source.function.body)
        if not self.is_terminated(self.current_block):
            self.emit(OperationKind.RETURN, self.source.location(self.source.function))

    def lower_inline_helper(
        self,
        *,
        definition: Definition[object, object],
        source: SourceUnit,
        parameters: tuple[object, ...],
        arguments: tuple[object, ...],
    ) -> tuple[MlirValue, ...]:
        if len(parameters) != len(arguments):
            self.error(source.function, "helper argument count does not match call")
        frame = InlineHelperFrame(self.current_block)
        with lowering_scope(
            self, definition=definition, source=source,
            environment={parameter.name: argument for parameter, argument in zip(parameters, arguments)},
            loop_stack=[], inline_helpers=[*self.inline_helpers, frame],
            current_block=self.current_block,
        ):
            from .statements import normalize_helper_returns

            body = normalize_helper_returns(self, source.function)
            if body is source.function.body:
                self.lower_statements(body)
            else:
                self.lower_statements(body[:-1])
                if body[-1].value.id not in self.environment:
                    self.error(source.function, "@intent.fn must return a value on every runtime branch")
                self.lower_statements(body[-1:])
            return frame.returned if frame.returned is not None else ()

    def lower_statements(self, statements: list[ast.stmt]) -> None:
        from .statements import lower_statement

        for statement in statements:
            if self.inline_helpers and self.inline_helpers[-1].returned is not None:
                self.error(statement, "statement is unreachable after helper return")
            if self.is_terminated(self.current_block):
                self.error(statement, "statement is unreachable after a terminator")
            lower_statement(self, statement)

    def lower_expression(self, node: ast.AST) -> Expression:
        from .expressions import lower_expression

        if node in self.call_arguments:
            return self.call_arguments[node]
        return lower_expression(self, node)

    def emit(
        self,
        opcode: OperationKind,
        location: Location,
        *,
        operands: tuple[MlirValue, ...] = (),
        result_types: tuple[ValueType, ...] = (),
        operand_groups: tuple[tuple[MlirValue, ...], ...] | None = None,
        result_type_groups: tuple[tuple[ValueType, ...], ...] | None = None,
        attributes: dict[str, object] | None = None,
        regions: tuple[RegionState, ...] = (),
        effects: tuple[Effect, ...] = (),
        result_names: tuple[str | None, ...] = (),
    ) -> EmittedOperation:
        if operand_groups is not None:
            if operands:
                raise ValueError("provide operand groups or flat operands, not both")
            operands = tuple(value for group in operand_groups for value in group)
            attributes = {
                **(attributes or {}),
                "operandSegmentSizes": DenseI32Array(tuple(len(group) for group in operand_groups)),
            }
        if result_type_groups is not None:
            if result_types:
                raise ValueError("provide result groups or flat result types, not both")
            result_types = tuple(value_type for group in result_type_groups for value_type in group)
            attributes = {
                **(attributes or {}),
                "resultSegmentSizes": DenseI32Array(tuple(len(group) for group in result_type_groups)),
            }
        key = self.shapes.integer_key(
            self.current_block, opcode, operands, result_types, attributes, regions, effects,
        )
        existing = self.shapes.existing_operation(key)
        if existing is not None:
            return existing
        operation = self.compiler.builder.emit(
            self.current_block,
            opcode,
            location,
            operands=operands,
            result_types=result_types,
            attributes=attributes,
            regions=regions,
            effects=effects,
            result_names=result_names,
        )
        self.shapes.record_operation(key, operation)
        return operation

    def make_region(
        self,
        location: Location,
        argument_types: tuple[ValueType, ...] = (),
        argument_names: tuple[str | None, ...] = (),
        *,
        isolated_from_above: bool = False,
    ) -> RegionState:
        region = self.compiler.builder.region(
            location, argument_types, argument_names, parent_block=self.current_block,
            isolated_from_above=isolated_from_above,
        )
        return region

    def materialize(
        self,
        expression: Expression,
        node: ast.AST,
        expected_type: ValueType | None = None,
    ) -> MlirValue:
        if isinstance(expression, MlirValue):
            if expected_type is not None and not self.types_compatible_for_literal(
                expression.type, expected_type
            ):
                self.error(node, f"value type {expression.type} does not match {expected_type}")
            return expression
        if isinstance(expression, ConstexprBinding):
            constexpr_type = expression.ir_value.type
            return self.emit_literal(
                expression.python_value,
                node,
                expected_type
                or (
                    constexpr_type.value_type
                    if isinstance(constexpr_type, ConstexprType)
                    else constexpr_type
                ),
            )
        if isinstance(expression, ShapeDimension):
            return self.materialize_dimension(expression, node)
        if isinstance(expression, Literal):
            return self.emit_literal(expression.value, node, expected_type)
        if isinstance(expression, StaticTuple):
            expected_components: tuple[ValueType, ...] | None = None
            if expected_type is not None:
                if not isinstance(expected_type, TupleType):
                    self.error(node, f"tuple value does not match {expected_type}")
                if len(expression.elements) != len(expected_type.components):
                    self.error(node, "tuple value/expected type arity mismatch")
                expected_components = expected_type.components
            components = tuple(
                self.read_value(
                    self.materialize(
                        element, node,
                        expected_components[index] if expected_components is not None else None,
                    ),
                    node,
                )
                for index, element in enumerate(expression.elements)
            )
            result_type = TupleType(tuple(component.type for component in components))
            return build_product(self, components, result_type, node)
        self.error(node, "expression is compile-time metadata, not an SSA value")

    def project_value_schema(
        self, value: MlirValue, expected_type: ValueType, node: ast.AST,
    ) -> MlirValue:
        if self._same_emitted_type(value.type, expected_type):
            return value
        if not self.types_compatible_for_literal(value.type, expected_type):
            self.error(node, f"value type {value.type} does not match {expected_type}")
        return project_product(
            self, value, expected_type, node,
            lambda component, target: self.broadcast_value(component, target.shape, node)
            if isinstance(target, TensorType) else component,
        )

    def _same_emitted_type(self, actual: ValueType, expected: ValueType) -> bool:
        def same_leaf(left, right):
            if isinstance(left, TensorType) and isinstance(right, TensorType):
                return (left.dtype == right.dtype and len(left.shape) == len(right.shape)
                        and all(self.compiler.builder.dimension_id(a) == self.compiler.builder.dimension_id(b)
                                for a, b in zip(left.shape, right.shape)))
            return left == right
        return same_product_type(actual, expected, same_leaf)

    def emit_literal(
        self,
        value: bool | int | float,
        node: ast.AST,
        expected_type: ValueType | None = None,
    ) -> MlirValue:
        if expected_type is None:
            if isinstance(value, bool):
                expected_type = ScalarType(intent_bool)
            elif isinstance(value, int):
                expected_type = ScalarType(i64)
            else:
                expected_type = ScalarType(f64)
        if isinstance(expected_type, ConstexprType):
            expected_type = expected_type.value_type
        if not isinstance(expected_type, ScalarType):
            self.error(node, "literal context must require a scalar type")
        category = expected_type.dtype.category
        if category is DTypeCategory.BOOL and not isinstance(value, bool):
            self.error(node, "bool literal context requires a Python bool")
        if category in (
            DTypeCategory.SIGNED_INTEGER,
            DTypeCategory.UNSIGNED_INTEGER,
            DTypeCategory.INDEX,
        ) and (isinstance(value, bool) or not isinstance(value, int)):
            self.error(node, "integer literal context requires a Python int")
        if category in (DTypeCategory.FLOAT, DTypeCategory.BFLOAT) and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            self.error(node, "floating literal context requires int/float")
        if category in (DTypeCategory.FLOAT, DTypeCategory.BFLOAT) and isinstance(
            value, int
        ):
            value = float(value)
        operation = self.emit(
            OperationKind.CONSTANT,
            self.location(node),
            result_types=(expected_type,),
            attributes={"value": value},
        )
        return operation.results[0]

    def materialize_dimension(self, dimension: ShapeDimension, node: ast.AST) -> MlirValue:
        if isinstance(dimension.dimension, StaticDim):
            return self.emit_literal(
                dimension.dimension.value, node, ScalarType(intent_index)
            )
        operation = self.emit(
            OperationKind.DIM,
            self.location(node),
            operands=(dimension.source,),
            result_types=(ScalarType(intent_index),),
            attributes={
                "axis": dimension.axis,
                "dimension": self.compiler.builder.dimension_id(dimension.dimension),
            },
        )
        result = operation.results[0]
        self.shapes.remember_dimension(result, dimension.dimension)
        return result

    def materialize_shape_extent(self, dimension: object, node: ast.AST) -> MlirValue:
        source = self.shapes.dimension_source(dimension, self.current_block)
        if isinstance(source, ShapeDimension):
            return self.materialize_dimension(source, node)
        if isinstance(source, MlirValue):
            if source.type == ScalarType(intent_index):
                return source
            return self.emit(
                OperationKind.CAST, self.location(node), operands=(source,),
                result_types=(ScalarType(intent_index),),
            ).results[0]
        self.error(node, f"dynamic shape extent {dimension} has no SSA source")

    def integer_shape_dimension(self, value: MlirValue, preferred: object = None) -> object:
        return self.shapes.shape_dimension(value, preferred)

    def integer_expression_bounds(self, value: MlirValue) -> tuple[int, int] | None:
        return self.shapes.integer_bounds(value)

    def read_value(self, expression: Expression, node: ast.AST) -> MlirValue:
        value = self.materialize(expression, node)
        if value.view_kind is None:
            return value
        self.require_readable_view(value, node)
        if not isinstance(value.type, TensorType):
            self.error(node, "view parameter must have tensor type")
        relation = IndexRelation(
            len(value.type.shape),
            len(value.type.shape),
            tuple(
                self.compiler.builder.dimension_id(dimension)
                for dimension in value.type.shape
            ),
            tuple(IndexTerm(IndexTermKind.FULL_SLICE) for _ in value.type.shape)
        )
        operation = self.emit(
            OperationKind.VIEW_LOAD,
            self.location(node),
            operand_groups=((value,), (), (), ()),
            result_types=(value.type,),
            attributes={"index": relation},
            effects=(Effect(EffectKind.READ, ResourceKind.EXTERNAL_VIEW, value),),
        )
        return operation.results[0]

    def require_readable_view(self, value: MlirValue, node: ast.AST) -> None:
        kind = value.view_kind
        if kind not in (ViewKind.IN, ViewKind.INOUT, ViewKind.OUT):
            self.error(node, "view read requires an external-view ABI")

    def require_writable_view(self, value: MlirValue, node: ast.AST) -> None:
        kind = value.view_kind
        if kind not in (ViewKind.OUT, ViewKind.INOUT):
            self.error(node, "view write requires Out or InOut ABI")

    def require_atomic_view(self, value: MlirValue, node: ast.AST) -> None:
        if value.view_kind is not ViewKind.INOUT:
            self.error(node, "external atomic target requires InOut ABI")

    def shape_value(self, value: MlirValue, node: ast.AST) -> ShapeValue:
        if not isinstance(value.type, TensorType):
            self.error(node, ".shape is only valid on a tensor/view")
        return ShapeValue(
            tuple(
                ShapeDimension(dimension, value, axis)
                for axis, dimension in enumerate(value.type.shape)
            )
        )

    def value_result_type(
        self,
        dtype: DType,
        shape: tuple[object, ...],
        *,
        ranked: bool = False,
    ) -> ValueType:
        return TensorType(dtype, shape) if shape or ranked else ScalarType(dtype)

    def broadcast_result_type(self, lhs: ValueType, rhs: ValueType, node: ast.AST) -> ValueType:
        lhs_dtype, lhs_shape = self.dtype_and_shape(lhs, node)
        rhs_dtype, rhs_shape = self.dtype_and_shape(rhs, node)
        if lhs_dtype != rhs_dtype:
            self.error(node, "mixed dtypes require an explicit I.cast")
        try:
            shape = broadcast_shape(lhs_shape, rhs_shape)
        except ValueError as error:
            self.error(node, str(error))
        return self.value_result_type(
            lhs_dtype, shape, ranked=isinstance(lhs, TensorType) or isinstance(rhs, TensorType)
        )

    def broadcast_value(
        self,
        value: MlirValue,
        result_shape: tuple[object, ...],
        node: ast.AST,
        *,
        ranked: bool = False,
    ) -> MlirValue:
        dtype, source_shape = self.dtype_and_shape(value.type, node)
        target_shape = tuple(result_shape)
        if (not ranked or isinstance(value.type, TensorType)) and len(source_shape) == len(target_shape) and all(
            self.compiler.builder.dimension_id(source)
            == self.compiler.builder.dimension_id(target)
            for source, target in zip(source_shape, target_shape)
        ):
            return value
        if not target_shape and not ranked:
            self.error(node, "tensor value cannot broadcast to a scalar")
        try:
            broadcasted = tuple(broadcast_shape(source_shape, target_shape))
        except ValueError as error:
            self.error(node, str(error))
        if len(broadcasted) != len(target_shape) or not all(
            dims_compatible(source, destination)
            for source, destination in zip(broadcasted, target_shape)
        ):
            self.error(node, "value cannot broadcast to the required result shape")
        shape = ShapeBuilder(self.compiler.builder.dimension_id, first_operand_position=1)
        for dimension in target_shape:
            shape.append_extent(
                dimension,
                None if isinstance(dimension, StaticDim) else self.materialize_shape_extent(dimension, node),
            )
        lowered = shape.finish()
        operation = self.emit(
            OperationKind.BROADCAST,
            self.location(node),
            operands=(value, *lowered.operands),
            result_types=(TensorType(dtype, lowered.dimensions),),
            attributes={"shape": lowered.relation},
        )
        return operation.results[0]

    def dtype_and_shape(
        self,
        value_type: ValueType,
        node: ast.AST,
    ) -> tuple[DType, tuple[object, ...]]:
        if isinstance(value_type, ScalarType):
            return value_type.dtype, ()
        if isinstance(value_type, LogicalIndexType):
            return intent_index, ()
        if isinstance(value_type, TensorType):
            return value_type.dtype, tuple(value_type.shape)
        self.error(node, f"expected scalar/tensor value, got {value_type}")

    def coerce_pair(
        self,
        lhs: Expression,
        rhs: Expression,
        node: ast.AST,
    ) -> tuple[MlirValue, MlirValue]:
        from .expressions import compile_time_value

        lhs_known, lhs_static = compile_time_value(lhs)
        rhs_known, rhs_static = compile_time_value(rhs)
        if lhs_known and not rhs_known:
            lhs = Literal(lhs_static)
        if rhs_known and not lhs_known:
            rhs = Literal(rhs_static)
        if isinstance(lhs, Literal) and not isinstance(rhs, Literal):
            rhs_value = self.read_value(rhs, node)
            dtype, _ = self.dtype_and_shape(rhs_value.type, node)
            return self.materialize(lhs, node, ScalarType(dtype)), rhs_value
        if isinstance(rhs, Literal) and not isinstance(lhs, Literal):
            lhs_value = self.read_value(lhs, node)
            dtype, _ = self.dtype_and_shape(lhs_value.type, node)
            return lhs_value, self.materialize(rhs, node, ScalarType(dtype))
        return self.read_value(lhs, node), self.read_value(rhs, node)

    def types_compatible_for_literal(self, actual: ValueType, expected: ValueType) -> bool:
        from intent.frontend.semantics.types import types_compatible

        return types_compatible(actual, expected)

    def is_terminated(self, block: BlockState) -> bool:
        from intent.frontend.semantics import TERMINATORS

        return block.last_operation in TERMINATORS

    def location(self, node: ast.AST) -> Location:
        return self.source.location(node)

    def error(self, node: ast.AST, message: str) -> None:
        raise FrontendError(message, self.location(node))

    def dynamic_shape_for_region(self, value: MlirValue) -> tuple[object, ...]:
        return self.shapes.shape(value)

    def fresh_dynamic_dimension(self, role: str) -> DynamicDim:
        return self.shapes.fresh_dimension(role)

    def logical_index_type(self, source: MlirValue, axis: int) -> LogicalIndexType:
        source_id = (
            source.type.source_id
            if isinstance(source.type, RegionType)
            else getattr(source.type, "origin_id", None)
        )
        if source_id is None:
            raise TypeError("logical index source requires stable domain provenance")
        return LogicalIndexType(source_id, axis)
