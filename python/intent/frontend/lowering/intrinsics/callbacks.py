"""Pure structured callbacks, product identities and explicit captures."""
from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from intent.api import HelperDefinition
from intent.frontend.mlir import MlirValue
from intent.frontend.semantics import BinaryOperator
from intent.frontend.semantics import OperationKind
from intent.frontend.semantics import ScalarType
from intent.frontend.semantics import TensorType
from intent.frontend.semantics import ValueType
from intent.language.builtins import Intrinsic
from ..regions import pure_region
from ..products import ProductSchema, build_product, product_components
from ..ast.model import ConstexprBinding
from ..ast.model import Expression
from ..ast.model import Literal
from ..ast.model import StaticTuple

if TYPE_CHECKING:
    from ..ast.context import FunctionLowerer

def helper_region(
    lowerer: FunctionLowerer,
    helper_node: ast.AST,
    argument_types: tuple[ValueType, ...],
    static_bindings: tuple[ConstexprBinding | None, ...],
    call_node: ast.AST,
) -> tuple[object, tuple[ValueType, ...]]:
    helper = lowerer.lower_expression(helper_node)
    if not isinstance(helper, HelperDefinition):
        lowerer.error(helper_node, "structured callable must be an @intent.fn")
    if len(static_bindings) != len(argument_types):
        raise ValueError("helper static-binding schema mismatch")
    def invoke(arguments):
        bound = tuple(ConstexprBinding(binding.python_value, value) if binding is not None else value
                      for value, binding in zip(arguments, static_bindings))
        return lowerer.compiler.lower_helper_inline(lowerer, helper, bound, lowerer.location(call_node))
    return pure_region(lowerer, argument_types, call_node, invoke)


def combine_region(
    lowerer: FunctionLowerer,
    helper_node: ast.AST,
    accumulator_types: tuple[ValueType, ...],
    captures: tuple[MlirValue, ...],
    capture_bindings: tuple[ConstexprBinding | None, ...],
    component_names: ProductSchema,
    node: ast.AST,
) -> tuple[object, tuple[ValueType, ...]]:
    expression = lowerer.lower_expression(helper_node)
    if isinstance(expression, Intrinsic):
        if captures:
            lowerer.error(node, "built-in combine does not accept captures")
        operator = {
            "add": BinaryOperator.ADD,
            "maximum": BinaryOperator.MAXIMUM,
            "minimum": BinaryOperator.MINIMUM,
        }.get(expression.name)
        if operator is None:
            lowerer.error(helper_node, "unsupported built-in combine")
        return (
            builtin_combine_region(lowerer, accumulator_types, operator, node),
            accumulator_types,
        )
    if not isinstance(expression, HelperDefinition):
        lowerer.error(helper_node, "combine must be an Intent intrinsic or @intent.fn")
    def combine_body(block_arguments):
        arguments: list[Expression]
        product_kind = component_names.kind
        if product_kind == "scalar":
            arguments = list(block_arguments)
        else:
            count = len(accumulator_types)
            lhs = rebuild_components(lowerer, tuple(block_arguments[:count]), component_names, node)
            rhs = rebuild_components(lowerer, tuple(block_arguments[count:2 * count]), component_names, node)
            arguments = [lhs, rhs, *block_arguments[2 * count :]]
        capture_offset = 2 * len(accumulator_types)
        for index, binding in enumerate(capture_bindings):
            if binding is not None:
                argument_index = (
                    capture_offset + index
                    if product_kind == "scalar"
                    else 2 + index
                )
                arguments[argument_index] = ConstexprBinding(
                    binding.python_value,
                    block_arguments[capture_offset + index],
                )
        helper_results = lowerer.compiler.lower_helper_inline(
            lowerer,
            expression,
            tuple(arguments),
            lowerer.location(node),
        )
        if product_kind == "scalar":
            results = helper_results
        else:
            if len(helper_results) != 1:
                lowerer.error(node, "product combine must return one typed product")
            results, result_names = source_components(lowerer, helper_results[0], node)
            if result_names != component_names:
                lowerer.error(node, "product combine result schema does not match accumulator")
        return results
    return pure_region(
        lowerer, accumulator_types + accumulator_types + tuple(value.type for value in captures),
        node, combine_body, purpose="combine",
    )


def lower_component_identity(
    lowerer: FunctionLowerer,
    identity_node: ast.AST,
    source_components: tuple[MlirValue, ...],
    component_names: ProductSchema,
    call_node: ast.AST,
) -> MlirValue:
    if component_names.kind == "record" and isinstance(identity_node, ast.Call):
        callee = lowerer.lower_expression(identity_node.func)
        if isinstance(callee, Intrinsic) and callee.name == "record":
            if identity_node.args or any(
                keyword.arg is None for keyword in identity_node.keywords
            ):
                lowerer.error(identity_node, "record identity requires named fields")
            names = tuple(keyword.arg for keyword in identity_node.keywords)
            if names != component_names.names:
                lowerer.error(
                    identity_node,
                    "record identity fields must match source record order",
                )
            values: list[MlirValue] = []
            for keyword, source_component in zip(
                identity_node.keywords, source_components
            ):
                expression = lowerer.lower_expression(keyword.value)
                expected = (
                    ScalarType(source_component.type.dtype)
                    if isinstance(expression, Literal)
                    and isinstance(source_component.type, TensorType)
                    else None
                )
                values.append(
                    lowerer.materialize(expression, keyword.value, expected)
                )
            return rebuild_components(lowerer, tuple(values), component_names, call_node)
    expression = lowerer.lower_expression(identity_node)
    expected = (
        ScalarType(source_components[0].type.dtype)
        if len(source_components) == 1
        and isinstance(source_components[0].type, TensorType)
        and isinstance(expression, Literal)
        else None
    )
    return lowerer.materialize(expression, identity_node, expected)


def builtin_combine_region(lowerer: FunctionLowerer, accumulator_types: tuple[ValueType, ...],
                            operator: BinaryOperator, node: ast.AST):
    def combine(arguments):
        count = len(accumulator_types)
        return tuple(lowerer.emit(
            OperationKind.BINARY, lowerer.location(node),
            operands=(arguments[index], arguments[count + index]),
            result_types=(accumulator_types[index],), attributes={"operator_kind": operator},
        ).results[0] for index in range(count))
    return pure_region(lowerer, accumulator_types + accumulator_types, node, combine, purpose="combine")[0]


def source_components(lowerer: FunctionLowerer, source: MlirValue, node: ast.AST):
    return product_components(lowerer, source, node), ProductSchema.of(source.type)


def rebuild_components(lowerer: FunctionLowerer, values: tuple[MlirValue, ...],
                        schema: ProductSchema, node: ast.AST) -> MlirValue:
    return build_product(lowerer, values, schema.type(tuple(value.type for value in values)), node)


def region_captures(
    lowerer: FunctionLowerer,
    node: ast.AST | None,
    call_node: ast.AST,
) -> tuple[tuple[MlirValue, ...], tuple[ConstexprBinding | None, ...]]:
    if node is None:
        return (), ()
    expression = lowerer.lower_expression(node)
    elements = expression.elements if isinstance(expression, StaticTuple) else (expression,)
    values: list[MlirValue] = []
    bindings: list[ConstexprBinding | None] = []
    for element in elements:
        value = lowerer.materialize(element, call_node)
        if isinstance(element, ConstexprBinding):
            binding = element
        elif isinstance(element, Literal):
            binding = ConstexprBinding(element.value, value)
        else:
            binding = None
        values.append(value)
        bindings.append(binding)
    return tuple(values), tuple(bindings)
