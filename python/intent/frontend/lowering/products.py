from __future__ import annotations

from dataclasses import dataclass

from intent.frontend.mlir import MlirValue
from intent.frontend.semantics import OperationKind, RecordType, TupleType, ValueType


@dataclass(frozen=True, slots=True)
class ProductSchema:
    kind: str
    names: tuple[str, ...]

    @classmethod
    def of(cls, value_type: ValueType) -> ProductSchema:
        if isinstance(value_type, TupleType):
            return cls("tuple", tuple(f"#{index}" for index in range(len(value_type.components))))
        if isinstance(value_type, RecordType):
            return cls("record", tuple(name for name, _ in value_type.fields))
        return cls("scalar", ())

    def type(self, components: tuple[ValueType, ...]) -> ValueType:
        if self.kind == "scalar":
            if len(components) != 1:
                raise ValueError("a scalar schema has exactly one component")
            return components[0]
        if len(components) != len(self.names):
            raise ValueError("product schema and components have different arity")
        return (TupleType(components) if self.kind == "tuple"
                else RecordType(tuple(zip(self.names, components))))


def component_types(value_type: ValueType) -> tuple[ValueType, ...]:
    if isinstance(value_type, TupleType):
        return value_type.components
    if isinstance(value_type, RecordType):
        return tuple(field_type for _, field_type in value_type.fields)
    return (value_type,)


def map_product_type(value_type: ValueType, leaf) -> ValueType:
    schema = ProductSchema.of(value_type)
    if schema.kind == "scalar":
        return leaf(value_type)
    return schema.type(tuple(map_product_type(component, leaf) for component in component_types(value_type)))


def same_product_type(actual: ValueType, expected: ValueType, leaf) -> bool:
    schema = ProductSchema.of(actual)
    if schema != ProductSchema.of(expected):
        return False
    if schema.kind == "scalar":
        return leaf(actual, expected)
    return all(same_product_type(left, right, leaf)
               for left, right in zip(component_types(actual), component_types(expected)))


def extract_product(lowerer, value: MlirValue, index: int, node) -> MlirValue:
    return lowerer.emit(
        OperationKind.EXTRACT, lowerer.location(node), operands=(value,),
        result_types=(component_types(value.type)[index],), attributes={"field": index},
    ).results[0]


def product_components(lowerer, value: MlirValue, node) -> tuple[MlirValue, ...]:
    if ProductSchema.of(value.type).kind == "scalar":
        return (value,)
    return tuple(extract_product(lowerer, value, index, node)
                 for index in range(len(component_types(value.type))))


def build_product(lowerer, values, value_type: ValueType, node) -> MlirValue:
    schema = ProductSchema.of(value_type)
    values = tuple(values)
    if schema.kind == "scalar":
        if len(values) != 1:
            raise ValueError("a scalar schema has exactly one value")
        return values[0]
    if len(values) != len(schema.names):
        raise ValueError("product schema and values have different arity")
    return lowerer.emit(
        OperationKind.MAKE_TUPLE if schema.kind == "tuple" else OperationKind.MAKE_RECORD,
        lowerer.location(node), operands=values, result_types=(value_type,),
    ).results[0]


def project_product(lowerer, value: MlirValue, target: ValueType, node, leaf) -> MlirValue:
    schema = ProductSchema.of(value.type)
    if schema != ProductSchema.of(target):
        lowerer.error(node, "product fields or arity do not match the required schema")
    if schema.kind == "scalar":
        return leaf(value, target)
    values = tuple(project_product(lowerer, component, expected, node, leaf)
                   for component, expected in zip(product_components(lowerer, value, node), component_types(target)))
    return build_product(lowerer, values, target, node)
