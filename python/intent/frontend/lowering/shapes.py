from __future__ import annotations

from intent.frontend.mlir import BlockState, EmittedOperation, MlirValue
from intent.frontend.semantics import (
    BinaryOperator, BufferType, DomainType, DynamicDim, LogicalIndexType,
    OperationKind, RegionType, ScalarType, StaticDim, TensorType,
)
from intent.language import DTypeCategory, index as intent_index

from .ast.model import ShapeDimension


def visible_values(block: BlockState):
    blocks = []
    while block is not None:
        blocks.append(block)
        block = block.visible_parent
    for scope in reversed(blocks):
        yield from scope.arguments
        for operation in scope.operations:
            yield from operation.results


def iteration_bounds(value: MlirValue) -> tuple[MlirValue, MlirValue] | None:
    operation = value.owner
    if not isinstance(operation, EmittedOperation):
        return None
    if operation.kind is OperationKind.DOMAIN:
        return operation.operands[:2]
    if operation.kind is OperationKind.SUBREGION:
        source, *boundaries = operation.operands
        inherited = iteration_bounds(source)
        has_start = operation.attributes["has_start"]
        has_stop = operation.attributes["has_stop"]
        start = boundaries[0] if has_start else inherited[0] if inherited else None
        stop = boundaries[-1] if has_stop else inherited[1] if inherited else None
        if start is not None and stop is not None:
            return start, stop
    return None


class ShapeAnalysis:
    """Shape identities interned from the append-only typed construction graph.

    Defining operations, lexical ownership, domain bounds and integer terms are
    read from that graph. The two tables intern exact integer operations and
    dimension identities; neither is an alternative operation or scope graph.
    """

    def __init__(self, builder) -> None:
        self.builder = builder
        self._integer_operations: dict[tuple, EmittedOperation] = {}
        self._dimensions: dict[tuple, object] = {}
        self._dynamic_dimension_counter = 0

    @staticmethod
    def integer_key(block, kind, operands, result_types, attributes, regions, effects):
        if (kind not in (OperationKind.CONSTANT, OperationKind.DIM, OperationKind.BINARY)
                or len(result_types) != 1 or not isinstance(result_types[0], ScalarType)
                or result_types[0].dtype.category not in (
                    DTypeCategory.INDEX, DTypeCategory.SIGNED_INTEGER, DTypeCategory.UNSIGNED_INTEGER)
                or effects or regions or block.terminated):
            return None
        return block, kind, operands, result_types, tuple(sorted((attributes or {}).items()))

    def existing_operation(self, key):
        return self._integer_operations.get(key) if key is not None else None

    def record_operation(self, key, operation: EmittedOperation) -> None:
        if key is not None:
            self._integer_operations[key] = operation

    def fresh_dimension(self, role: str) -> DynamicDim:
        identity = self._dynamic_dimension_counter
        self._dynamic_dimension_counter += 1
        return DynamicDim(f"{role}_{identity}")

    def shape(self, value: MlirValue) -> tuple[object, ...]:
        if isinstance(value.type, (TensorType, BufferType)):
            return value.type.shape
        if isinstance(value.type, (DomainType, RegionType)):
            operation = value.owner
            if isinstance(operation, EmittedOperation):
                dimensions = operation.attributes.get("extent_dimensions")
                if dimensions is not None:
                    return tuple(self.builder.dimension(identity) for identity in dimensions)
            return tuple(DynamicDim(f"region_{value.id}_{axis}") for axis in range(value.type.rank))
        return ()

    def remember_dimension(self, value: MlirValue, dimension: object) -> None:
        self._dimensions[value.type, self.integer_terms(value)] = dimension

    def shape_dimension(self, value: MlirValue, preferred: object = None) -> object:
        operation = value.owner
        if isinstance(operation, EmittedOperation) and operation.kind is OperationKind.DIM:
            return self.builder.dimension(operation.attributes["dimension"])
        terms = self.integer_terms(value)
        key = value.type, terms
        dimension = self._dimensions.get(key)
        if dimension is None and not terms[0] and isinstance(value.type, ScalarType):
            dtype = value.type.dtype
            modulus = 1 << dtype.bits
            extent = terms[1] % modulus
            if dtype.category in (DTypeCategory.SIGNED_INTEGER, DTypeCategory.INDEX) and extent >= modulus // 2:
                extent -= modulus
            if extent >= 0:
                dimension = StaticDim(extent)
        if dimension is None:
            dimension = preferred if preferred is not None else DynamicDim(f"value_{value.id}")
        self._dimensions[key] = dimension
        return dimension

    def dimension_source(self, dimension: object, block: BlockState) -> ShapeDimension | MlirValue | None:
        values = tuple(visible_values(block))
        for value in values:
            for axis, known in enumerate(self.shape(value)):
                if known == dimension:
                    return ShapeDimension(dimension, value, axis)
        for value in values:
            if self._dimensions.get((value.type, self.integer_terms(value))) == dimension:
                return value
        return None

    def integer_terms(self, value: MlirValue) -> tuple[frozenset, int]:
        unknown = frozenset({(("value", value), 1)}), 0
        operation = value.owner
        if (not isinstance(operation, EmittedOperation)
                or not isinstance(value.type, ScalarType)
                or value.type.dtype.category not in (
                    DTypeCategory.INDEX, DTypeCategory.SIGNED_INTEGER, DTypeCategory.UNSIGNED_INTEGER)
                or operation.effects or operation.regions):
            return unknown
        if operation.kind is OperationKind.CONSTANT:
            return frozenset(), operation.attributes["value"]
        if operation.kind is OperationKind.DIM:
            return frozenset({(("dimension", operation.attributes["dimension"]), 1)}), 0
        if operation.kind is not OperationKind.BINARY:
            return unknown
        operator = operation.attributes["operator_kind"]
        operands = operation.operands
        if operator in (BinaryOperator.ADD, BinaryOperator.SUBTRACT):
            terms, constant = {}, 0
            for index, operand in enumerate(operands):
                sign = -1 if index == 1 and operator is BinaryOperator.SUBTRACT else 1
                nested, amount = (self.integer_terms(operand) if operand.type == value.type
                                  else (frozenset({(("value", operand), 1)}), 0))
                constant += sign * amount
                for atom, coefficient in nested:
                    terms[atom] = terms.get(atom, 0) + sign * coefficient
            return frozenset((atom, coefficient) for atom, coefficient in terms.items() if coefficient), constant
        if operator is not BinaryOperator.MULTIPLY:
            return unknown
        for factor, operand in (operands, tuple(reversed(operands))):
            factors, scale = self.integer_terms(factor)
            if factor.type != value.type or factors:
                continue
            bounds = self.integer_bounds(operand)
            if bounds is None:
                continue
            lower, upper = sorted(bound * scale for bound in bounds)
            minimum, maximum = self._limits(value.type.dtype)
            if not minimum <= lower <= upper <= maximum:
                continue
            nested, constant = (self.integer_terms(operand) if operand.type == value.type
                                else (frozenset({(("value", operand), 1)}), 0))
            return frozenset((atom, coefficient * scale) for atom, coefficient in nested
                             if coefficient * scale), constant * scale
        if any(operand.type != value.type for operand in operands):
            return unknown
        factors, coefficient = {}, 1
        for operand in operands:
            atoms, constant = self.integer_terms(operand)
            if (atoms, constant) == (frozenset({(("value", operand), 1)}), 0):
                return unknown
            if not atoms:
                coefficient *= constant
                continue
            if constant or len(atoms) != 1:
                return unknown
            (atom, scale), = atoms
            coefficient *= scale
            product = atom[1] if atom[0] == "product" else ((atom, 1),)
            for factor, power in product:
                factors[factor] = factors.get(factor, 0) + power
        if not factors or coefficient == 0:
            return frozenset(), coefficient
        atom = ("product", frozenset(factors.items()))
        if len(factors) == 1:
            factor, power = next(iter(factors.items()))
            if power == 1:
                atom = factor
        return frozenset({(atom, coefficient)}), 0

    @staticmethod
    def _limits(dtype) -> tuple[int, int]:
        signed = dtype.category in (DTypeCategory.SIGNED_INTEGER, DTypeCategory.INDEX)
        return (-(1 << (dtype.bits - 1)) if signed else 0,
                (1 << (dtype.bits - int(signed))) - 1)

    def integer_bounds(self, value: MlirValue) -> tuple[int, int] | None:
        if not isinstance(value.type, (ScalarType, LogicalIndexType)):
            return None
        dtype = value.type.dtype if isinstance(value.type, ScalarType) else intent_index
        terms, constant = self.integer_terms(value)
        lower = upper = constant
        for (kind, atom), coefficient in terms:
            if kind != "value" or not isinstance(atom.type, LogicalIndexType):
                return None
            domain = next((source for source in visible_values(atom.block)
                           if isinstance(source.type, DomainType)
                           and source.type.origin_id == atom.type.source_id), None)
            bounds = iteration_bounds(domain) if domain is not None else None
            if bounds is None:
                return None
            start_bounds, stop_bounds = (self.integer_bounds(bound) for bound in bounds)
            if start_bounds is None or stop_bounds is None:
                return None
            begin, end = start_bounds[0], stop_bounds[1] - 1
            if begin > end:
                return None
            lower += coefficient * (begin if coefficient > 0 else end)
            upper += coefficient * (end if coefficient > 0 else begin)
        minimum, maximum = self._limits(dtype)
        return (lower, upper) if minimum <= lower <= upper <= maximum else None
