from __future__ import annotations

from collections.abc import Iterable

from ..semantics.types import ValueType
from .attributes import FunctionKindAttribute, ParameterAttribute, emit_dictionary
from .state import EmittedOperation, FunctionKind, FunctionState, MlirValue, RegionState
from .types import DimensionID, emit_type, emit_view_type, quote


class AssemblyPrinter:
    """Spell the completed construction graph once, including final source names.

    The registered native Intent dialect owns parsing, verification and KIR
    normalization. This printer does not rewrite operations or infer semantics.
    """

    def __init__(self, dimension_id: DimensionID) -> None:
        self.dimension_id = dimension_id

    def module(self, name: str, functions: Iterable[FunctionState]) -> str:
        attributes = emit_dictionary({"intent.source_module": name})
        lines = [f"module attributes {attributes} {{"]
        for function in functions:
            lines.extend(self.function(function, 1))
        lines.append("}")
        return "\n".join(lines) + "\n"

    def function(self, function: FunctionState, indent: int) -> list[str]:
        prefix = "  " * indent
        if len(function.body.blocks) != 1:
            raise NotImplementedError("Intent functions require one entry block")
        arguments = ", ".join(
            self.argument(parameter.value) + " " + emit_dictionary({
                "intent.parameter": ParameterAttribute(parameter.spec.name, parameter.value.id),
            })
            for parameter in function.parameters
        )
        attributes = {
            "intent.kind": FunctionKindAttribute(0 if function.kind is FunctionKind.KERNEL else 1),
            "intent.source": function.location.format(),
            **{f"intent.{key}": value for key, value in function.attributes.items()},
        }
        results = " -> " + self.result_types(function.result_types) if function.result_types else ""
        lines = [f"{prefix}func.func @{function.name}({arguments}){results} "
                 f"attributes {emit_dictionary(attributes)} {{"]
        for operation in function.body.blocks[0].operations:
            lines.extend(self.operation(operation, indent + 1))
        lines.append(f"{prefix}}}")
        return lines

    def operation(self, operation: EmittedOperation, indent: int) -> list[str]:
        prefix = "  " * indent
        results = ", ".join(self.value_name(value) for value in operation.results)
        result_prefix = results + " = " if results else ""
        operands = ", ".join(self.value_name(value) for value in operation.operands)
        lines = [f'{prefix}{result_prefix}"intent.{operation.kind.value}"({operands})']
        if operation.regions:
            lines[-1] += " ("
            for index, region in enumerate(operation.regions):
                if index:
                    lines[-1] += ","
                lines.extend(self.region(region, indent + 1))
            lines.append(prefix + ")")
        attributes = self.operation_attributes(operation)
        operand_types = ", ".join(self.value_type(value) for value in operation.operands)
        result_types = self.result_types(tuple(value.type for value in operation.results))
        span = operation.location.primary
        location = f"loc({quote(span.filename)}:{span.start_line}:{span.start_column + 1})"
        lines[-1] += f" {emit_dictionary(attributes)} : ({operand_types}) -> {result_types} {location}"
        return lines

    def region(self, region: RegionState, indent: int) -> list[str]:
        prefix = "  " * indent
        lines = [prefix + "{"]
        for index, block in enumerate(region.blocks):
            arguments = ", ".join(self.argument(value) for value in block.arguments)
            signature = f"({arguments})" if arguments else ""
            lines.append(f"{prefix}  ^bb{index}{signature}:")
            for operation in block.operations:
                lines.extend(self.operation(operation, indent + 2))
        lines.append(prefix + "}")
        return lines

    @staticmethod
    def operation_attributes(operation: EmittedOperation) -> dict[str, object]:
        attributes = {
            **operation.attributes,
            "intent.node": operation.id,
            "intent.result_nodes": [value.id for value in operation.results],
            "intent.result_names": [value.name_hint or f"v{value.id}" for value in operation.results],
        }
        if operation.regions:
            attributes["intent.region_argument_nodes"] = [
                [[value.id for value in block.arguments] for block in region.blocks]
                for region in operation.regions
            ]
            attributes["intent.region_argument_names"] = [
                [[value.name_hint or f"v{value.id}" for value in block.arguments] for block in region.blocks]
                for region in operation.regions
            ]
        return attributes

    def value_type(self, value: MlirValue) -> str:
        if value.view_kind is not None:
            return emit_view_type(value.type, value.view_kind, value.view_constraints, self.dimension_id)
        return emit_type(value.type, self.dimension_id)

    @staticmethod
    def value_name(value: MlirValue) -> str:
        return f"%v{value.id}"

    def argument(self, value: MlirValue) -> str:
        return f"{self.value_name(value)}: {self.value_type(value)}"

    def result_types(self, types: tuple[ValueType, ...]) -> str:
        spellings = [emit_type(value, self.dimension_id) for value in types]
        return spellings[0] if len(spellings) == 1 else "(" + ", ".join(spellings) + ")"
