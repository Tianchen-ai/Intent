from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from ..diagnostics.locations import Location
from ..semantics.effects import Effect
from ..semantics.operations import OperationKind
from ..semantics.operations import REGION_OPS
from ..semantics.operations import TERMINATORS
from ..semantics.types import ValueType
from .state import BlockState
from .state import EmittedOperation
from .state import FunctionKind
from .state import FunctionState
from .state import MlirValue
from .state import ParameterKind
from .state import ParameterSpec
from .state import ParameterState
from .state import RegionState
from .serialization import AssemblyPrinter


class MlirBuilder:
    """Construct typed operations; spelling and native normalization are separate."""
    def __init__(self, module_name: str, location: Location) -> None:
        self.module_name = module_name
        self.location = location
        self.functions: list[FunctionState] = []
        self._next_value_id = 0
        self._next_operation_id = 0
        self._dimension_ids: dict[object, int] = {}
        self._dimensions: list[object] = []

    def _value(
        self,
        value_type: ValueType,
        location: Location,
        *,
        owner: BlockState | EmittedOperation,
        name_hint: str | None = None,
    ) -> MlirValue:
        from ..semantics.types import BufferType
        from ..semantics.types import DomainType
        from ..semantics.types import RegionType

        value_id = self._next_value_id
        if isinstance(value_type, (DomainType, RegionType, BufferType)) and (
            value_type.origin_id is None
        ):
            value_type = replace(value_type, origin_id=value_id)
        value = MlirValue(
            id=value_id,
            type=value_type,
            location=location,
            owner=owner,
            name_hint=name_hint,
        )
        self._next_value_id += 1
        return value

    def region(
        self,
        location: Location,
        argument_types: Iterable[ValueType] = (),
        argument_names: Iterable[str | None] = (),
        *,
        parent_block: BlockState | None = None,
        isolated_from_above: bool = False,
    ) -> RegionState:
        types = tuple(argument_types)
        names = tuple(argument_names)
        if names and len(names) != len(types):
            raise ValueError("region argument names and types must have equal length")
        if not names:
            names = (None,) * len(types)
        region = RegionState(
            location, _parent_block=parent_block,
            isolated_from_above=isolated_from_above,
        )
        block = BlockState(location, owner=region)
        region.blocks.append(block)
        block.arguments.extend(
            self._value(value_type, location, owner=block, name_hint=name)
            for value_type, name in zip(types, names)
        )
        return region

    def function(
        self,
        name: str,
        kind: FunctionKind,
        parameters: Iterable[ParameterSpec],
        result_types: Iterable[ValueType],
        location: Location,
        *,
        attributes: dict[str, object] | None = None,
    ) -> FunctionState:
        if not isinstance(name, str) or not name or not name.isidentifier():
            raise ValueError(f"invalid function name: {name!r}")
        parameter_specs = tuple(parameters)
        body = self.region(
            location,
            (parameter.type for parameter in parameter_specs),
            (parameter.name for parameter in parameter_specs),
        )
        values = body.blocks[0].arguments
        for spec, value in zip(parameter_specs, values):
            if spec.kind is ParameterKind.VIEW:
                value.view_kind = spec.view_kind
                value.view_constraints = spec.constraints
        function = FunctionState(
            name=name,
            kind=kind,
            parameters=tuple(
                ParameterState(spec, value) for spec, value in zip(parameter_specs, values)
            ),
            result_types=tuple(result_types),
            body=body,
            location=location,
            attributes=dict(attributes or {}),
        )
        self.functions.append(function)
        return function

    def emit(
        self,
        block: BlockState,
        operation_kind: OperationKind,
        location: Location,
        *,
        operands: Iterable[MlirValue] = (),
        result_types: Iterable[ValueType] = (),
        attributes: dict[str, object] | None = None,
        regions: Iterable[RegionState] = (),
        effects: Iterable[Effect] = (),
        result_names: Iterable[str | None] = (),
    ) -> EmittedOperation:
        if not isinstance(block, BlockState):
            raise TypeError("MLIR emission requires a BlockState")
        if not isinstance(operation_kind, OperationKind):
            raise TypeError("MLIR emission requires an OperationKind")
        if block.last_operation in TERMINATORS:
            raise ValueError("cannot emit after a block terminator")
        operands = tuple(operands)
        result_types = tuple(result_types)
        regions = tuple(regions)
        effects = tuple(effects)
        names = tuple(result_names)
        if any(not isinstance(value, MlirValue) for value in operands):
            raise TypeError("operation operands must be MLIR values")
        for value in operands:
            if not value.block.is_visible_from(block):
                raise ValueError(
                    f"intent.{operation_kind.value} operand %{value.id} is outside "
                    "its SSA scope; structured helpers require explicit arguments"
                )
        if any(not isinstance(value_type, ValueType) for value_type in result_types):
            raise TypeError("operation result types must be frontend ValueType values")
        if any(not isinstance(effect, Effect) for effect in effects):
            raise TypeError("operation effects must be Effect values")
        if any(effect.target is not None and effect.target not in operands for effect in effects):
            raise ValueError("effect target must be one of the operation operands")
        expected_regions = {
            OperationKind.IF: 2,
            OperationKind.WHILE: 2,
            OperationKind.REGION_FOLD: 2,
            OperationKind.REGION_SCAN: 4,
        }.get(operation_kind, 1)
        if operation_kind in REGION_OPS and len(regions) != expected_regions:
            raise ValueError(
                f"intent.{operation_kind.value} requires {expected_regions} region(s)"
            )
        if operation_kind not in REGION_OPS and regions:
            raise ValueError(f"intent.{operation_kind.value} does not own regions")
        for region in regions:
            if region.owner_operation is not None:
                raise ValueError("region is already attached to an operation")
            if len(region.blocks) != 1:
                raise NotImplementedError("Intent structured regions require one block")
            if region.blocks[0].last_operation not in TERMINATORS:
                raise ValueError("Intent structured region must end in a terminator")
        if names and len(names) != len(result_types):
            raise ValueError("result names and result types must have equal length")
        if not names:
            names = (None,) * len(result_types)

        operation_id = self._next_operation_id
        self._next_operation_id += 1
        operation = EmittedOperation(
            operation_id, operation_kind, location, block, operands, (),
            dict(attributes or {}), regions, effects,
        )
        operation.results = tuple(
            self._value(value_type, location, owner=operation, name_hint=name)
            for value_type, name in zip(result_types, names)
        )
        for region in regions:
            region.attach(operation)
        block.operations.append(operation)
        return operation

    def emit_module(self) -> str:
        return AssemblyPrinter(self.dimension_id).module(self.module_name, self.functions)

    def dimension_id(self, dimension: object) -> int:
        from ..semantics.types import StaticDim

        key = (StaticDim, id(dimension)) if isinstance(dimension, StaticDim) else dimension
        existing = self._dimension_ids.get(key)
        if existing is not None:
            return existing
        identity = len(self._dimensions) + 1
        self._dimensions.append(dimension)
        self._dimension_ids[key] = identity
        return identity

    def dimension(self, identity: int) -> object:
        if identity < 1 or identity > len(self._dimensions):
            raise ValueError(f"unknown logical dimension identity {identity}")
        return self._dimensions[identity - 1]
