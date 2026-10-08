from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum

from intent.language.annotations import ViewConstraints
from intent.language.annotations import ViewKind

from ..diagnostics.locations import Location
from ..semantics.effects import Effect
from ..semantics.operations import OperationKind
from ..semantics.operations import TERMINATORS
from ..semantics.types import ValueType


class ParameterKind(Enum):
    VIEW = "view"
    RUNTIME_SCALAR = "runtime_scalar"
    CONSTEXPR = "constexpr"
    VALUE = "value"


class FunctionKind(Enum):
    KERNEL = "kernel"
    HELPER = "helper"


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    type: ValueType
    kind: ParameterKind
    location: Location
    view_kind: ViewKind | None = None
    constraints: ViewConstraints | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not self.name.isidentifier():
            raise ValueError(f"invalid parameter name: {self.name!r}")
        if not isinstance(self.type, ValueType):
            raise TypeError("parameter type must be a frontend ValueType")
        if not isinstance(self.kind, ParameterKind):
            raise TypeError("parameter kind must be a ParameterKind")
        if not isinstance(self.location, Location):
            raise TypeError("parameter location must be a Location")
        if self.kind is ParameterKind.VIEW:
            if not isinstance(self.view_kind, ViewKind) or not isinstance(
                self.constraints, ViewConstraints
            ):
                raise ValueError("view parameter requires view kind and constraints")
        elif self.view_kind is not None or self.constraints is not None:
            raise ValueError("only view parameters may carry view metadata")


@dataclass(eq=False, slots=True)
class MlirValue:
    id: int
    type: ValueType
    location: Location
    owner: BlockState | EmittedOperation
    name_hint: str | None = None
    view_kind: ViewKind | None = None
    view_constraints: ViewConstraints | None = None

    def __post_init__(self) -> None:
        if isinstance(self.id, bool) or not isinstance(self.id, int) or self.id < 0:
            raise ValueError("MLIR value id must be a non-negative integer")
        if not isinstance(self.type, ValueType):
            raise TypeError("MLIR value requires a frontend ValueType")
        if not isinstance(self.location, Location):
            raise TypeError("MLIR value location must be a Location")
        if self.view_kind is not None and not isinstance(self.view_kind, ViewKind):
            raise TypeError("view kind must be a ViewKind value")
        if self.view_constraints is not None and not isinstance(
            self.view_constraints, ViewConstraints
        ):
            raise TypeError("view constraints must be a ViewConstraints value")

    def __hash__(self) -> int:
        return self.id

    @property
    def block(self) -> BlockState:
        return self.owner if isinstance(self.owner, BlockState) else self.owner.block


@dataclass(frozen=True, slots=True)
class ParameterState:
    spec: ParameterSpec
    value: MlirValue


@dataclass(eq=False, slots=True)
class RegionState:
    location: Location
    blocks: list[BlockState] = field(default_factory=list)
    owner_operation: EmittedOperation | None = None
    _parent_block: BlockState | None = None
    isolated_from_above: bool = False

    @property
    def parent_block(self) -> BlockState | None:
        if self.owner_operation is not None:
            return self.owner_operation.block
        return self._parent_block

    def attach(self, operation: EmittedOperation) -> None:
        if self.owner_operation is not None:
            raise ValueError("region is already attached to an operation")
        if self._parent_block is not None and self._parent_block is not operation.block:
            raise ValueError("region must be attached in its construction scope")
        self.owner_operation = operation
        self._parent_block = None

    @property
    def effects(self) -> tuple[Effect, ...]:
        return tuple(effect for block in self.blocks for effect in block.effects)


@dataclass(eq=False, slots=True)
class BlockState:
    location: Location
    arguments: list[MlirValue] = field(default_factory=list)
    operations: list[EmittedOperation] = field(default_factory=list)
    owner: RegionState | None = None

    @property
    def operation_count(self) -> int:
        return len(self.operations)

    @property
    def last_operation(self) -> OperationKind | None:
        return self.operations[-1].kind if self.operations else None

    @property
    def terminated(self) -> bool:
        return self.last_operation in TERMINATORS

    @property
    def effects(self) -> tuple[Effect, ...]:
        return tuple(effect for op in self.operations for effect in op.all_effects)

    def has_effect_since(self, operation_count: int) -> bool:
        return any(operation.has_effects for operation in self.operations[operation_count:])

    @property
    def visible_parent(self) -> BlockState | None:
        """Enclosing SSA scope, stopping at an isolated region's arguments."""
        if self.owner is None or self.owner.isolated_from_above:
            return None
        return self.owner.parent_block

    def is_visible_from(self, block: BlockState) -> bool:
        """Whether this block's values may be used in the construction scope."""
        while block is not self:
            block = block.visible_parent
            if block is None:
                return False
        return True


@dataclass(eq=False, slots=True)
class FunctionState:
    name: str
    kind: FunctionKind
    parameters: tuple[ParameterState, ...]
    result_types: tuple[ValueType, ...]
    body: RegionState
    location: Location
    attributes: dict[str, object] = field(default_factory=dict)


@dataclass(eq=False, slots=True)
class EmittedOperation:
    id: int
    kind: OperationKind
    location: Location
    block: BlockState
    operands: tuple[MlirValue, ...]
    results: tuple[MlirValue, ...]
    attributes: dict[str, object]
    regions: tuple[RegionState, ...]
    effects: tuple[Effect, ...]

    @property
    def all_effects(self) -> tuple[Effect, ...]:
        return self.effects + tuple(effect for region in self.regions for effect in region.effects)

    @property
    def has_effects(self) -> bool:
        return bool(self.effects) or any(
            operation.has_effects
            for region in self.regions
            for block in region.blocks
            for operation in block.operations
        )
