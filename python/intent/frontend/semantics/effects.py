from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mlir.state import MlirValue


class EffectKind(Enum):
    READ = "read"
    WRITE = "write"
    ATOMIC = "atomic"


class ResourceKind(Enum):
    EXTERNAL_VIEW = "external_view"
    LOGICAL_BUFFER = "logical_buffer"


@dataclass(frozen=True, slots=True)
class Effect:
    kind: EffectKind
    resource: ResourceKind
    target: MlirValue | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EffectKind):
            raise TypeError("effect kind must be an EffectKind")
        if not isinstance(self.resource, ResourceKind):
            raise TypeError("effect resource must be a ResourceKind")
        if self.target is not None:
            from ..mlir.state import MlirValue

            if not isinstance(self.target, MlirValue):
                raise TypeError("effect target must be an SSA MlirValue or None")


PURE: tuple[Effect, ...] = ()
