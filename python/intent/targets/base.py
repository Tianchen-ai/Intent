from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from intent.runtime import CompiledArtifact
from .specification import CompilationTarget

if TYPE_CHECKING:
    from intent.compiler.artifact import GeneratedProgram


@runtime_checkable
class ResolvedTarget(Protocol):
    """A local runtime binding with separately owned compiler facts."""

    @property
    def compilation(self) -> CompilationTarget: ...

    def materialize(self, program: GeneratedProgram) -> CompiledArtifact: ...


class SourceTarget(Protocol):
    def resolve(self) -> CompilationTarget | ResolvedTarget: ...


class Target(SourceTarget, Protocol):
    def resolve(self) -> ResolvedTarget: ...
