from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..provider import provider
from ..specification import GPUCompilationTarget, require_matching_target
from .device import resolve_gpu_device

if TYPE_CHECKING:
    from intent.runtime import CompiledArtifact


@dataclass(frozen=True, slots=True)
class ResolvedGPUTarget:
    compilation: GPUCompilationTarget
    device: int

    def __post_init__(self) -> None:
        if provider(self.compilation.provider).family != "gpu":
            raise ValueError(f"unsupported GPU source provider: {self.compilation.provider}")

    def materialize(self, program) -> CompiledArtifact:
        current = GPUCompilationTarget(self.compilation.provider, resolve_gpu_device(self.device))
        require_matching_target(program.target, current)
        binding = provider(self.compilation.provider).bind
        if binding is None:
            raise NotImplementedError(f"{self.compilation.provider} does not provide a local runtime binding")
        return binding(program, self)


@dataclass(frozen=True, slots=True)
class GPUTarget(ABC):
    device: int = 0

    @property
    @abstractmethod
    def provider(self) -> str: ...

    def __post_init__(self) -> None:
        adapter = provider(self.provider)
        if adapter.family != "gpu":
            raise ValueError(f"unsupported GPU source provider: {self.provider}")
        if isinstance(self.device, bool) or not isinstance(self.device, int) or self.device < 0:
            raise ValueError(f"{adapter.target.__name__} target device must be a non-negative integer")

    def resolve(self) -> ResolvedGPUTarget:
        return ResolvedGPUTarget(
            GPUCompilationTarget(self.provider, resolve_gpu_device(self.device)), self.device)
