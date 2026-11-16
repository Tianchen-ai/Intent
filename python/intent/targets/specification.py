"""Compiler inputs and serialized target facts, independent of a runtime installation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from typing import Protocol, runtime_checkable
from .cpu import VectorCapabilities, ContractionCapability, ExecutionConfig


@dataclass(frozen=True, slots=True)
class CPUCompilationTarget:
    """CPU construction capabilities and budgets; no native toolchain or CPU affinity."""

    provider: str
    vector: VectorCapabilities
    contractions: tuple[ContractionCapability, ...]
    execution: ExecutionConfig

    def __post_init__(self) -> None:
        from .provider import provider
        if provider(self.provider).family != "cpu":
            raise ValueError(f"unsupported CPU source provider: {self.provider}")
        if not isinstance(self.vector, VectorCapabilities) or not isinstance(self.execution, ExecutionConfig):
            raise TypeError("CPU hardware vector facts and execution budgets must be separate typed objects")
        if not isinstance(self.contractions, tuple) or any(not isinstance(value, ContractionCapability) for value in self.contractions):
            raise TypeError("CPU contractions require a typed capability tuple")
        if len(set(self.contractions)) != len(self.contractions):
            raise ValueError("CPU contraction capabilities must be distinct")

    @property
    def vector_bits(self):
        return self.vector.minimum_bits

    @property
    def workers(self):
        return self.execution.workers

    @property
    def private_bytes(self):
        return self.execution.private_bytes

    def resolve(self) -> CPUCompilationTarget:
        return self

    @property
    def compiler_options(self) -> tuple[str, ...]:
        options = asdict(self)
        options.pop("provider")
        return (f"--target={self.provider}", f"--target-options={json.dumps(options)}")

    @property
    def compiler_role(self) -> str:
        return f"Intent CPU to {self.provider} compiler"


@runtime_checkable
class CompilationTarget(Protocol):
    """Compiler facts supplied by a registered provider; no device SDK is required."""

    @property
    def provider(self) -> str: ...

    @property
    def compiler_options(self) -> tuple[str, ...]: ...

    @property
    def compiler_role(self) -> str: ...


def read_compilation_target(provider: str, facts: dict) -> CompilationTarget:
    """Read target facts emitted by the compiler, without interpreting IR text."""
    if not isinstance(facts, dict):
        raise TypeError("compiler target facts must be a JSON object")
    values = dict(facts)
    family = values.pop("family")
    from .provider import provider as get_provider
    if get_provider(provider).execution_model != family:
        raise ValueError(f"unsupported compiler target family/provider: {family}/{provider}")
    return get_provider(provider).read_target(provider, values)


def read_cpu_target(provider: str, values: dict) -> CPUCompilationTarget:
    from .cpu import VectorCapabilities, ContractionCapability, ExecutionConfig
    expected = {field.name for field in fields(CPUCompilationTarget)} - {"provider"}
    if values.keys() != expected:
        raise ValueError(f"CPU target requires exactly these fields: {', '.join(sorted(expected))}")
    return CPUCompilationTarget(provider, VectorCapabilities(**values["vector"]),
        tuple(ContractionCapability(**item) for item in values["contractions"]),
        ExecutionConfig(**values["execution"]))


def require_matching_target(compiled: CompilationTarget, selected: CompilationTarget) -> None:
    if compiled != selected:
        raise ValueError(f"generated target {compiled} does not match the selected runtime target {selected}")
