"""Compiler inputs and serialized target facts, independent of a runtime installation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class GPUCapabilities:
    compute_units: int
    shared_memory_per_unit: int
    max_dynamic_shared_memory_per_block: int
    registers_per_unit: int
    max_threads_per_block: int
    compute_capability_major: int
    compute_capability_minor: int
    single_to_double_precision_perf_ratio: int
    matrix_units: bool
    dynamic_vector_width: bool

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            expected = bool if name in {"matrix_units", "dynamic_vector_width"} else int
            if type(value) is not expected:
                raise TypeError(f"GPU capability {name} must be {expected.__name__}")
            if expected is int and (value < 0 if name == "compute_capability_minor" else value <= 0):
                raise ValueError(f"GPU capability {name} violates the compiler's positive resource contract")

    @property
    def compiler_options(self) -> tuple[str, ...]:
        return (f"--target-options={json.dumps(asdict(self))}",)


@dataclass(frozen=True, slots=True)
class GPUCompilationTarget:
    """Explicit GPU compiler facts; constructing this target does not query a device or SDK."""

    provider: str
    capabilities: GPUCapabilities

    def __post_init__(self) -> None:
        from .provider import provider
        if provider(self.provider).family != "gpu":
            raise ValueError(f"unsupported GPU source provider: {self.provider}")
        if not isinstance(self.capabilities, GPUCapabilities):
            raise TypeError("GPU compilation requires GPUCapabilities")

    def resolve(self) -> GPUCompilationTarget:
        return self

    @property
    def compiler_options(self) -> tuple[str, ...]:
        return (f"--target={self.provider}", *self.capabilities.compiler_options)

    @property
    def compiler_role(self) -> str:
        return f"Intent {self.provider} compiler"


@dataclass(frozen=True, slots=True)
class CPUCompilationTarget:
    """CPU construction capabilities and budgets; no native toolchain or CPU affinity."""

    provider: str
    vector_bits: int
    workers: int
    matrix_i8_i32: bool = False
    private_bytes: int = 262144
    matrix_bf16_f32: bool = False

    def __post_init__(self) -> None:
        from .provider import provider
        if provider(self.provider).family != "cpu":
            raise ValueError(f"unsupported CPU source provider: {self.provider}")
        if any(type(value) is not int for value in (self.vector_bits, self.workers, self.private_bytes)) or any(
            type(value) is not bool for value in (self.matrix_i8_i32, self.matrix_bf16_f32)
        ):
            raise TypeError("CPU budgets must be integers and matrix capabilities must be boolean")
        if min(self.vector_bits, self.workers, self.private_bytes) <= 0:
            raise ValueError("CPU compilation capabilities and storage budgets must be positive")

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


@dataclass(frozen=True, slots=True)
class DSACapabilities:
    clusters: int
    cores_per_cluster: int
    nram_bytes: int
    wram_bytes: int
    sram_bytes: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("DSA capabilities require positive integer hardware capacities")


@dataclass(frozen=True, slots=True)
class DSACompilationTarget:
    """BANG C hardware identity and source ABI refinements, independent of tuning."""

    architecture: str = "mtp_372"
    # Source ABI refinements are recorded in the resulting interface, not in
    # target metadata. They do not participate in runtime hardware matching.
    shapes: tuple[tuple[str, tuple[int, ...]], ...] = field(default=(), compare=False)
    strides: tuple[tuple[str, tuple[int, ...]], ...] = field(default=(), compare=False)
    capabilities: DSACapabilities | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.capabilities, DSACapabilities):
            raise TypeError("DSA compilation requires explicit DSACapabilities; local JIT discovers them")
        if self.architecture != "mtp_372":
            raise NotImplementedError("BANG C currently provides an MLU370 implementation profile")
        if any(not isinstance(name, str) or any(type(extent) is not int or extent < -1 for extent in shape)
               for name, shape in self.shapes):
            raise ValueError("DSA shape bindings require integer extents, with -1 for dynamic axes")
        if any(not isinstance(name, str) or any(type(value) is not int or not -(1 << 63) <= value < (1 << 63)
               for value in strides) for name, strides in self.strides):
            raise ValueError("DSA strides must be signed 64-bit element strides")

    @property
    def provider(self) -> str:
        return "bangc"

    def resolve(self) -> DSACompilationTarget:
        return self

    @property
    def compiler_options(self) -> tuple[str, ...]:
        options = asdict(self)
        options.update(shapes=dict(self.shapes), strides=dict(self.strides))
        return ("--target=bangc", f"--target-options={json.dumps(options)}")

    @property
    def compiler_role(self) -> str:
        return "Intent DSA compiler"


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


def read_gpu_target(provider: str, values: dict) -> GPUCompilationTarget:
    capabilities = GPUCapabilities(**values.pop("capabilities"))
    if values:
        raise ValueError(f"unknown NVIDIA target fields: {', '.join(values)}")
    return GPUCompilationTarget(provider, capabilities)


def read_cpu_target(provider: str, values: dict) -> CPUCompilationTarget:
    expected = {field.name for field in fields(CPUCompilationTarget)} - {"provider"}
    if values.keys() != expected:
        raise ValueError(f"CPU target requires exactly these fields: {', '.join(sorted(expected))}")
    return CPUCompilationTarget(provider, **values)


def read_bangc_target(provider: str, values: dict) -> DSACompilationTarget:
    expected = {field.name for field in fields(DSACompilationTarget)} - {"shapes", "strides"}
    if values.keys() != expected:
        raise ValueError(f"BANG C target requires exactly these fields: {', '.join(sorted(expected))}")
    values["capabilities"] = DSACapabilities(**values["capabilities"])
    return DSACompilationTarget(**values)


def require_matching_target(compiled: CompilationTarget, selected: CompilationTarget) -> None:
    if compiled != selected:
        raise ValueError(f"generated target {compiled} does not match the selected runtime target {selected}")
