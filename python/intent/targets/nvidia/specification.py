"""NVIDIA resource facts, independent of the CUDA runtime installation."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json

@dataclass(frozen=True, slots=True)
class NVIDIACapabilities:
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
class NVIDIACompilationTarget:
    """Explicit GPU compiler facts; constructing this target does not query a device or SDK."""

    provider: str
    capabilities: NVIDIACapabilities

    def __post_init__(self) -> None:
        from ..provider import provider
        if provider(self.provider).family != "gpu":
            raise ValueError(f"unsupported GPU source provider: {self.provider}")
        if not isinstance(self.capabilities, NVIDIACapabilities):
            raise TypeError("GPU compilation requires NVIDIACapabilities")

    def resolve(self) -> NVIDIACompilationTarget:
        return self

    @property
    def compiler_options(self) -> tuple[str, ...]:
        return (f"--target={self.provider}", *self.capabilities.compiler_options)

    @property
    def compiler_role(self) -> str:
        return f"Intent {self.provider} compiler"


def read_nvidia_target(provider: str, values: dict) -> NVIDIACompilationTarget:
    capabilities = NVIDIACapabilities(**values.pop("capabilities"))
    if values:
        raise ValueError(f"unknown NVIDIA target fields: {', '.join(values)}")
    return NVIDIACompilationTarget(provider, capabilities)


