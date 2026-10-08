"""Read generated execution contracts without importing a provider SDK."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intent.compiler.options import CompileOptions
    from intent.targets.specification import CompilationTarget
    from .interface import PublicInterface
    from .gpu.interface import GPUInterface
    from .native import NativeABI
    from .triton.contract import TritonFacts
    from .cutile.contract import CuTileFacts
    from .mojo.contract import MojoFacts
    from .weft.contract import WeftFacts
    from .bangc.contract import BangCFacts


def object_field(value, description: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be an object")
    return value


def name_field(value, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a nonempty name")
    return value


def integer_field(value, description: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{description} must be an integer >= {minimum}")
    return value


def sequence_field(value, description: str) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{description} must be an array")
    return tuple(value)


def names_field(value, description: str) -> tuple[str, ...]:
    names = tuple(name_field(item, description) for item in sequence_field(value, description))
    if len(set(names)) != len(names):
        raise ValueError(f"{description} must not contain duplicate names")
    return names


@dataclass(frozen=True, slots=True)
class CPUImplementation:
    """A CPU candidate's implementation description, shared by its providers."""

    name: str
    parameters: tuple[tuple[str, int], ...]

    @classmethod
    def read(cls, value) -> CPUImplementation:
        entry = object_field(value, "CPU implementation")
        parameters = object_field(entry["parameters"], "CPU implementation parameters")
        return cls(name_field(entry["name"], "CPU implementation name"), tuple(
            (name_field(name, "CPU implementation parameter"),
             integer_field(value, "CPU implementation parameter value", minimum=-(1 << 63)))
            for name, value in parameters.items()))

    def metadata(self) -> dict:
        return {"name": self.name, "parameters": dict(self.parameters)}


@dataclass(frozen=True, slots=True)
class ProgramContract:
    """One parsed projection of a private, immutable-by-ownership metadata snapshot."""

    provider: str
    target: CompilationTarget
    options: CompileOptions
    abi: GPUInterface | NativeABI
    facts: TritonFacts | CuTileFacts | MojoFacts | WeftFacts | BangCFacts
    _metadata: dict

    @property
    def interface(self) -> PublicInterface:
        from .gpu.interface import GPUInterface
        return self.abi.public if isinstance(self.abi, GPUInterface) else self.abi.interface

    @property
    def metadata(self) -> dict:
        """Return an inspection copy; changes cannot alter a bound program."""
        return deepcopy(self._metadata)

    @classmethod
    def read(cls, source: str, metadata: dict) -> ProgramContract:
        from intent.compiler.options import CompileOptions
        from intent.targets.specification import read_compilation_target
        from intent.targets.provider import provider as get_provider
        from .native import NativeABI

        if not isinstance(source, str) or not source.strip():
            raise ValueError("generated program source must be nonempty text")
        data = deepcopy(object_field(metadata, "generated program metadata"))
        try:
            provider = name_field(data["provider"], "provider")
            name_field(data["entry_name"], "generated entry name")
            target = read_compilation_target(provider, data["target"])
            adapter = get_provider(provider)
            options = CompileOptions.read(data["compile_options"])
            if adapter.family == "gpu":
                from .gpu.interface import GPUInterface
                abi = GPUInterface(data)
            else:
                abi = NativeABI.read(data)
            facts = adapter.read_facts(source, data, abi)
        except KeyError as error:
            raise ValueError(
                f"generated program lacks required field {error}; regenerate it from the original Intent definition or KIR"
            ) from error
        return cls(provider, target, options, abi, facts, data)
