from dataclasses import dataclass
from typing import ClassVar

from .gpu import GPUTarget
from .provider import EnvironmentCheck, Provider


@dataclass(frozen=True, slots=True)
class CuTileTarget(GPUTarget):
    provider: ClassVar[str] = "cutile"


def _read_facts(source, metadata, abi):
    from intent.runtime.cutile.contract import CuTileFacts
    return CuTileFacts.read(metadata["cutile"], abi)


def _bind(program, target):
    from intent.runtime.cutile import materialize_cutile_artifact
    return materialize_cutile_artifact(
        program.source, program.ir, program.entry_name, target.device, program._contract,
        source_path=program.cache_directory / "kernel.source")


def _compiler():
    from cuda.tile._compile import _find_compiler_bin
    return {"path": _find_compiler_bin().path, "resolver": "cuda.tile"}


def _environment_checks(target):
    return (EnvironmentCheck("cuTile native compiler", "provider_toolchain", _compiler),)


PROVIDER = Provider("gpu", CuTileTarget, _read_facts, _bind, _environment_checks)
