from dataclasses import dataclass
from functools import partial
from typing import ClassVar

from .nvidia import NVIDIATarget
from .provider import EnvironmentCheck, LocalExecution, Provider
from .execution import indexed_key, indexed_target
from .nvidia.specification import read_nvidia_target

INSTALLATION = {
    "modules": ("torch", "cuda.tile", "cuda.tile.tune"),
    "requirements": ("triton==3.6.0", "numpy==1.26.4",
                     "cuda-toolkit[tileiras,nvvm,nvcc]==13.3.1", "cuda-tile==1.6.0"),
    "torch": "2.10.0+cu128", "torch_index": "https://download.pytorch.org/whl/cu128",
    "python_max": (3, 12), "toolchain": "NVIDIA driver and the CUDA tile compiler",
}


@dataclass(frozen=True, slots=True)
class CuTileTarget(NVIDIATarget):
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


PROVIDER = Provider("gpu", CuTileTarget, _read_facts, _bind, _environment_checks,
    LocalExecution("cuda", True, frozenset(), partial(indexed_target, CuTileTarget), indexed_key),
    "block", read_nvidia_target)
