from dataclasses import dataclass
from typing import ClassVar

from .gpu import GPUTarget
from .provider import Provider


@dataclass(frozen=True, slots=True)
class TritonTarget(GPUTarget):
    provider: ClassVar[str] = "triton"


def _read_facts(source, metadata, abi):
    from intent.runtime.triton.contract import TritonFacts
    return TritonFacts.read(metadata["triton"], abi)


def _bind(program, target):
    from intent.runtime.triton import materialize_triton_artifact
    return materialize_triton_artifact(
        program.source, program.ir, program.entry_name, target.device, program._contract,
        source_path=program.cache_directory / "kernel.source")


def _environment_checks(target):
    return ()


PROVIDER = Provider("gpu", TritonTarget, _read_facts, _bind, _environment_checks)
