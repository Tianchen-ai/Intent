from dataclasses import dataclass
from functools import partial
from typing import ClassVar

from .nvidia import NVIDIATarget
from .provider import EnvironmentCheck, LocalExecution, Provider
from .execution import indexed_key, indexed_target
from .nvidia.specification import read_nvidia_target

INSTALLATION = {
    "modules": ("torch", "triton"),
    "requirements": ("triton==3.6.0", "numpy==1.26.4"), "torch": "2.10.0+cu130",
    "torch_index": "https://download.pytorch.org/whl/cu130",
    "python_max": (3, 12), "toolchain": "NVIDIA driver and CUDA-capable PyTorch",
}


@dataclass(frozen=True, slots=True)
class TritonTarget(NVIDIATarget):
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
    def compiler():
        if target is None:
            raise RuntimeError("Triton assembler discovery requires a resolved local GPU")
        from triton.backends.nvidia.compiler import get_ptxas, get_ptxas_version
        capabilities = target.compilation.capabilities
        arch = 10 * capabilities.compute_capability_major + capabilities.compute_capability_minor
        selected = get_ptxas(arch)
        return {"path": selected.path, "version": get_ptxas_version(arch).strip(),
                "resolver": "triton", "architecture": arch}

    return (EnvironmentCheck("Triton native assembler", "provider_toolchain", compiler),)


PROVIDER = Provider("gpu", TritonTarget, _read_facts, _bind, _environment_checks,
    LocalExecution("cuda", True, frozenset(), partial(indexed_target, TritonTarget), indexed_key),
    "block", read_nvidia_target)
