"""Triton-Ascend source component. SDK imports happen only during resolution."""
from dataclasses import dataclass
from functools import partial

from ..execution import indexed_key, indexed_target
from ..provider import EnvironmentCheck, LocalExecution, Provider
from ..specification import require_matching_target
from .specification import AscendCapabilities, AscendCompilationTarget

INSTALLATION = {
    "modules": ("torch", "torch_npu", "triton.backends.ascend"),
    "requirements": (), "torch": None, "torch_index": None, "python_max": (3, 12),
    "toolchain": "matching CANN, torch-npu and official Triton-Ascend",
}


@dataclass(frozen=True, slots=True)
class ResolvedAscendTarget:
    compilation: AscendCompilationTarget
    device: int

    def materialize(self, program):
        require_matching_target(program.target, self.compilation)
        return _bind(program, self)


@dataclass(frozen=True, slots=True)
class TritonAscendTarget:
    device: int = 0

    def resolve(self):
        from .discovery import resolve_device
        if type(self.device) is not int or self.device < 0:
            raise ValueError("Ascend device must be a nonnegative ordinal")
        return ResolvedAscendTarget(AscendCompilationTarget(resolve_device(self.device)), self.device)


def _read_target(provider, values):
    if set(values) != {"capabilities"}:
        raise ValueError("Ascend target requires its device capabilities")
    return AscendCompilationTarget(AscendCapabilities(**values["capabilities"]))


def _read_facts(source, metadata, abi):
    from intent.runtime.triton.contract import TritonFacts
    return TritonFacts.read(metadata["triton"], abi)


def _bind(program, target):
    from intent.runtime.triton_ascend import materialize
    return materialize(program.source, program.ir, program.entry_name, target.device,
                       program._contract, source_path=program.cache_directory / "kernel.source")


def _checks(target):
    def toolchain():
        from triton.backends.ascend.utils import _get_bisheng_path
        import subprocess
        executable = _get_bisheng_path()
        result = subprocess.run([executable, "-print-targets"], check=True, capture_output=True, text=True)
        if "hiipu64" not in result.stdout.split():
            raise RuntimeError("BiSheng does not provide the Ascend hiipu64 target")
        return {"path": executable, "targets": result.stdout.strip()}
    return (EnvironmentCheck("Ascend native compiler", "provider_toolchain", toolchain),)


def _host_storage(device, options):
    import torch_npu  # Registers the real NPU device before the first transfer.
    from ..execution import torch_storage
    return torch_storage(device, options)


PROVIDER = Provider("dsa", TritonAscendTarget, _read_facts, _bind, _checks,
    LocalExecution("npu", True, frozenset(), partial(indexed_target, TritonAscendTarget), indexed_key, _host_storage),
    "block", _read_target)
