from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..artifact import CompiledArtifact
from ..diagnostics import ConfigurationAssessment, NativeObservation, ObservedCall, resolve_observation
from intent.compiler.toolchain import CompilationStageError
from ..source import load_python_source
from .interface import BoundInvocation, GPUInterface


@dataclass(slots=True)
class LaunchResult:
    replay: Callable[[], object] | None
    kernel: object = None
    observation: NativeObservation | None = None


class Provider(Protocol):
    def compile(self, invocation: BoundInvocation) -> NativeObservation: ...

    def launch(self, invocation: BoundInvocation) -> LaunchResult: ...

    def tuning_configurations(self, invocation: BoundInvocation) -> tuple: ...

    def inspect_configurations(self, invocation: BoundInvocation) -> tuple[ConfigurationAssessment, ...]: ...


class PreparedCall(ObservedCall):
    """A call bound to these tensors and scalars, including invocation-owned workspace."""

    def __init__(self, program: GPUProgram, invocation: BoundInvocation) -> None:
        self.program = program
        self.invocation = invocation
        self._launch: LaunchResult | None = None
        self._observation: NativeObservation | None = None

    def inspect_configurations(self) -> tuple[ConfigurationAssessment, ...]:
        """Explain the bound invocation's candidates without allocation, JIT or tuning."""
        with self.program.invocation_context():
            return self.program.provider.inspect_configurations(self.invocation)

    @property
    def outputs(self) -> tuple:
        return self.invocation.outputs

    def result(self):
        return self.invocation.result()

    def compile(self) -> None:
        """Compile this binding's eligible candidates without tuning or launching."""
        import torch

        try:
            with self.program.invocation_context(), torch.cuda.device(self.program.device):
                details = self.program.provider.compile(self.invocation)
        except CompilationStageError as error:
            self._record_observation(error.observation)
            raise
        except Exception as error:
            raise CompilationStageError("provider_native_compilation", str(error)) from error
        self._record_observation(details)

    def launch(self) -> None:
        import torch

        with self.program.invocation_context():
            if torch.cuda.current_device() != self.program.device:
                with torch.cuda.device(self.program.device):
                    self._invoke()
            else:
                self._invoke()

    def _invoke(self):
        replay = self._launch is not None and self._launch.replay is not None
        try:
            if not replay:
                self._launch = self.program.provider.launch(self.invocation)
            else:
                self._launch.replay()
        except CompilationStageError as error:
            self._record_observation(error.observation)
            raise
        self._record_execution("launched", self._launch.observation, replay=replay)
        if self.program.artifact is not None:
            self.program.artifact._capture_backend_ir(self._launch.kernel)

    __call__ = launch


class GPUProgram:
    def __init__(self, interface: GPUInterface, provider: Provider, device: int) -> None:
        self.interface = interface.public
        self.binding = interface
        self.device = device
        self.provider = provider
        self.artifact: CompiledArtifact | None = None
        self._observation: NativeObservation | None = None
        self.invocation_context = nullcontext

    @property
    def observation(self) -> NativeObservation | None:
        return resolve_observation(self._observation)

    def prepare(self, *arguments, outputs: tuple | None = None,
                explicit_outputs: bool = False) -> PreparedCall:
        with self.invocation_context():
            invocation = self.binding.bind(arguments, device=self.device, outputs=outputs, explicit_outputs=explicit_outputs)
        return PreparedCall(self, invocation)

    def prepare_call(self, arguments: tuple, *, outputs: tuple | None = None) -> PreparedCall:
        return self.prepare(*arguments, outputs=outputs)

    def run(self, *arguments):
        call = self.prepare(*arguments)
        call.launch()
        return call.result()

    def infer_outputs(self, arguments: tuple):
        return self.binding.bind(arguments, device=self.device, abstract=True).result()

    def launch(self, *arguments) -> None:
        self.prepare(*arguments, explicit_outputs=True).launch()

    def tuning_configurations(self, *arguments):
        invocation = self.binding.bind(arguments, device=self.device, explicit_outputs=True)
        return self.provider.tuning_configurations(invocation)


def materialize_gpu_program(*, provider_name: str, provider_type, source: str,
                            module_text: str, contract, entry_name: str,
                            device: int, backend_ir_collector=None,
                            source_path: Path | None = None) -> CompiledArtifact:
    if contract.provider != provider_name:
        raise ValueError(f"{provider_name} runtime cannot load {contract.provider} metadata")
    interface = contract.abi
    namespace = load_python_source(target_name=provider_name, source=source, entry_name=entry_name,
                                   source_path=source_path)
    provider = provider_type(interface, namespace, contract.facts, contract.metadata["target"])
    program = GPUProgram(interface, provider, device)
    artifact = CompiledArtifact(source=source, mlir=module_text, device=device,
                                runtime=program,
                                _backend_ir_collector=backend_ir_collector)
    program.artifact = artifact
    return artifact
