from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from enum import IntEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from .interface import PublicInterface
from .diagnostics import ConfigurationAssessment, NativeObservation


BackendIRCollector = Callable[[object], dict[str, str]]


class ArtifactRuntime(Protocol):
    def run(self, *arguments: Any) -> object: ...

    def launch(self, *arguments: Any) -> object: ...


@runtime_checkable
class PreparedCall(Protocol):
    """Invocation-owned arguments and outputs, using the provider's launch semantics.

    result() returns output containers; it does not perform synchronization.
    """

    def compile(self) -> None:
        """Compile eligible native candidates without tuning or executing this call.

        An existing AOT native artifact is validated instead of recompiled.
        This does not require loading a native library, select a winner or
        certify native execution.
        """
        ...

    def launch(self) -> None:
        """Execute with the bound arguments and the provider's synchronization semantics."""
        ...

    def result(self) -> object:
        """Return the runtime's output containers without launching or synchronizing."""
        ...

    @property
    def observation(self) -> NativeObservation | None:
        """Read this call's latest native facts without execution."""
        ...

    def inspect_configurations(self) -> tuple[ConfigurationAssessment, ...]:
        """Explain this binding's declared candidates without choosing or timing one."""
        ...


@runtime_checkable
class PreparedRuntime(Protocol):
    def prepare_call(self, arguments: tuple, *, outputs: tuple | None = None) -> PreparedCall: ...


class ParameterRole(IntEnum):
    OWNERSHIP_M = 0
    OWNERSHIP_N = 1
    REDUCTION = 2
    SCAN_CHUNK = 3
    PROVIDER_WARPS = 4
    PROVIDER_STAGES = 5
    PROVIDER_CTAS = 6
    TRAVERSAL_WORKERS = 8
    TRAVERSAL_GROUP = 9
    RESIDENT_WORKERS = 10
    FULL_COVERAGE = 11
    REDUCTION_OUTER = 12
    REDUCTION_INNER = 13
    PROVIDER_ACCESS_FORM = 14
    PROVIDER_OCCUPANCY = 15
    PROVIDER_LOAD_POLICY = 16


@dataclass(frozen=True, slots=True)
class TuningParameter:
    name: str
    role: ParameterRole
    category: int
    candidates: tuple[int, ...]
    dimension: int | None
    source: tuple[int, int, bool] | None
    # Tensor argument position in artifact.entry, followed by its logical axis.
    view_axis: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class TuningConfiguration:
    parameters: tuple[TuningParameter, ...]
    values: tuple[int, ...]


@dataclass(slots=True)
class CompiledArtifact:
    source: str
    mlir: str
    device: int
    runtime: ArtifactRuntime = field(repr=False, metadata={
        "doc": "Explicit provider runtime implementing run/launch. Backend extensions use this object; "
               "additional capabilities depend on the provider, not a generated module namespace."
    })
    _backend_ir_collector: BackendIRCollector | None = field(repr=False)
    device_type: str = field(default="cuda", kw_only=True)
    cache_directory: Path | None = field(default=None, kw_only=True)
    backend_ir: dict[str, str] = field(default_factory=dict, init=False)
    _backend_ir_kernel: object = field(default_factory=object, init=False, repr=False)

    @property
    def observation(self) -> NativeObservation | None:
        """Latest native invocation snapshot, or None before a provider reports one.

        Reading it never compiles or launches. Resource fields come from the
        provider; unavailable fields are explicit and are not Intent estimates.
        """
        return getattr(self.runtime, "observation", None)

    @property
    def entry(self) -> Callable[..., None]:
        return self.__call__

    @property
    def ir(self) -> dict[str, str]:
        return {"intent": self.mlir, **self.backend_ir}

    def run(self, *arguments: Any) -> object:
        """Allocate declared Out buffers and execute with inputs in declaration order.

        Return None, the single Out, or an ordered tuple of Out buffers. InOut
        updates are observed through the original arguments, not extra results.
        """
        return self._invoke(self.runtime.run, arguments)

    @property
    def interface(self) -> PublicInterface:
        """Read the common typed public parameters, excluding compiler-private resources.

        This describes dtype, shape, stride and allocation constraints in author
        order. Device binding and native launch arguments remain runtime-owned.
        """
        if not hasattr(self.runtime, "interface"):
            raise NotImplementedError("this runtime does not expose a typed invocation interface")
        return self.runtime.interface

    def prepare(self, *arguments: Any, outputs: tuple | None = None) -> PreparedCall:
        """Bind arguments and allocate invocation-owned outputs/workspace without executing.

        The returned call's compile() can perform native JIT without execution;
        launch performs remaining first-use JIT/tuning and execution.
        GPU calls use the provider's current stream; CPU calls complete their join;
        BANG C calls synchronize their queue. result() only returns output containers.
        Explicit outputs replace declared Out buffers, in declaration order.
        Prepare again when arguments or their shape/stride metadata change, and keep
        their allocations alive while using the prepared call.
        """
        if not isinstance(self.runtime, PreparedRuntime):
            raise NotImplementedError("this runtime does not expose prepared calls")
        return self._invoke(lambda *args: self.runtime.prepare_call(args, outputs=outputs), arguments)

    def as_torch_op(self, name: str):
        """Return a PyTorch CustomOpDef with the declared Out and InOut behavior.

        GPU and Mojo CPU calls return fresh Out tensors, or None with no Out,
        and declare InOut mutations to the dispatcher. An InOut must not share
        Torch storage with another input; output aliases remain unsupported.
        Fake uses the same interface without execution or data-pointer access.
        Backward is not inferred. Authors may register it for functional calls;
        PyTorch does not accept register_autograd on mutable custom operators.
        """
        from .torch import register_operator

        return register_operator(self, name)

    def tuning_configurations(
        self, *arguments: Any,
    ) -> tuple[TuningConfiguration, ...]:
        """List eligible GPU bindings using all arguments, including explicit Out buffers.

        For an already prepared call, inspect_configurations() reuses its binding
        and includes reasons for rejected rows without allocating again.
        """
        if not hasattr(self.runtime, "tuning_configurations"):
            raise NotImplementedError(
                "this provider does not export structured tuning configurations"
            )
        return self._invoke(self.runtime.tuning_configurations, arguments)

    def inspect_configurations(self, *arguments: Any, outputs: tuple | None = None) -> tuple[ConfigurationAssessment, ...]:
        """Prepare inputs/optional Out buffers and explain candidates without execution.

        Preparation can allocate outputs and workspace. Use the prepared call's
        method to reuse an existing binding. Eligibility is not native compilation
        success; each provider reports its actual complete candidate set.
        """
        return self.prepare(*arguments, outputs=outputs).inspect_configurations()

    def launch(self, *arguments: Any) -> None:
        """Execute with every runtime argument, including explicit Out buffers."""
        compiled_kernel = self._invoke(self.runtime.launch, arguments)
        if compiled_kernel is not None:
            self._capture_backend_ir(compiled_kernel)

    __call__ = launch

    def _capture_backend_ir(self, compiled_kernel: object) -> None:
        if self._backend_ir_collector is not None and (
            compiled_kernel is not self._backend_ir_kernel or not self.backend_ir
        ):
            self.backend_ir = self._backend_ir_collector(compiled_kernel)
            self._backend_ir_kernel = compiled_kernel

    def _invoke(
        self,
        function: Callable[..., object],
        arguments: tuple[Any, ...],
    ) -> object:
        if self.device_type in {"cpu", "mlu"}:
            return function(*arguments)
        import torch

        expected = torch.device(self.device_type, self.device) if self.device_type == "cuda" else torch.device(self.device_type)
        for index, argument in enumerate(arguments):
            if isinstance(argument, torch.Tensor) and argument.device != expected:
                raise ValueError(
                    f"compiled artifact is bound to {expected}, but tensor argument "
                    f"{index} is on {argument.device}"
                )
        if expected.type == "cuda":
            if torch.cuda.current_device() == self.device:
                return function(*arguments)
            with torch.cuda.device(self.device):
                return function(*arguments)
        raise NotImplementedError(f"artifact invocation for {expected.type}")
