from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections.abc import Mapping
import os

from .provider import EnvironmentCheck, LocalExecution, Provider
from .execution import cpu_key, cpu_workers
from .specification import CPUCompilationTarget, read_cpu_target, require_matching_target
from .cpu import VectorCapabilities, ContractionCapability, ExecutionConfig

INSTALLATION = {
    "modules": ("triton", "weft_core"), "requirements": (), "torch": None,
    "torch_index": None, "python_max": (3, 14),
    "toolchain": "Triton–Weft with its matching weft-core native SDK and CPU toolchain; use a separate environment from other Triton distributions",
}


@dataclass(frozen=True, slots=True)
class StridedWeftTarget(CPUCompilationTarget):
    """The same native CPU target with a strided ABI for the named inputs."""
    strided_inputs: tuple[str, ...]


def _read_target(provider, values):
    if "strided_inputs" not in values:
        return read_cpu_target(provider, values)
    values = dict(values)
    names = values.pop("strided_inputs")
    if (not isinstance(names, list) or not names or
            any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names)):
        raise ValueError("Weft strided_inputs requires distinct input names")
    cpu = read_cpu_target(provider, values)
    return StridedWeftTarget(cpu.provider, cpu.vector, cpu.contractions, cpu.execution, tuple(names))


def _specialize(target, definition, arguments, outputs):
    from intent.language.annotations import ConstexprSpec, ViewKind, ViewSpec
    from intent.runtime.weft.buffer import Buffer
    if not any(isinstance(value, Buffer) and not value.canonical for value in arguments):
        return target, ()
    inputs = (p for p in definition.signature.parameters.values()
              if not isinstance(p.annotation, ConstexprSpec) and
              not (isinstance(p.annotation, ViewSpec) and p.annotation.kind is ViewKind.OUT))
    names = tuple(p.name for p, value in zip(inputs, arguments, strict=True)
                  if isinstance(value, Buffer) and not value.canonical and
                  isinstance(p.annotation, ViewSpec) and p.annotation.kind is ViewKind.IN)
    if not names:
        return target, ()
    cpu = target.compilation
    return replace(target, compilation=StridedWeftTarget(
        cpu.provider, cpu.vector, cpu.contractions, cpu.execution, names)), names


@dataclass(frozen=True, slots=True)
class ResolvedWeftTarget:
    compilation: CPUCompilationTarget
    native: object
    cpus: tuple[int, ...]
    options: Mapping = field(repr=False)

    def validate_execution(self):
        if tuple(sorted(os.sched_getaffinity(0))) != self.cpus:
            raise RuntimeError("Weft execution CPU affinity changed; prepare a new local JIT call")

    def materialize(self, program):
        require_matching_target(program.target, self.compilation)
        return _bind(program, self)


@dataclass(frozen=True, slots=True)
class WeftTarget:
    workers: int = 8
    private_bytes: int = 262144
    options: Mapping = field(default_factory=dict)

    def resolve(self):
        try:
            from triton.backends.weft.driver import WeftDriver
            from triton.backends.weft.compiler import WeftBackend
            from weft_core import compiler as core
        except ImportError as error:
            raise ImportError("Intent Weft requires the Triton–Weft distribution and matching weft-core; install them in this execution environment") from error
        workers = cpu_workers(self.workers)
        if type(self.private_bytes) is not int or self.private_bytes <= 0:
            raise ValueError("Weft private storage budget must be positive")
        native = WeftDriver().get_current_target()
        if native.arch == "x86_64":
            facts = core.x86_target_capabilities(native.march, native.abi, native.vlen_bits)
            vector_bits = facts["register_file"]["vector_bits"]
            matrix_i8_i32 = "amx-int8" in facts["matrix_extensions"]
        elif native.arch == "riscv64":
            vector_bits = native.vlen_bits
            matrix_i8_i32 = native.matrix_extension not in (None, "none")
        else:
            raise NotImplementedError(f"Weft CPU architecture is unsupported: {native.arch}")
        options = dict(self.options)
        if options.get("num_cpu_threads", 1) != 1:
            raise ValueError("Intent owns CPU task dispatch; Weft task launchers must use one thread")
        options["num_cpu_threads"] = 1
        if native.arch == "riscv64":
            options["private_stack_bytes"] = self.private_bytes
        WeftBackend(native).parse_options(options)
        return ResolvedWeftTarget(
            CPUCompilationTarget("weft", VectorCapabilities(vector_bits, native.arch == "riscv64"),
                (ContractionCapability("i8", "i8", "i32", False),) if matrix_i8_i32 else (),
                ExecutionConfig(workers, self.private_bytes)),
            native, tuple(sorted(os.sched_getaffinity(0))), options)


def _read_facts(source, metadata, abi):
    from intent.runtime.weft.contract import WeftFacts
    return WeftFacts.read(source, metadata, abi)


def _bind(program, target):
    from intent.runtime.weft import materialize_weft_artifact
    return materialize_weft_artifact(program, target)


def _environment_checks(target):
    def check():
        from triton.backends.weft.compiler import WeftBackend
        options = WeftBackend(target.native).parse_options(dict(target.options))
        return {"architecture": target.native.arch, "march": target.native.march,
                "abi": target.native.abi, "vlen_bits": target.native.vlen_bits,
                "cpus": target.cpus, "toolchain": options.toolchain}
    return (EnvironmentCheck("Triton–Weft native toolchain", "provider_toolchain", check),)


def _resolve_local(device, *, workers, **options):
    return WeftTarget(workers=cpu_workers(workers), **options).resolve()


def _host_storage(device, options):
    from intent.runtime.weft.staging import HostStorage
    return HostStorage()


PROVIDER = Provider("cpu", WeftTarget, _read_facts, _bind, _environment_checks,
    LocalExecution("cpu", False, frozenset({"options", "private_bytes"}),
                   _resolve_local, cpu_key, _host_storage, _specialize), "cpu", _read_target,
    "Triton–Weft owns native CPU discovery, canonical compilation, loading and worker validation")
