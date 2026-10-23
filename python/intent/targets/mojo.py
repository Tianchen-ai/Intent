from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING

from .provider import EnvironmentCheck, Provider
from .specification import CPUCompilationTarget, require_matching_target

if TYPE_CHECKING:
    from intent.runtime import CompiledArtifact


def _enable_integer_matrix(feature_set, cpu_flags):
    if not {"+amx-tile", "+amx-int8"}.issubset(feature_set):
        return False
    if not {"amx_tile", "amx_int8"}.issubset(cpu_flags):
        raise RuntimeError("Mojo AMX target features disagree with the execution CPU set")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    supported = ctypes.c_ulonglong()
    if libc.syscall(ctypes.c_long(158), ctypes.c_ulong(0x1021), ctypes.byref(supported)) != 0:
        raise OSError(ctypes.get_errno(), "cannot query Linux AMX XSTATE support")
    if supported.value & ((1 << 17) | (1 << 18)) != ((1 << 17) | (1 << 18)):
        return False
    # Linux grants dynamic XSTATE permission to the process, including existing
    # worker threads. No matrix instruction or tile allocation is issued here.
    if libc.syscall(ctypes.c_long(158), ctypes.c_ulong(0x1023), ctypes.c_ulong(18)) != 0:
        raise OSError(ctypes.get_errno(), "Linux denied AMX TILE_DATA permission")
    return True


@dataclass(frozen=True, slots=True)
class ResolvedMojoTarget:
    executable: str
    triple: str
    cpu: str
    features: str
    compilation: CPUCompilationTarget
    build_threads: int
    cpus: tuple[int, ...]

    def validate_execution(self) -> None:
        if tuple(sorted(os.sched_getaffinity(0))) != self.cpus:
            raise RuntimeError("Mojo execution CPU affinity changed; prepare a new local JIT call")

    @property
    def vector_bits(self) -> int:
        return self.compilation.vector_bits

    @property
    def workers(self) -> int:
        return self.compilation.workers

    @property
    def native_options(self) -> tuple[str, ...]:
        return (
            f"--target-triple={self.triple}", f"--target-cpu={self.cpu}",
            f"--target-features={self.features}", "--fp-mode=contract=off",
            f"--num-threads={self.build_threads}", "-O3",
        )

    def materialize(self, program) -> CompiledArtifact:
        require_matching_target(program.target, self.compilation)
        return _bind(program, self)


@dataclass(frozen=True, slots=True)
class MojoTarget:
    workers: int = 8
    compiler: str | Path | None = None
    build_threads: int = 4
    private_bytes: int = 262144

    def resolve(self) -> ResolvedMojoTarget:
        if self.workers <= 0 or self.build_threads <= 0:
            raise ValueError("Mojo worker and compiler thread budgets must be positive")
        selected = str(self.compiler) if self.compiler is not None else os.environ.get("INTENT_MOJO", "mojo")
        executable = shutil.which(selected)
        if executable is None:
            raise FileNotFoundError("Mojo compiler was not found; set INTENT_MOJO or MojoTarget(compiler=...)")
        cpus = tuple(sorted(os.sched_getaffinity(0)))
        # LLVM's native target query describes one executing CPU. Reject a set
        # with different ISA flags rather than applying that answer to all CPUs.
        flags = {}
        for record in Path("/proc/cpuinfo").read_text().split("\n\n"):
            fields = dict(line.split(":", 1) for line in record.splitlines() if ":" in line)
            fields = {key.strip(): value.strip() for key, value in fields.items()}
            if "processor" in fields and "flags" in fields:
                flags[int(fields["processor"])] = frozenset(fields["flags"].split())
        if not set(cpus).issubset(flags):
            raise RuntimeError("Mojo could not read ISA flags for the execution CPU set")
        if len({flags[cpu] for cpu in cpus}) != 1:
            raise NotImplementedError("Mojo local JIT requires an ISA-homogeneous CPU affinity set")
        result = subprocess.run([executable, "build", "--print-effective-target"],
                                text=True, capture_output=True, check=True)
        configuration = {}
        for line in result.stdout.splitlines():
            if line.strip().startswith("--target-"):
                key, value = line.strip().split(maxsplit=1)
                configuration[key] = value
        triple = configuration["--target-triple"]
        if not triple.startswith("x86_64-") or "linux" not in triple:
            raise NotImplementedError("Mojo CPU currently requires Linux x86-64")
        features = configuration["--target-features"]
        feature_set = set(features.split(","))
        if "+avx512f" in feature_set:
            vector_bits = 512
        elif "+avx2" in feature_set:
            vector_bits = 256
        else:
            raise NotImplementedError("Mojo CPU currently requires AVX2 or AVX512")
        compilation = CPUCompilationTarget("mojo", vector_bits, self.workers,
            private_bytes=self.private_bytes,
            matrix_i8_i32=_enable_integer_matrix(feature_set, flags[cpus[0]]))
        return ResolvedMojoTarget(executable, triple, configuration["--target-cpu"], features,
                                  compilation, self.build_threads, cpus)


def resolve_c_compiler(environment) -> str:
    from intent.compiler.toolchain import CompilationStageError
    executable = shutil.which("cc", path=environment.get("PATH"))
    if executable is None:
        raise CompilationStageError("native_toolchain_resolution",
                                    "Mojo CPU runtime compilation requires a C compiler (cc) and GNU OpenMP (libgomp.so.1)")
    return executable


def _read_facts(source, metadata, abi):
    from intent.runtime.mojo.contract import MojoFacts
    return MojoFacts.read(source, metadata, abi)


def _bind(program, target):
    from intent.runtime.mojo import materialize_mojo_artifact
    return materialize_mojo_artifact(program.source, program.ir, program._contract, target)


def _c_compiler():
    return {"path": resolve_c_compiler(os.environ),
            "scope": "Executable discovery only; compilation was not performed"}


def _environment_checks(target):
    return (EnvironmentCheck("C/OpenMP runtime compiler", "provider_toolchain", _c_compiler),)


PROVIDER = Provider("cpu", MojoTarget, _read_facts, _bind, _environment_checks)
