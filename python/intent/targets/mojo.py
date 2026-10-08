from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING

from .provider import EnvironmentCheck, Provider
from .specification import CPUCompilationTarget, require_matching_target

if TYPE_CHECKING:
    from intent.runtime import CompiledArtifact


@dataclass(frozen=True, slots=True)
class ResolvedMojoTarget:
    executable: str
    triple: str
    cpu: str
    features: str
    compilation: CPUCompilationTarget
    build_threads: int

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
                                            private_bytes=self.private_bytes)
        return ResolvedMojoTarget(executable, triple, configuration["--target-cpu"], features,
                                  compilation, self.build_threads)


def resolve_c_compiler(environment) -> str:
    from intent.compiler.toolchain import CompilationStageError
    executable = shutil.which("cc", path=environment.get("PATH"))
    if executable is None:
        raise CompilationStageError("native_toolchain_resolution",
                                    "Mojo floating-point environment compilation requires a C compiler (cc)")
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
    return (EnvironmentCheck("FP environment compiler", "provider_toolchain", _c_compiler),)


PROVIDER = Provider("cpu", MojoTarget, _read_facts, _bind, _environment_checks)
