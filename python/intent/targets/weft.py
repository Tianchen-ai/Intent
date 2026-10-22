from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
import shutil
import os
import platform
import shlex
from typing import TYPE_CHECKING

from .provider import EnvironmentCheck, Provider
from .specification import CPUCompilationTarget, require_matching_target
from intent.runtime.weft.target import TargetProfile

if TYPE_CHECKING:
    from intent.runtime import CompiledArtifact


@dataclass(frozen=True, slots=True)
class ResolvedWeftTarget:
    compilation: CPUCompilationTarget
    profile: TargetProfile
    compiler: str
    cc: tuple[str, ...]
    cflags: tuple[str, ...]
    native: object | None = None

    def materialize(self, program) -> CompiledArtifact:
        require_matching_target(program.target, self.compilation)
        return _bind(program, self)


def native_target(*, workers, compiler=None, cc=None, cflags=(), private_bytes=262144):
    """Resolve the executing RISC-V CPU through the installed Weft native probe."""
    if platform.system() != "Linux" or platform.machine() != "riscv64":
        raise NotImplementedError("local Weft JIT requires a RISC-V Linux host")
    from weft.runtime.driver import Toolchain
    from weft.runtime.target import NativeTarget, extension_catalog
    selected = str(compiler) if compiler is not None else (
        os.environ.get("INTENT_WEFT_COMPILER") or os.environ.get("WEFT_COMPILER"))
    toolchain = Toolchain(compiler=selected)
    executable = toolchain.compiler_path()
    native = NativeTarget.discover(toolchain)
    facts = native.to_dict()
    cpus = tuple(facts["cpus"])
    if not set(os.sched_getaffinity(0)).issubset(cpus):
        raise ValueError("Weft discovery does not cover the calling process CPU affinity")
    catalog = extension_catalog(toolchain=toolchain) if facts["matrix_extensions"] else None
    extensions = []
    for name, allowed in facts["matrix_extensions"].items():
        if not set(cpus).issubset(allowed):
            continue
        fragments = catalog.get(name)["fragments"]
        if any((fragment["xlen"], fragment["vlen_bits"], fragment["lhs_element_bits"],
                fragment["rhs_element_bits"], fragment["accumulator_element_bits"]) ==
               (64, native.vlen_bits, 8, 8, 32) and
               all(fragment[field] == "signed" for field in
                   ("lhs_signedness", "rhs_signedness", "accumulator_signedness"))
               for fragment in fragments):
            extensions.append(name)
    if len(extensions) > 1:
        raise NotImplementedError("local Weft JIT has multiple supported matrix extensions")
    extension = extensions[0] if extensions else None
    profile = TargetProfile(facts["march"], facts["abi"], facts["vlen_bits"], cpus,
                            matrix_extension=extension, private_stack_bytes=private_bytes)
    command = tuple(cc) if cc is not None else tuple(shlex.split(os.environ.get("CC", "cc")))
    if not command or shutil.which(command[0]) is None:
        raise FileNotFoundError("local Weft JIT requires a native C compiler; set CC")
    return replace(WeftTarget(profile.vlen_bits, workers, extension, private_bytes,
                             profile, executable, command, tuple(cflags)).resolve(), native=native)


@dataclass(frozen=True, slots=True)
class WeftTarget:
    # These are explicit CPU program construction budgets, not ISA discovery.
    vector_bits: int
    workers: int
    matrix_extension: str | None = None
    private_bytes: int = 262144
    profile: TargetProfile | Mapping[str, object] | None = None
    compiler: str | Path | None = None
    cc: tuple[str, ...] | None = None
    cflags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.profile, Mapping):
            object.__setattr__(self, "profile", TargetProfile(**self.profile))
        elif self.profile is not None and not isinstance(self.profile, TargetProfile):
            raise TypeError("Weft native profile must be a TargetProfile or its field mapping")
        for name in ("cc", "cflags"):
            values = getattr(self, name)
            if values is None and name == "cc":
                continue
            if not isinstance(values, (tuple, list)) or any(
                    not isinstance(value, str) or not value for value in values):
                raise TypeError(f"Weft {name} must be a sequence of nonempty command arguments")
            object.__setattr__(self, name, tuple(values))
        if self.compiler is not None:
            if not isinstance(self.compiler, (str, Path)) or not str(self.compiler):
                raise TypeError("Weft compiler must name an executable")
            object.__setattr__(self, "compiler", str(Path(self.compiler).expanduser())
                               if isinstance(self.compiler, Path) or "~" in self.compiler
                               else self.compiler)

    def resolve(self) -> CPUCompilationTarget | ResolvedWeftTarget:
        compilation = CPUCompilationTarget("weft", self.vector_bits, self.workers,
                                           self.matrix_extension is not None, self.private_bytes)
        if self.profile is None and self.compiler is None and self.cc is None and not self.cflags:
            return compilation
        if self.profile is None or self.compiler is None or not self.cc:
            raise ValueError("Weft native binding requires an explicit profile, compiler and nonempty cc command")
        if self.profile.matrix_extension != self.matrix_extension:
            raise ValueError("Weft construction and native profile must name the same matrix extension")
        # vector_bits/workers describe the constructed program. The profile owns
        # physical VLEN and the allowed execution CPUs; they are checked at load.
        return ResolvedWeftTarget(compilation, self.profile, self.compiler,
                                  self.cc, self.cflags)


def _read_facts(source, metadata, abi):
    from intent.runtime.weft.contract import WeftFacts
    return WeftFacts.read(source, metadata, abi)


def _bind(program, target):
    from intent.runtime.weft import materialize_weft_artifact
    return materialize_weft_artifact(program, target)


def _environment_checks(target):
    if not isinstance(target, ResolvedWeftTarget):
        return ()

    def toolchain():
        paths = {}
        for name, command in (("weft_compiler", target.compiler), ("c_compiler", target.cc[0])):
            executable = shutil.which(command)
            if executable is None:
                raise FileNotFoundError(f"Weft {name} is not executable: {command}")
            paths[name] = str(Path(executable).absolute())
        return {**paths, "march": target.profile.march, "abi": target.profile.abi,
                "vlen_bits": target.profile.vlen_bits,
                "scope": "Executable discovery only; no compilation, library loading or device execution"}

    return (EnvironmentCheck("Weft native toolchain", "provider_toolchain", toolchain),)


PROVIDER = Provider("cpu", WeftTarget, _read_facts, _bind, _environment_checks,
                    "Source facts or explicit native toolchain discovery only; RISC-V execution is checked when loading native code")
