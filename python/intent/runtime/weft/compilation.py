from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import shutil
from uuid import uuid4

from intent.compiler.cache import file_identity, locked_cache_entry
from intent.compiler.toolchain import CompilationStageError

from .target import TargetProfile
from ..contract import ProgramContract
from ..native import NativeABI
from .. import native_artifact
from ..native_artifact import NativeArtifact, NativeBuildResult, build_native_artifact, run_native_command, write_source


def _executable(command: str) -> Path:
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(f"native compiler is not executable: {command}")
    return Path(resolved).absolute()


def lower_artifact(source: str, *, compiler: str, profile: TargetProfile,
                   source_bindings: tuple[tuple[str, int], ...] = ()) -> dict:
    executable = _executable(compiler)
    command = [str(executable), "--emit=artifact", f"--march={profile.march}", f"--abi={profile.abi}",
         f"--vlen-bits={profile.vlen_bits}",
         f"--private-stack-bytes={profile.private_stack_bytes}",
         *([f"--matrix-extension={profile.matrix_extension}"] if profile.matrix_extension else []),
         *(f"--meta={name}={value}" for name, value in source_bindings)]
    key = json.dumps((file_identity(executable), command, source, asdict(profile)))
    with locked_cache_entry("weft-source", key) as entry:
        directory = entry.create_attempt()
        write_source(directory / "canonical.mlir", source)
        output = run_native_command(command, directory, "provider_native_lowering", source=source)
        try:
            artifact = json.loads(output)
            _validate_target(artifact, profile)
        except (KeyError, TypeError, ValueError) as error:
            raise CompilationStageError("provider_native_lowering", str(error),
                                        cache_directory=directory) from error
        write_source(directory / "weft.json", output)
    return artifact


def _validate_target(artifact: dict, profile: TargetProfile) -> None:
    if artifact["kind"] != "weft-riscv-artifact":
        raise ValueError("Weft compiler did not produce a native artifact")
    for kernel in artifact["kernels"]:
        if (kernel["march"], kernel["abi"], kernel["vlen_bits"]) != (
            profile.march, profile.abi, profile.vlen_bits,
        ) or set(kernel["matrix_extensions"]) != ({profile.matrix_extension} if profile.matrix_extension else set()):
            raise ValueError("Weft artifact does not match the selected target profile")
        if not set(kernel["required_extensions"]).issubset(kernel["used_extensions"]) or not set(
                kernel["used_extensions"]).issubset(kernel["matrix_extensions"]):
            raise ValueError("Weft artifact extension requirements disagree with its selected instructions")
    # An Intent invocation can also contain RVV-only preparation and copy tasks.
    # Its extension requirement applies to the complete executable candidate.
    used = {extension for kernel in artifact["kernels"] for extension in kernel["used_extensions"]}
    if not set(profile.required_extensions).issubset(used):
        raise ValueError("Weft program does not use the required matrix extension")


def validate_artifact(manifest: dict, contract: ProgramContract) -> None:
    facts = contract.facts
    artifact = manifest["weft"]
    _validate_target(artifact, TargetProfile(**manifest["profile"]))
    kernels = {kernel["symbol"]: kernel for kernel in artifact["kernels"]}
    expected = {task.abi.symbol: task.abi for task in facts.tasks}
    if (len(kernels) != len(artifact["kernels"]) or
            len(expected) != len(facts.tasks) or kernels.keys() != expected.keys()):
        raise ValueError("Weft artifact kernel symbols disagree with the CPU task calls")
    for symbol, abi in expected.items():
        abi.verify_native(kernels[symbol])
    profile = TargetProfile(**manifest["profile"])
    for candidate in facts.candidates:
        used = {extension for task in facts.tasks
                if task.cpu_entry == candidate.entry
                for extension in kernels[task.abi.symbol]["used_extensions"]}
        required = set(profile.required_extensions)
        if candidate.requires_matrix_i8_i32:
            if profile.matrix_extension is None:
                raise ValueError("Weft candidate requires an unavailable integer matrix capability")
            required.add(profile.matrix_extension)
        if not required.issubset(used):
            raise ValueError("Weft candidate does not use the required matrix extension")


@dataclass(frozen=True, slots=True)
class WeftArtifact:
    """A parsed, portable AOT program and its optional local native build."""

    source: str
    contract: ProgramContract
    profile: TargetProfile
    _weft: dict
    ir: str | None = None
    directory: Path | None = None
    native: NativeArtifact | None = None
    native_cache_observed: bool = False

    @classmethod
    def create(cls, source: str, contract: ProgramContract, profile: TargetProfile,
               weft: dict, *, ir: str | None = None) -> WeftArtifact:
        if contract.provider != "weft":
            raise ValueError("Weft AOT requires a generated Weft CPU program")
        artifact = cls(source, contract, profile, deepcopy(weft), ir)
        validate_artifact(artifact.manifest, contract)
        return artifact

    @property
    def manifest(self) -> dict:
        return {"profile": asdict(self.profile), "program": self.contract.metadata,
                "weft": deepcopy(self._weft)}

    @property
    def kernels(self) -> tuple[dict, ...]:
        return tuple(deepcopy(self._weft["kernels"]))

    @classmethod
    def read(cls, directory: str | Path, *, load_native: bool = False) -> WeftArtifact:
        directory = Path(directory).expanduser().resolve()
        manifest = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
        source = (directory / "canonical.mlir").read_text(encoding="utf-8")
        contract = ProgramContract.read(source, manifest["program"])
        ir_path = directory / "cpu.mlir"
        artifact = cls.create(source, contract, TargetProfile(**manifest["profile"]), manifest["weft"],
                              ir=ir_path.read_text(encoding="utf-8") if ir_path.is_file() else None)
        artifact = replace(artifact, directory=directory)
        if load_native:
            record = json.loads((directory / "native-attempt.json").read_text(encoding="utf-8"))
            attempt = Path(record["directory"])
            if not attempt.is_absolute():
                raise ValueError("Weft native attempt must name an absolute local cache directory")
            native = NativeArtifact.read(attempt)
            if ((attempt / "artifact.json").read_text(encoding="utf-8") !=
                    json.dumps(artifact.manifest, sort_keys=True) or
                    (attempt / "canonical.mlir").read_text(encoding="utf-8") != source):
                raise ValueError("Weft native attempt does not belong to this exported program; compile it again")
            artifact = replace(artifact, native=native)
        return artifact

    def save(self, directory: str | Path) -> WeftArtifact:
        """Export portable sources; native attempts remain local and immutable."""
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        write_source(directory / "canonical.mlir", self.source)
        if self.ir is not None:
            write_source(directory / "cpu.mlir", self.ir)
        else:
            (directory / "cpu.mlir").unlink(missing_ok=True)
        write_source(directory / "kernels.c", self._weft["intrinsic_c"])
        write_source(directory / "host.c", self.contract.facts.host_source)
        write_source(directory / "artifact.json", json.dumps(self.manifest))
        (directory / "native-attempt.json").unlink(missing_ok=True)
        return replace(self, directory=directory, native=None, native_cache_observed=False)


def export_artifact(program, directory: Path | None = None, *, compiler: str,
                    profile: TargetProfile) -> WeftArtifact:
    """AOT lowering; system compilation and native loading remain separate."""
    from intent.targets.specification import CPUCompilationTarget

    target = program.target
    if not isinstance(target, CPUCompilationTarget) or target.provider != "weft":
        raise ValueError("Weft AOT requires a generated Weft CPU program")
    if target.matrix_i8_i32 and not profile.matrix_extension:
        raise ValueError("CPU program matrix capability disagrees with native materialization")
    weft = lower_artifact(program.source, compiler=compiler, profile=profile)
    artifact = WeftArtifact.create(program.source, program._contract, profile, weft, ir=program.ir)
    return artifact if directory is None else artifact.save(directory)


def flattened_signature(abi: NativeABI) -> tuple[list[str], list[str]]:
    signature, arguments = [], []
    carriers = {"bool": "_Bool", "i8": "int8_t", "i16": "int16_t", "i32": "int32_t",
                "i64": "int64_t", "f32": "float", "f64": "double"}
    for slot in abi.slots:
        name = slot.name
        if slot.role == "pointer":
            signature.append(f"void *{name}")
            arguments.append(name)
        else:
            signature.append(f"{carriers[slot.carrier]} {name}")
            arguments.append(name)
    return signature, arguments


def native_exports(contract: ProgramContract) -> str:
    signature, arguments = flattened_signature(contract.abi)
    mutable = contract.abi.trial_regions()
    sections = ["#include <stdint.h>\n#include <time.h>\n#include <fenv.h>\n"
                "#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n"]
    for candidate in contract.facts.candidates:
        entry = candidate.entry
        sections.append(
            f"extern void {entry}({', '.join(signature)});\n"
            f"void {entry}_invoke({', '.join(signature)}) {{\n"
            "  fenv_t saved; fegetenv(&saved); fesetround(FE_TONEAREST);\n"
            f"  {entry}({', '.join(arguments)});\n"
            "  fesetenv(&saved);\n}\n"
            f"double {entry}_benchmark({', '.join(signature)}, int64_t repetitions) {{\n"
            "  if (repetitions <= 0) return NAN;\n"
        )
        if not mutable:
            sections.append(
                "  struct timespec begin, end;\n"
                "  fenv_t saved; fegetenv(&saved); fesetround(FE_TONEAREST);\n"
                "  clock_gettime(CLOCK_MONOTONIC, &begin);\n"
                "  for (int64_t iteration = 0; iteration < repetitions; ++iteration)\n"
                f"    {entry}({', '.join(arguments)});\n"
                "  clock_gettime(CLOCK_MONOTONIC, &end); fesetenv(&saved);\n"
                "  return ((end.tv_sec - begin.tv_sec) * 1.e3 + (end.tv_nsec - begin.tv_nsec) * 1.e-6) / repetitions;\n}\n"
            )
            continue
        for region in mutable:
            name = region.pointer.name
            sections.append(f"  size_t bytes_{name} = {region.element_bytes};\n")
            for extent in region.extents:
                sections.append(
                    f"  if ({extent.name} < 0 || (bytes_{name} && (uint64_t){extent.name} > SIZE_MAX / bytes_{name})) return NAN;\n"
                    f"  bytes_{name} *= (size_t){extent.name};\n"
                )
        for region in mutable:
            sections.append(f"  void *saved_{region.pointer.name} = NULL;\n")
        for region in mutable:
            name = region.pointer.name
            sections.append(
                f"  if (bytes_{name}) {{\n"
                f"    saved_{name} = malloc(bytes_{name});\n"
                f"    if (!saved_{name}) goto allocation_failed;\n"
                f"    memcpy(saved_{name}, {name}, bytes_{name});\n"
                "  }\n"
            )
        sections.append(
            "  double elapsed = 0.0;\n"
            "  struct timespec begin, end;\n"
            "  fenv_t saved; fegetenv(&saved); fesetround(FE_TONEAREST);\n"
            "  for (int64_t iteration = 0; iteration < repetitions; ++iteration) {\n"
        )
        for region in mutable:
            name = region.pointer.name
            sections.append(f"    if (bytes_{name}) memcpy({name}, saved_{name}, bytes_{name});\n")
        sections.append(
            "    clock_gettime(CLOCK_MONOTONIC, &begin);\n"
            f"    {entry}({', '.join(arguments)});\n"
            "    clock_gettime(CLOCK_MONOTONIC, &end);\n"
            "    elapsed += (end.tv_sec - begin.tv_sec) * 1.e3 + (end.tv_nsec - begin.tv_nsec) * 1.e-6;\n"
            "  }\n"
        )
        for region in mutable:
            name = region.pointer.name
            sections.append(
                f"  if (bytes_{name}) memcpy({name}, saved_{name}, bytes_{name});\n"
                f"  free(saved_{name});\n"
            )
        sections.append("  fesetenv(&saved);\n  return elapsed / repetitions;\nallocation_failed:\n")
        for region in mutable:
            sections.append(f"  free(saved_{region.pointer.name});\n")
        sections.append("  return NAN;\n}\n")
    sections.append("int64_t intent_weft_vlen_bits(void) { unsigned long v; __asm__ volatile(\"csrr %0, vlenb\" : \"=r\"(v)); return v * 8; }\n")
    sections.append("void intent_weft_evict(void *storage, int64_t bytes) { volatile uint8_t *p = storage; for (int64_t i = 0; i < bytes; i += 64) p[i] = p[i] + 1; }\n")
    return "".join(sections)


def compile_artifact(artifact: WeftArtifact | str | Path, *, cc: tuple[str, ...],
                     cflags: tuple[str, ...] = ()) -> WeftArtifact:
    if not isinstance(artifact, WeftArtifact):
        artifact = WeftArtifact.read(artifact)
    if not cc:
        raise ValueError("Weft native compilation requires an explicit C compiler command")
    executable = _executable(cc[0])
    command = (str(executable), *cc[1:])
    contract, profile = artifact.contract, artifact.profile
    exports = native_exports(contract)
    manifest = json.dumps(artifact.manifest, sort_keys=True)
    implementation = (Path(__file__), Path(native_artifact.__file__))
    key = json.dumps((artifact.source, manifest, exports, command, cflags,
                      file_identity(executable), tuple(file_identity(path) for path in implementation)))

    def build(directory: Path) -> NativeBuildResult:
        write_source(directory / "canonical.mlir", artifact.source)
        write_source(directory / "artifact.json", manifest)
        write_source(directory / "host.c", contract.facts.host_source)
        write_source(directory / "kernels.c", artifact._weft["intrinsic_c"])
        write_source(directory / "exports.c", exports)
        library = directory / "kernel.so"
        run_native_command([
            *command, "-O3", "-shared", "-fPIC", "-std=c11", "-D_POSIX_C_SOURCE=200809L",
            f"-march={profile.march}", f"-mabi={profile.abi}", *cflags,
            "-fno-fast-math", "-ffp-contract=off",
            *(["-fopenmp"] if contract.target.workers > 1 else []),
            str(directory / "host.c"), str(directory / "kernels.c"), str(directory / "exports.c"),
            "-lm", "-o", str(library),
        ], directory, "provider_native_compilation")
        return NativeBuildResult(library, (executable, *implementation,
            *(directory / name for name in ("canonical.mlir", "artifact.json", "host.c", "kernels.c", "exports.c"))))

    native = build_native_artifact("weft-native", key, build, reusable=False,
        cache_reason="The supplied C compiler command has no declared complete compiler/linker dependency closure")
    compiled = replace(artifact, native=native, native_cache_observed=True)
    if artifact.directory is not None:
        temporary = artifact.directory / f".native-attempt-{uuid4().hex}"
        try:
            temporary.write_text(json.dumps({"directory": str(native.directory)}), encoding="utf-8")
            temporary.replace(artifact.directory / "native-attempt.json")
        finally:
            temporary.unlink(missing_ok=True)
    return compiled
