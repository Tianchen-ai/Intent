"""Canonical task modules through Triton–Weft JIT, plus author-owned host code."""
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from intent.compiler.cache import file_identity, locked_cache_entry
from intent.compiler.toolchain import CompilationStageError
from ..native_artifact import NativeArtifact, NativeBuildResult, FileDependency, build_native_artifact, run_native_command, write_source
from .binding import bridge_source
from .exports import native_exports


@dataclass(frozen=True)
class WeftArtifact:
    source: str
    contract: object
    ir: str
    directory: Path
    native: object
    tasks: tuple[Path, ...]
    kernels: tuple

    def export(self, directory):
        directory = Path(directory).expanduser().absolute()
        directory.mkdir(parents=True, exist_ok=False)
        self.native.validate()
        write_source(directory / "source.weftir", self.source)
        write_source(directory / "program.mlir", self.ir)
        write_source(directory / "program.json", json.dumps(self.contract.metadata))
        shutil.copyfile(self.native.library, directory / "kernel.so")
        if (self.native.directory / "lib").is_dir():
            shutil.copytree(self.native.directory / "lib", directory / "lib")
        for index, task in enumerate(self.tasks):
            shutil.copytree(task, directory / f"task-{index}")
        return directory

    @classmethod
    def read(cls, directory):
        from ..contract import ProgramContract
        directory = Path(directory).expanduser().absolute()
        source = (directory / "source.weftir").read_text()
        contract = ProgramContract.read(source, json.loads((directory / "program.json").read_text()))
        if contract.provider != "weft":
            raise ValueError("exported native program does not belong to Weft")
        tasks = tuple(directory / f"task-{i}" for i in range(len(contract.facts.tasks)))
        kernels = []
        for task, path in zip(contract.facts.tasks, tasks, strict=True):
            metadata = json.loads((path / "manifest.json").read_text())
            task.abi.verify_native(metadata["kernel"])
            kernels.append(metadata["kernel"])
        library = directory / "kernel.so"
        native = NativeArtifact(directory, library, tuple(FileDependency.capture(path)
            for path in directory.rglob("*") if path.is_file()), False,
            "Explicit native export loaded without compilation")
        return cls(source, contract, (directory / "program.mlir").read_text(), directory,
                   native, tasks, tuple(kernels))


def compile_program(program, target):
    import triton
    from triton.backends.weft.compiler import WeftBackend
    from weft_core import sdk_path

    target.validate_execution()
    contract = program._contract
    options = dict(target.options)
    selected = WeftBackend(target.native).parse_options(options)
    toolchain = selected.toolchain
    sdk = Path(sdk_path())
    compiled, abis = [], []
    key = json.dumps((program.source, str(target.native), options, target.cpus), sort_keys=True)
    with locked_cache_entry("weft-jit-sources", key) as entry:
        directory = entry.ready_attempt() or entry.create_attempt()
        for index, task in enumerate(contract.facts.tasks):
            path = directory / f"task-{index}.weftir"
            write_source(path, task.source)
            try:
                kernel = triton.compile(str(path), target=target.native, options=options)
            except Exception as error:
                raise CompilationStageError("provider_native_compilation", str(error),
                    cache_directory=directory, artifacts={"canonical": path}) from error
            task.abi.verify_native(kernel.metadata.kernel_abi)
            compiled.append(kernel)
            abis.append(kernel.metadata.kernel_abi)
        entry.publish(directory)
    bridge = bridge_source(contract.facts.tasks, abis)
    wrappers = native_exports(contract)
    dependencies = tuple(Path(path) for kernel in compiled for path in kernel.metadata.runtime_files.values())
    binaries = tuple(Path(next(path for name, path in kernel.metadata_group.items() if name.endswith(".so")))
                     for kernel in compiled)
    host_key = json.dumps((key, contract.facts.host_source, bridge, wrappers, toolchain,
                          tuple(file_identity(path) for path in (*binaries, *dependencies)),
                          file_identity(sdk / "include/weft/runtime.h"),
                          file_identity(sdk / "lib/libweft_runtime.so")), sort_keys=True)

    def build(directory):
        for index, kernel in enumerate(compiled):
            kernel.export(directory / f"task-{index}")
        for name, source in (("host.c", contract.facts.host_source), ("bridge.c", bridge), ("exports.c", wrappers)):
            write_source(directory / name, source)
        for name, source in toolchain.get("runtime_files", {}).items():
            output = directory / name
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output)
        library = directory / "kernel.so"
        command = [toolchain["cc"], "--target=" + toolchain["target"], "-O3", "-shared", "-fPIC",
                   "-std=c11", "-D_POSIX_C_SOURCE=200809L", "-fno-fast-math", "-ffp-contract=off",
                   *toolchain["cflags"], "-I" + str(sdk / "include"),
                   *(str(directory / name) for name in ("host.c", "bridge.c", "exports.c")),
                   "-L" + str(sdk / "lib"), "-Wl,-rpath," + str(sdk / "lib"),
                   "-lweft_runtime", "-lm", *toolchain["ldflags"], "-o", str(library)]
        if target.native.arch == "riscv64":
            command.extend((f"-march={target.native.march}", f"-mabi={target.native.abi}"))
        if toolchain["sysroot"]:
            command.append("--sysroot=" + toolchain["sysroot"])
        if contract.target.workers > 1:
            command.append("-fopenmp")
        run_native_command(command, directory, "provider_host_compilation")
        return NativeBuildResult(library, (Path(toolchain["cc"]), sdk / "lib/libweft_runtime.so",
            sdk / "include/weft/runtime.h", *binaries, *dependencies,
            *(directory / f"task-{i}/kernel.so" for i in range(len(compiled))),
            *(directory / name for name in ("host.c", "bridge.c", "exports.c"))))

    native = build_native_artifact("weft-host", host_key, build, reusable=True)
    return WeftArtifact(program.source, contract, program.ir, native.directory,
                        native, tuple(native.directory / f"task-{i}" for i in range(len(compiled))), tuple(abis))
