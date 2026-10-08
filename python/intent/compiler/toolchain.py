from __future__ import annotations

import subprocess
import json
import os
from pathlib import Path
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from .artifact import OptimizedIR
from .cache import _compilation_key, compilation_directory, file_identity, locked_cache_entry


class CompilationStageError(RuntimeError):
    def __init__(self, stage: str, message: str, *, cache_directory: Path | None = None,
                 candidate: str | None = None, artifacts: dict[str, Path] | None = None,
                 observation=None) -> None:
        super().__init__(message)
        self.stage = stage
        self.cache_directory = cache_directory
        self.candidate = candidate
        self.artifacts = dict(artifacts or {})
        self.observation = observation


def _resolve_executable(executable_path: str | Path | None, name: str,
                        environment_variable: str, stage: str, role: str) -> Path:
    selected = executable_path if executable_path is not None else os.environ.get(environment_variable)
    if selected is None:
        bundled = Path(__file__).resolve().parents[1] / "_bin" / name
        selected = bundled if bundled.exists() else shutil.which(name)
    if selected is None:
        raise CompilationStageError(
            stage, f"{name} was not found. Install IntentDSL with its compiler tools, "
            f"or set {environment_variable} / select an explicit {name} executable.",
        )
    executable = Path(selected).expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise CompilationStageError(
            stage, f"{role} is not an executable file: {executable}"
        )
    return executable


def _resolve_compiler(executable_path: str | Path | None, role: str) -> Path:
    return _resolve_executable(executable_path, "intent-compile", "INTENT_COMPILER",
                               "compiler_invocation", role)


class CompilerStage(str, Enum):
    KIR = "kir"
    SHARED = "shared"
    PROVIDER = "provider"

    @property
    def options(self) -> tuple[str, ...]:
        return () if self is CompilerStage.PROVIDER else (f"--stop-after-{self.value}",)

    @property
    def outputs(self) -> tuple[str, ...]:
        return ("kernel.mlir", "kernel.source", "artifact.json") if self is CompilerStage.PROVIDER else ("kernel.mlir",)


@dataclass(frozen=True)
class CompilerOutput:
    stage: CompilerStage
    directory: Path
    ir: str
    source: str | None = None
    metadata: dict[str, object] | None = None


@lru_cache(maxsize=16)
def _compiler_description(executable: Path, identity: tuple, library_path: str) -> str:
    # Only the compiled catalog is cached. Profile contents are read afresh for
    # each compile key, and no SDK or runtime device observations live here.
    try:
        completed = subprocess.run([str(executable), "--compiler-info"], text=True,
                                   capture_output=True, check=False)
    except OSError as error:
        raise CompilationStageError("compiler_invocation", str(error)) from error
    if completed.returncode:
        raise CompilationStageError("compiler_invocation", f"Cannot start {executable}:\n"
                                    f"{completed.stderr}{completed.stdout}")
    if file_identity(executable)[1] != identity:
        raise CompilationStageError("compiler_invocation", "compiler changed during capability discovery")
    return completed.stdout


def compiler_info(executable_path: str | Path | None = None) -> dict[str, object]:
    """Query the installed compiler catalog, without loading a target or device."""
    executable = _resolve_compiler(executable_path, "Intent compiler")
    description = _compiler_description(
        executable, file_identity(executable)[1], os.environ.get("LD_LIBRARY_PATH", ""))
    try:
        info = json.loads(description)
        if not isinstance(info, dict):
            raise ValueError("compiler info must be a JSON object")
        providers = info.get("providers")
        if not isinstance(providers, dict) or not providers:
            raise ValueError("compiler info providers must be a nonempty catalog")
        for name, provider in providers.items():
            if not isinstance(name, str) or not name or not isinstance(provider, dict) or set(provider) != {
                "family", "available", "profiles"
            }:
                raise ValueError("compiler provider requires a name, family, availability and profiles")
            if not isinstance(provider["family"], str) or not provider["family"] or type(provider["available"]) is not bool:
                raise ValueError("compiler provider family/availability are invalid")
            paths = provider["profiles"]
            if not isinstance(paths, list) or any(not isinstance(path, str) or not Path(path).is_absolute() for path in paths):
                raise ValueError("compiler profiles must be a list of absolute resource paths")
            if len(set(paths)) != len(paths) or (not provider["available"] and paths):
                raise ValueError("compiler profiles must be unique resources of an available provider")
        stages = info.get("stages")
        if not isinstance(stages, list) or not stages or any(not isinstance(item, str) for item in stages):
            raise ValueError("compiler info stages must be a nonempty list of names")
        outputs = info.get("outputs")
        if not isinstance(outputs, dict) or any(
            not isinstance(outputs.get(stage), list)
            or any(not isinstance(item, str) for item in outputs[stage])
            for stage in info["stages"]
        ):
            raise ValueError("compiler info must describe outputs for every stage")
        failures = info.get("failure_stages")
        if not isinstance(failures, dict) or not failures:
            raise ValueError("compiler info must describe failure stages by exit code")
        for code, stage in failures.items():
            if not code.isascii() or not code.isdecimal() or int(code) <= 0 or str(int(code)) != code:
                raise ValueError("compiler failure stage keys must be positive decimal exit codes")
            if not isinstance(stage, str) or not stage.strip():
                raise ValueError("compiler failure stages must have nonempty names")
    except (ValueError, TypeError) as error:
        raise CompilationStageError("compiler_output", f"Invalid compiler info from {executable}: {error}") from error
    return {**info, "executable": str(executable)}


def _compiler_resources(description: dict, options: tuple[str, ...],
                        stage: CompilerStage) -> tuple[Path, ...]:
    if stage.value not in description["stages"]:
        raise CompilationStageError("compiler_invocation", f"compiler does not provide the {stage.value!r} stage")
    if stage is CompilerStage.KIR:
        return ()
    provider = next((option.split("=", 1)[1] for option in options if option.startswith("--target=")), None)
    selected = description["providers"].get(provider)
    if selected is None or not selected["available"]:
        raise CompilationStageError("provider_program", f"selected compiler does not provide {provider!r}")
    if "--input-stage=shared" in options:
        return ()
    resources = [Path(path) for path in selected["profiles"]]
    if resources:
        for option in options:
            if option.startswith("--tuning-config="):
                resources.append(Path(option.split("=", 1)[1]).resolve())
    return tuple(resources)


def _read_resources(resources: tuple[Path, ...]) -> tuple[tuple[str, str], ...]:
    try:
        return tuple((str(path), path.read_text(encoding="utf-8")) for path in resources)
    except (OSError, UnicodeError) as error:
        raise CompilationStageError("compiler_resources", str(error)) from error


def _outputs(directory: Path, *, stage: CompilerStage, role: str) -> CompilerOutput:
    paths = tuple(directory / name for name in stage.outputs)
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise CompilationStageError("compiler_output", f"{role} did not produce complete outputs in {directory}",
                                    cache_directory=directory)
    try:
        ir = paths[0].read_text(encoding="utf-8")
        if stage is not CompilerStage.PROVIDER:
            return CompilerOutput(stage, directory, ir)
        source = paths[1].read_text(encoding="utf-8")
        metadata = json.loads(paths[2].read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("compiler metadata must be a JSON object")
        return CompilerOutput(stage, directory, ir, source, metadata)
    except (UnicodeDecodeError, ValueError) as error:
        raise CompilationStageError("compiler_output", f"{role} produced invalid outputs in {directory}: {error}",
                                    cache_directory=directory) from error


def run_compiler(executable_path: str | Path | None, module_text: str,
                 options: tuple[str, ...], role: str, *,
                 stage: CompilerStage = CompilerStage.PROVIDER) -> CompilerOutput:
    if not isinstance(stage, CompilerStage):
        raise TypeError("compiler stage must be a CompilerStage")
    executable = _resolve_compiler(executable_path, role)
    options = (*options, *stage.options)
    description = compiler_info(executable)
    resources = _compiler_resources(description, options, stage)
    resource_inputs = _read_resources(resources)
    with compilation_directory(executable, module_text, options, resource_inputs) as (directory, key):
        complete = directory / "complete"
        if complete.is_file():
            try:
                return _outputs(directory, stage=stage, role=role)
            except CompilationStageError:
                # An incomplete/invalid cached group is a miss. Fresh compiler
                # output below still has to pass the same validation.
                complete.unlink()
        (directory / "input.mlir").write_text(module_text, encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix=".building-", dir=directory) as temporary:
            staging = Path(temporary)
            command = [str(executable), *options, f"--ir-output={staging / 'kernel.mlir'}"]
            if stage is CompilerStage.PROVIDER:
                command.extend((f"--source-output={staging / 'kernel.source'}",
                                f"--metadata-output={staging / 'artifact.json'}"))
            command.append("-")
            try:
                completed = subprocess.run(command, input=module_text, text=True,
                                           capture_output=True, check=False)
            except OSError as error:
                raise CompilationStageError("compiler_invocation", str(error),
                                            cache_directory=directory) from error
            (directory / "compiler.log").write_text(completed.stderr + completed.stdout, encoding="utf-8")
            if completed.returncode:
                raise CompilationStageError(
                    description["failure_stages"].get(str(completed.returncode), "compiler_process"),
                    f"{role} failed with exit code {completed.returncode}:\n"
                    f"{completed.stderr}{completed.stdout}\nCompiler artifacts: {directory}",
                    cache_directory=directory,
                )
            for name in ("kernel.source", "kernel.mlir", "artifact.json"):
                (directory / name).unlink(missing_ok=True)
            for output in staging.iterdir():
                output.replace(directory / output.name)
        result = _outputs(directory, stage=stage, role=role)
        if _compilation_key(executable, module_text, options, _read_resources(resources)) != key:
            raise CompilationStageError(
                "compiler_invocation", "compiler or profiles changed during compilation; retry with stable inputs",
                cache_directory=directory,
            )
        complete.touch()
        return result


def run_optimizer(executable_path: str | Path | None, module_text: str,
                  pipeline: str) -> OptimizedIR:
    executable = _resolve_executable(executable_path, "intent-opt", "INTENT_OPTIMIZER",
                                     "optimizer_invocation", "Intent optimizer")
    options = (f"--pass-pipeline={pipeline}", "--mlir-print-debuginfo")
    key = _compilation_key(executable, module_text, options, ())
    # The existing identity/locking mechanism organizes diagnostic attempts.
    # A pass may read explicitly selected files, so no attempt is reused here.
    with locked_cache_entry("optimizer", key) as entry:
        directory = entry.create_attempt()
        (directory / "input.mlir").write_text(module_text, encoding="utf-8")
        command = [str(executable), *options, "-o", str(directory / "kernel.mlir"), "-"]
        (directory / "request.json").write_text(
            json.dumps({"tool_inputs": json.loads(key), "command": command,
                        "working_directory": os.getcwd(),
                        "pipeline": pipeline}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(command, input=module_text, text=True,
                                       capture_output=True, check=False)
        except OSError as error:
            (directory / "compiler.log").write_text(str(error), encoding="utf-8")
            raise CompilationStageError("optimizer_invocation", str(error),
                                        cache_directory=directory) from error
        (directory / "compiler.log").write_text(completed.stderr + completed.stdout, encoding="utf-8")
        if completed.returncode:
            raise CompilationStageError(
                "optimizer_process" if completed.returncode < 0 else "ir_optimization",
                f"Intent optimizer failed with exit code {completed.returncode}:\n"
                f"{completed.stderr}{completed.stdout}\nOptimizer artifacts: {directory}",
                cache_directory=directory,
            )
        path = directory / "kernel.mlir"
        if not path.is_file() or path.stat().st_size == 0:
            raise CompilationStageError("optimizer_output", "Intent optimizer produced no IR",
                                        cache_directory=directory)
        try:
            result = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise CompilationStageError("optimizer_output", str(error),
                                        cache_directory=directory) from error
        if _compilation_key(executable, module_text, options, ()) != key:
            raise CompilationStageError("optimizer_invocation", "optimizer or profiles changed during optimization",
                                        cache_directory=directory)
        return OptimizedIR(result, pipeline, directory)
