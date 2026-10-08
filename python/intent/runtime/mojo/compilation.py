from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import base64
import json
import os
from pathlib import Path
from threading import Lock
from uuid import uuid4

from intent.compiler.cache import cache_root, file_identity, locked_cache_entry
from intent.compiler.toolchain import CompilationStageError
from .toolchain import resolve_toolchain, runtime_dependencies
from .contract import MojoCandidate, MojoFacts
from ..native import NativeABI
from ..native_artifact import (
    NativeArtifact, NativeBuildResult, build_native_artifact,
    run_native_command, write_source,
)


@dataclass(frozen=True)
class NativeCompilation:
    artifacts: tuple[NativeArtifact, ...]
    identity: tuple[object, ...]


_compilations: dict[tuple[object, ...], Future[NativeCompilation]] = {}
_compilation_lock = Lock()
_compilers = ThreadPoolExecutor(max_workers=2)
_BUILD_OPTIONS = ("--emit", "shared-lib")

ELEMENT_TYPES = {
    "f16": "Float16", "bf16": "BFloat16", "f32": "Float32", "f64": "Float64",
    "bool": "SIMD[DType.bool, 1]", "index": "Int64", "i8": "Int8", "i16": "Int16", "i32": "Int32", "i64": "Int64",
    "u8": "UInt8", "u16": "UInt16", "u32": "UInt32", "u64": "UInt64",
    "f8e4m3fn": "SIMD[DType.float8_e4m3fn, 1]", "f8e5m2": "SIMD[DType.float8_e5m2, 1]",
}

def flattened_signature(abi: NativeABI) -> tuple[list[str], list[str]]:
    signature: list[str] = []
    arguments: list[str] = []
    for slot in abi.slots:
        name = slot.name
        if slot.role == "pointer":
            signature.append(f"{name}: Pointer[{ELEMENT_TYPES[slot.element.name]}, MutUntrackedOrigin]")
        else:
            signature.append(f"{name}: {'Bool' if slot.carrier == 'bool' else ELEMENT_TYPES[slot.carrier]}")
        arguments.append(name)
    return signature, arguments


def benchmark_exports(candidate: MojoCandidate, abi: NativeABI) -> str:
    signature, arguments = flattened_signature(abi)
    sections = ["\nfrom std.time import monotonic\nfrom std.memory import Layout, alloc, dealloc\n"]
    mutable = abi.trial_regions()
    entry = candidate.entry
    sections.append(
        f'\n@export("{entry}_benchmark")\n'
        f"def {entry}_benchmark({', '.join(signature)}, repetitions: Int64) abi(\"C\") -> Float64:\n"
    )
    if not mutable:
        sections.append(
            "    var begin = monotonic()\n"
            "    for iteration in range(Int(repetitions)):\n"
            f"        {entry}({', '.join(arguments)})\n"
            "    return Float64(monotonic() - begin) * 1.0e-6 / Float64(repetitions)\n"
        )
        return "".join(sections)
    for region in mutable:
        index = region.pointer.parameter.position
        dimensions = " * ".join(f"Int({slot.name})" for slot in region.extents) or "1"
        sections.append(
            f"    var bytes_a{index} = ({dimensions}) * {region.element_bytes}\n"
            f"    var saved_a{index} = alloc(Layout[UInt8](count=bytes_a{index}))\n"
            f'    external_call["memcpy", NoneType](saved_a{index}.unsafe_ptr(), a{index}, UInt(bytes_a{index}))\n'
        )
    sections.append("    var elapsed = Float64(0)\n    for iteration in range(Int(repetitions)):\n")
    for region in mutable:
        index = region.pointer.parameter.position
        sections.append(f'        external_call["memcpy", NoneType](a{index}, saved_a{index}.unsafe_ptr(), UInt(bytes_a{index}))\n')
    sections.append(
        "        var begin = monotonic()\n"
        f"        {entry}({', '.join(arguments)})\n"
        "        elapsed += Float64(monotonic() - begin)\n"
    )
    for region in mutable:
        index = region.pointer.parameter.position
        sections.append(
            f'    external_call["memcpy", NoneType](a{index}, saved_a{index}.unsafe_ptr(), UInt(bytes_a{index}))\n'
            f"    dealloc(saved_a{index}^)\n"
        )
    sections.append("    return elapsed * 1.0e-6 / Float64(repetitions)\n")
    return "".join(sections)


def _compile_fp_environment(environment: dict[str, str]) -> tuple[Path, bytes]:
    from intent.targets.mojo import resolve_c_compiler
    executable = resolve_c_compiler(environment)
    compiler = file_identity(Path(executable))
    source = Path(__file__).with_name("fp_environment.c").read_text(encoding="utf-8")
    options = ("-O2", "-fPIC", "-c")
    key = json.dumps((compiler, options, source))
    with locked_cache_entry("mojo-fp", key) as cache:
        source_path = cache.directory / "fp_environment.c"
        write_source(source_path, source)
        attempt = cache.create_attempt()
        output = attempt / "fp_environment.o"
        # Resolve headers and the C toolchain on each materialization. The actual
        # object is a native link input, shared by all candidates in this group.
        try:
            run_native_command(
                [executable, *options, str(source_path), "-o", str(output)],
                attempt, "fp_environment_compilation", environment=environment,
            )
            if file_identity(Path(executable)) != compiler:
                raise RuntimeError("C compiler changed during compilation")
            payload = output.read_bytes()
            if not payload:
                raise RuntimeError("C compiler produced an empty FP environment object")
        except Exception as error:
            raise CompilationStageError(
                "fp_environment_compilation", f"Mojo FP environment compilation failed: {error}\n"
                f"Native compiler artifacts: {attempt}",
                cache_directory=attempt, artifacts={"source": source_path},
            ) from error
        cache.publish(attempt)
        return output, payload


def _link_inputs(path: Path) -> list[Path]:
    # LLD emits Make escaping, not shell quoting. Only the first rule describes
    # link inputs; any following phony rules are not additional dependencies.
    rule = path.read_text(encoding="utf-8").replace("\\\n", "").splitlines()[0]
    escaped = False
    for index, character in enumerate(rule):
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == ":":
            rule = rule[index + 1:]
            break
    else:
        raise ValueError(f"linker dependency file has no target: {path}")
    words, word = [], []
    index = 0
    while index < len(rule):
        character = rule[index]
        if character == "\\":
            index += 1
            if index == len(rule):
                raise ValueError(f"incomplete linker dependency escape: {path}")
            word.append(rule[index])
        elif character == "$" and rule[index:index + 2] == "$$":
            word.append("$")
            index += 1
        elif character.isspace():
            if word:
                words.append(Path("".join(word)).absolute())
                word = []
        else:
            word.append(character)
        index += 1
    if word:
        words.append(Path("".join(word)).absolute())
    return words


def _runtime_paths(library: Path, environment: dict[str, str]) -> tuple[Path, ...]:
    return tuple(path.absolute() for path in runtime_dependencies(library, environment))


def runtime_inputs_current(artifact: NativeArtifact, environment: dict[str, str]) -> bool:
    recorded = json.loads((artifact.directory / "runtime.json").read_text(encoding="utf-8"))
    return recorded == [str(path) for path in _runtime_paths(artifact.library, environment)]


def _compile_unit(source: str, candidate: MojoCandidate, key: str, target,
                  environment: dict[str, str], snapshot, fp_object: Path) -> NativeArtifact:
    def build(directory: Path) -> NativeBuildResult:
        # The stable source retains its mtime for Mojo's own compilation cache.
        source_path = directory.parent.parent / "kernel.mojo"
        write_source(source_path, source)
        (directory / "artifact.json").write_text(
            json.dumps(candidate.metadata(), indent=2) + "\n", encoding="utf-8")
        (directory / "request.json").write_text(json.dumps({
            "source": str(source_path), "fp_object": str(fp_object),
            "cache_unavailable_reason": snapshot.reason,
        }, indent=2) + "\n", encoding="utf-8")
        library = directory / "kernel.so"
        dependencies = directory / "link.d"
        link_options = ["-Xlinker", str(fp_object)]
        build_environment = environment
        if snapshot.identity is not None:
            link_options += ["-Xlinker", "--dependency-file=" + str(dependencies)]
            temporary = directory / "temporary"
            temporary.mkdir()
            build_environment = {**environment, "TMPDIR": str(temporary)}
        run_native_command(
            [target.executable, "build", str(source_path), *_BUILD_OPTIONS,
             "-o", str(library), *target.native_options, *link_options],
            directory, "native_compilation", environment=build_environment,
        )
        snapshot.check_unchanged()
        runtime_paths = (_runtime_paths(library, environment)
                         if snapshot.identity is not None else ())
        runtime_record = directory / "runtime.json"
        runtime_record.write_text(json.dumps([str(path) for path in runtime_paths]) + "\n",
                                  encoding="utf-8")
        inputs = []
        if snapshot.identity is not None:
            inputs = [path for path in _link_inputs(dependencies)
                      if path.resolve() != fp_object.resolve() and
                      not path.resolve().is_relative_to(directory.parent.parent)]
        return NativeBuildResult(library, (runtime_record, *inputs, *runtime_paths))

    try:
        return build_native_artifact(
            "mojo", key, build, reusable=snapshot.identity is not None,
            cache_reason=snapshot.reason,
            validate_cached=lambda artifact: runtime_inputs_current(artifact, environment),
        )
    except CompilationStageError as error:
        raise CompilationStageError(
            error.stage, str(error), cache_directory=error.cache_directory,
            candidate=candidate.entry, artifacts=error.artifacts,
        ) from error


def compile_portfolio(facts: MojoFacts, target, *, abi: NativeABI) -> NativeCompilation:
    environment = dict(os.environ)
    try:
        snapshot = resolve_toolchain(target.executable, facts.native_dependencies, environment)
        fp_object, fp_bytes = _compile_fp_environment(environment)
    except CompilationStageError:
        raise
    except Exception as error:
        raise CompilationStageError("native_toolchain_resolution", str(error)) from error
    units = []
    for candidate in facts.candidates:
        complete_source = candidate.source + benchmark_exports(candidate, abi)
        unit_key = json.dumps((complete_source, candidate.metadata(), facts.native_dependencies, target.executable,
                              _BUILD_OPTIONS, target.native_options, snapshot.identity,
                              base64.b64encode(fp_bytes).decode("ascii")), sort_keys=True)
        units.append((complete_source, candidate, unit_key))
    # An unclosed toolchain is always delegated to Mojo, including repeated
    # requests in this process. It must never acquire a weak native cache key.
    key = (str(cache_root()), tuple(unit[2] for unit in units),
           None if snapshot.identity is not None else uuid4().hex)
    with _compilation_lock:
        pending = _compilations.get(key)
        owner = pending is None
        if owner:
            pending = Future()
            _compilations[key] = pending
    if not owner:
        return pending.result()

    futures = []
    try:
        for unit, candidate, unit_key in units:
            futures.append(_compilers.submit(_compile_unit, unit, candidate, unit_key,
                                             target, environment, snapshot, fp_object))
        artifacts = tuple(future.result() for future in futures)
        result = NativeCompilation(artifacts, (key, tuple(artifact.identity for artifact in artifacts)))
        snapshot.check_unchanged()
    except BaseException as error:
        for future in futures:
            future.cancel()
        wait(futures)
        with _compilation_lock:
            del _compilations[key]
        pending.set_exception(error)
        raise
    # Coalesce only concurrent requests. Later materializations revalidate disk
    # dependencies instead of keeping an unchecked, permanent successful Future.
    with _compilation_lock:
        del _compilations[key]
    pending.set_result(result)
    return result
