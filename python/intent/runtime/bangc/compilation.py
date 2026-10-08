from __future__ import annotations

import json
import os
from pathlib import Path

from intent.compiler.cache import file_identity
from intent.compiler.toolchain import CompilationStageError
from intent.targets.bangc import resolve_toolchain
from ..native_artifact import (
    NativeArtifact, NativeBuildResult, build_native_artifact,
    run_native_command, write_source,
)


def compile_library(source: str, target) -> NativeArtifact:
    """Produce an immutable CNCC artifact without opening a library or device."""
    environment = dict(os.environ)
    executable, neuware = resolve_toolchain(target, environment)
    library_dir = str(neuware / "lib64")
    options = (f"--neuware-path={neuware}", f"--bang-mlu-arch={target.compilation.architecture}", "-O2", "-fPIC", "-shared",
               "-std=c++14", "-ffp-contract=off", "-I" + str(neuware / "include"),
               "-L" + library_dir, "-Wl,-rpath," + library_dir, "-lcnrt")
    try:
        key = json.dumps((source, file_identity(executable), options), ensure_ascii=False)
    except OSError as error:
        raise CompilationStageError("native_toolchain_resolution", str(error)) from error

    def build(directory: Path) -> NativeBuildResult:
        path = directory / "kernel.mlu"
        write_source(path, source)
        library = directory / "kernel.so"
        run_native_command((executable, *options, str(path), "-o", str(library)),
                           directory, "native_compilation", environment=environment)
        return NativeBuildResult(library, (Path(executable), Path(library_dir) / "libcnrt.so"))

    # CNCC may resolve SDK headers, device libraries and host tools that are not
    # represented by its own executable identity. Keep the native evidence and
    # reuse the result inside its owning program, but do not publish a weak key.
    return build_native_artifact(
        "bangc", key, build, reusable=False,
        cache_reason="A complete CNCC/SDK dependency closure is unavailable; persistent binary reuse is disabled",
    )
