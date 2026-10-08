from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .buffer import Buffer
from .target import TargetProfile

if TYPE_CHECKING:
    from intent.runtime.artifact import CompiledArtifact
    from .compilation import WeftArtifact


def load_artifact(artifact: WeftArtifact | str | Path) -> CompiledArtifact:
    """Bind an existing native AOT artifact without compilation or execution.

    Paths name an exported Weft artifact with a completed local native attempt.
    Portable sources must first pass through compile_artifact on this host.
    Library loading and execution checks remain deferred until invocation.
    """
    from intent.runtime.artifact import CompiledArtifact
    from .program import NativeProgram

    runtime = NativeProgram(artifact)
    exported = runtime.artifact
    return CompiledArtifact(source=exported.source,
                            mlir=exported.ir if exported.ir is not None else "",
                            device=0, device_type="cpu", runtime=runtime,
                            _backend_ir_collector=None,
                            cache_directory=exported.directory or runtime.directory)


def materialize_weft_artifact(program, target):
    from .compilation import export_artifact, compile_artifact

    exported = export_artifact(program, compiler=target.compiler, profile=target.profile)
    native = compile_artifact(exported, cc=target.cc, cflags=target.cflags)
    return load_artifact(native)


__all__ = ["Buffer", "TargetProfile", "WeftArtifact", "NativeProgram", "export_artifact", "compile_artifact", "load_artifact"]


def __getattr__(name):
    if name == "NativeProgram":
        from .program import NativeProgram
        return NativeProgram
    if name in {"WeftArtifact", "export_artifact", "compile_artifact"}:
        from . import compilation
        return getattr(compilation, name)
    raise AttributeError(name)
