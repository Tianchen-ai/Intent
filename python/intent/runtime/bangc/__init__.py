from __future__ import annotations

from pathlib import Path

from intent.runtime.artifact import CompiledArtifact
from .buffer import DeviceBuffer, DeviceView


def materialize_bangc_artifact(source, module_text, contract, target) -> CompiledArtifact:
    from .program import NativeProgram

    program = NativeProgram(source, contract, target)

    return CompiledArtifact(source=source, mlir=module_text, device=target.device,
                            runtime=program, _backend_ir_collector=None,
                            device_type="mlu")


def export_artifact(program, directory: str | Path) -> Path:
    return program.save(directory)


def load_artifact(directory: str | Path, *, target) -> CompiledArtifact:
    from intent.compiler.artifact import GeneratedProgram

    return GeneratedProgram.load(directory).materialize(target=target)


__all__ = ["DeviceBuffer", "DeviceView", "NativeProgram", "compile_library", "export_artifact", "load_artifact"]


def __getattr__(name):
    if name == "NativeProgram":
        from .program import NativeProgram
        return NativeProgram
    if name == "compile_library":
        from .compilation import compile_library
        return compile_library
    raise AttributeError(name)
