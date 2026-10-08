from __future__ import annotations

from intent.runtime.artifact import CompiledArtifact

def materialize_mojo_artifact(source: str, module_text: str,
                             contract, target) -> CompiledArtifact:
    from .program import NativeProgram

    program = NativeProgram(contract.abi, contract.facts, target)
    return CompiledArtifact(
        source=source, mlir=module_text, device=0, device_type="cpu",
        runtime=program,
        _backend_ir_collector=None,
    )
