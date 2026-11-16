from .buffer import Buffer


def load_artifact(directory):
    """Open an explicit native export; compilation and execution are not implicit."""
    import os
    from types import SimpleNamespace
    from .compilation import WeftArtifact
    from .program import NativeProgram
    from intent.runtime.artifact import CompiledArtifact
    artifact = WeftArtifact.read(directory)
    cpus = tuple(sorted(os.sched_getaffinity(0)))
    def validate():
        if tuple(sorted(os.sched_getaffinity(0))) != cpus:
            raise RuntimeError("Weft execution affinity changed; load the artifact for the new CPU set")
    target = SimpleNamespace(compilation=artifact.contract.target, validate_execution=validate)
    program = SimpleNamespace(source=artifact.source, ir=artifact.ir,
                              _contract=artifact.contract, cache_directory=artifact.directory)
    runtime = NativeProgram(program, target, artifact=artifact)
    return CompiledArtifact(source=artifact.source, mlir=artifact.ir, device=0,
                            device_type="cpu", runtime=runtime, _backend_ir_collector=None,
                            cache_directory=artifact.directory)


def materialize_weft_artifact(program, target):
    from .program import NativeProgram
    from intent.runtime.artifact import CompiledArtifact

    runtime = NativeProgram(program, target)
    return CompiledArtifact(source=program.source, mlir=program.ir, device=0,
                            device_type="cpu", runtime=runtime, _backend_ir_collector=None,
                            cache_directory=program.cache_directory)


__all__ = ["Buffer", "load_artifact"]
