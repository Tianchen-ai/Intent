from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from collections.abc import Mapping
from pathlib import Path
import os
import shutil
from typing import TYPE_CHECKING

from .provider import EnvironmentCheck, Provider
from .specification import DSACompilationTarget, require_matching_target

if TYPE_CHECKING:
    from intent.runtime import CompiledArtifact


@dataclass(frozen=True, slots=True)
class ResolvedBangCTarget:
    compilation: DSACompilationTarget
    device: int
    neuware: str
    compiler: str | None

    def materialize(self, program) -> CompiledArtifact:
        require_matching_target(program.target, self.compilation)
        return _bind(program, self)


@dataclass(frozen=True, slots=True)
class BangCTarget:
    """BANG C device/toolchain binding; complete kernel configs are compiler inputs."""

    architecture: str = "mtp_372"
    device: int = 0
    neuware: str | Path = "/usr/local/neuware"
    compiler: str | Path | None = None
    shapes: Mapping[str, tuple[int, ...]] | None = None
    strides: Mapping[str, tuple[int, ...]] | None = None

    @classmethod
    def from_program(cls, program, *, device: int, neuware: str | Path,
                     compiler: str | Path | None = None) -> BangCTarget:
        """Select a local runtime while retaining this program's DSA construction bindings."""
        if not isinstance(program.target, DSACompilationTarget):
            raise ValueError("a BANG C runtime requires a DSA generated program")
        fields = asdict(program.target)
        fields.pop("capabilities")
        fields["shapes"] = dict(program.target.shapes)
        fields["strides"] = dict(program.target.strides)
        return cls(**fields, device=device, neuware=neuware, compiler=compiler)

    def resolve(self) -> ResolvedBangCTarget:
        if type(self.device) is not int or self.device < 0:
            raise ValueError("BANG C device must be a nonnegative integer")
        shapes = tuple(sorted((name, tuple(shape)) for name, shape in (self.shapes or {}).items()))
        strides = tuple(sorted((name, tuple(values)) for name, values in (self.strides or {}).items()))
        from .mlu import resolve_mlu_device
        neuware = str(Path(self.neuware).expanduser().resolve())
        capabilities = resolve_mlu_device(self.device, neuware)
        compilation = DSACompilationTarget(
            self.architecture, shapes, strides, capabilities)
        return ResolvedBangCTarget(
            compilation, self.device, str(Path(self.neuware).expanduser().resolve()),
            os.path.expanduser(str(self.compiler)) if self.compiler is not None else None,
        )


def specialize_call(target, definition, arguments, outputs):
    """Bind required DSA geometry from actual buffers, not from example helpers."""
    from intent.language.annotations import ConstexprSpec, ViewKind, ViewSpec
    parameters = [p for p in definition.signature.parameters.values()
                  if not isinstance(p.annotation, ConstexprSpec)]
    inputs = [p for p in parameters if not (isinstance(p.annotation, ViewSpec)
                                          and p.annotation.kind is ViewKind.OUT)]
    out_parameters = [p for p in parameters if p not in inputs]
    if len(arguments) != len(inputs) or (outputs is not None and len(outputs) != len(out_parameters)):
        raise TypeError("JIT arguments do not match the kernel input/output declaration")
    supplied = list(zip(inputs, arguments, strict=True))
    if outputs is not None:
        supplied.extend(zip(out_parameters, outputs, strict=True))
    shapes, strides = {}, {}
    for parameter, value in supplied:
        if not isinstance(parameter.annotation, ViewSpec):
            continue
        shapes[parameter.name] = tuple(value.shape)
        if callable(getattr(value, "stride", None)):
            stride = tuple(value.stride())
        elif hasattr(value, "__array_interface__"):
            stride = tuple(v // value.dtype.itemsize for v in value.strides)
        else:
            stride = tuple(value.strides)
        strides[parameter.name] = stride
    geometry = tuple(sorted(shapes.items())), tuple(sorted(strides.items()))
    return replace(target, compilation=replace(target.compilation,
                                               shapes=geometry[0], strides=geometry[1])), geometry


def resolve_toolchain(target, environment) -> tuple[str, Path]:
    from intent.compiler.toolchain import CompilationStageError
    neuware = Path(target.neuware).expanduser().resolve()
    selected = target.compiler or environment.get("INTENT_BANGC_CNCC") or str(neuware / "bin/cncc")
    executable = shutil.which(os.path.expanduser(selected), path=environment.get("PATH", os.defpath))
    if executable is None:
        raise CompilationStageError(
            "native_toolchain_resolution",
            "CNCC was not found; set BangCTarget(compiler=...) or INTENT_BANGC_CNCC",
        )
    return str(Path(executable).resolve()), neuware


def _read_facts(source, metadata, abi):
    from intent.runtime.bangc.contract import BangCFacts
    return BangCFacts.read(source, metadata, abi)


def _bind(program, target):
    from intent.runtime.bangc import materialize_bangc_artifact
    return materialize_bangc_artifact(program.source, program.ir, program._contract, target)


def _environment_checks(target):
    if target is None:
        return ()

    def toolchain():
        executable, neuware = resolve_toolchain(target, os.environ)
        library = neuware / "lib64/libcnrt.so"
        if not library.is_file():
            raise FileNotFoundError(f"CNRT runtime library is missing: {library}")
        return {"compiler": executable, "runtime": str(library)}

    def device():
        import ctypes
        from intent.runtime.bangc.buffer import runtime

        count = ctypes.c_uint()
        runtime(target.neuware).invoke("cnrtGetDeviceCount", ctypes.byref(count))
        if target.device >= count.value:
            raise ValueError(f"MLU device {target.device} is unavailable; CNRT reports {count.value} devices")
        return {"device": target.device, "visible_devices": count.value,
                "scope": "Runtime enumeration only; hardware compatibility and execution were not checked"}

    return (EnvironmentCheck("NeuWare tools", "provider_toolchain", toolchain),
            EnvironmentCheck("MLU device", "device", device))


PROVIDER = Provider("dsa", BangCTarget, _read_facts, _bind, _environment_checks)
