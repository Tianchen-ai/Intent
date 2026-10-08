"""Explicit host bindings from kernel requests to existing compiled artifacts."""

from dataclasses import asdict, dataclass
from enum import Enum
import inspect
import json
from pathlib import Path
import tempfile

from intent.compiler.artifact import GeneratedProgram
from intent.language.annotations import ConstexprSpec
from intent.targets.specification import (
    CPUCompilationTarget, DSACompilationTarget, GPUCompilationTarget,
    require_matching_target,
)
from intent.targets.weft import WeftTarget


def _constant(value):
    if isinstance(value, Enum):
        cls = type(value)
        return {"type": "enum", "module": cls.__module__, "qualname": cls.__qualname__,
                "member": value.name, "value": _constant(value.value)}
    if type(value) is float:
        return {"type": "float", "value": value.hex()}
    if type(value) in (bool, int, str):
        return {"type": type(value).__name__, "value": value}
    raise TypeError(f"kernel assets cannot record constexpr type {type(value).__name__}")


def _constants(definition, supplied):
    supplied = {} if supplied is None else dict(supplied)
    parameters = definition.signature.parameters
    declarations = {name: parameter for name, parameter in parameters.items()
                    if isinstance(parameter.annotation, ConstexprSpec)}
    unknown = supplied.keys() - declarations.keys()
    if unknown:
        raise ValueError(f"unknown constexpr arguments: {', '.join(sorted(unknown))}")
    result = {}
    for name, parameter in declarations.items():
        value = supplied[name] if name in supplied else parameter.default
        if value is inspect.Parameter.empty:
            raise ValueError(f"constexpr {name!r} requires an explicit value")
        expected = parameter.annotation.value_type
        if (not isinstance(expected, type) or not isinstance(value, expected) or
                (expected is int and isinstance(value, bool)) or
                (issubclass(expected, Enum) and type(value) is not expected)):
            raise TypeError(f"constexpr {name!r} does not match its declared type")
        result[name] = _constant(value)
    return result


def _compilation_target(target):
    if isinstance(target, WeftTarget):
        # A saved AOT program needs no Intent/Weft compiler on its execution host.
        from intent.runtime.weft.target import matrix_capability
        capability = matrix_capability(target.matrix_extension, target.vector_bits)
        return CPUCompilationTarget("weft", target.vector_bits, target.workers,
                                    capability is not None, target.private_bytes)
    resolved = target.resolve()
    if isinstance(resolved, (CPUCompilationTarget, GPUCompilationTarget, DSACompilationTarget)):
        return resolved
    return resolved.compilation


def _target_record(target):
    return {"provider": target.provider,
            "facts": json.loads(json.dumps(asdict(target), allow_nan=False))}


def _request(definition, constexprs, target):
    return {"definition": {"module": definition.__module__, "qualname": definition.__qualname__},
            "constexprs": _constants(definition, constexprs),
            "target": _target_record(target)}


@dataclass(frozen=True)
class _WeftProgram:
    artifact: object

    @property
    def interface(self):
        return self.artifact.contract.interface

    @property
    def cache_directory(self):
        return self.artifact.directory

    @property
    def _contract(self):
        return self.artifact.contract


class KernelAssets:
    """One host program's explicit kernel bindings; never a compiler fallback."""

    def __init__(self, directory, *, mode):
        if mode not in {"export", "load"}:
            raise ValueError("kernel asset mode must be export or load")
        self.directory = Path(directory).expanduser().absolute()
        self.mode = mode
        self._manifest = self.directory / "manifest.json"
        self._stored = {}
        if mode == "load":
            manifest = json.loads(self._manifest.read_text(encoding="utf-8"))
            if (not isinstance(manifest, dict) or set(manifest) != {"program", "kernels"} or
                    not isinstance(manifest["program"], str) or
                    not isinstance(manifest["kernels"], list)):
                raise ValueError("invalid host kernel asset manifest")
            self._entries = manifest["kernels"]
        else:
            self._entries = []

    def _publish(self):
        manifest = {"program": self.directory.name, "kernels": self._entries}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                         prefix=".manifest-", delete=False) as temporary:
            path = Path(temporary.name)
            json.dump(manifest, temporary, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.write("\n")
        try:
            path.replace(self._manifest)
        finally:
            path.unlink(missing_ok=True)

    def store(self, definition, constexprs, target, program, stage="source"):
        if self.mode != "export":
            raise RuntimeError("kernel assets are not open for export")
        if stage not in {"source", "native", "run"}:
            raise ValueError("kernel asset stage must be source, native, or run")
        selected = _compilation_target(target)
        require_matching_target(program.target, selected)
        request = _request(definition, constexprs, selected)
        key = json.dumps(request, sort_keys=True, allow_nan=False)
        contents = (program.source, program.ir, program.metadata,
                    target.profile if isinstance(target, WeftTarget) else None)
        if key in self._stored:
            if self._stored[key] != contents:
                raise ValueError("the same kernel asset request produced different compiled programs")
            entry = next(entry for entry in self._entries if entry["request"] == request)
            return self._load_entry(entry, target, selected, stage)
        destination = self.directory / f"kernel-{len(self._entries):04d}"
        if program.target.provider == "weft":
            from intent.runtime.weft import export_artifact
            if not isinstance(target, WeftTarget) or target.profile is None or target.compiler is None:
                raise ValueError("Weft asset export requires an explicit native profile and Weft compiler")
            export_artifact(program, compiler=target.compiler, profile=target.profile).save(destination)
            kind = "weft-aot"
        else:
            program.save(destination)
            kind = "generated"
        entry = {"request": request, "target": _target_record(program.target),
                 "kind": kind, "artifact": destination.name}
        self._entries.append(entry)
        self._publish()
        self._stored[key] = contents
        return self._load_entry(entry, target, selected, stage)

    def resolve(self, definition, constexprs, target, stage):
        if self.mode != "load":
            raise RuntimeError("kernel assets are not open for loading")
        if stage not in {"source", "native", "run"}:
            raise ValueError("kernel asset stage must be source, native, or run")
        selected = _compilation_target(target)
        request = _request(definition, constexprs, selected)
        matches = [entry for entry in self._entries if entry["request"] == request]
        if len(matches) != 1:
            raise ValueError(f"kernel asset request requires exactly one binding, found {len(matches)}: {request}")
        return self._load_entry(matches[0], target, selected, stage)

    def _load_entry(self, entry, target, selected, stage):
        relative = Path(entry["artifact"])
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name in {"", ".", ".."}:
            raise ValueError("kernel asset must name a direct child directory")
        directory = self.directory / relative
        if entry["kind"] == "weft-aot":
            from intent.runtime.weft import WeftArtifact, compile_artifact, load_artifact
            # Only an absent local build record permits compilation. Invalid or
            # stale native records retain their original diagnostic.
            has_native = (directory / "native-attempt.json").exists()
            artifact = WeftArtifact.read(directory, load_native=stage != "source" and has_native)
            require_matching_target(artifact.contract.target, selected)
            if entry["target"] != _target_record(artifact.contract.target):
                raise ValueError("Weft asset target disagrees with its host registration")
            profile = getattr(target, "profile", None)
            if profile is not None and profile != artifact.profile:
                raise ValueError("Weft native profile disagrees with the saved AOT program")
            compiled = None
            if stage != "source":
                if artifact.native is None:
                    cc = getattr(target, "cc", None)
                    if not cc:
                        raise ValueError("portable Weft assets require an explicit C compiler before native use")
                    artifact = compile_artifact(artifact, cc=cc, cflags=target.cflags)
                compiled = load_artifact(artifact)
            result = (_WeftProgram(artifact), compiled)
        elif entry["kind"] == "generated":
            program = GeneratedProgram.load(directory)
            require_matching_target(program.target, selected)
            if entry["target"] != _target_record(program.target):
                raise ValueError("generated asset target disagrees with its host registration")
            result = (program, None if stage == "source" else program.materialize(target=target))
        else:
            raise ValueError(f"unknown kernel asset kind: {entry['kind']}")
        return result
