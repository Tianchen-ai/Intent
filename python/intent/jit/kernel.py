from __future__ import annotations

from copy import deepcopy
from threading import RLock

from intent.api import KernelDefinition
from intent.compiler.pipeline import generate
from intent.language.annotations import ConstexprSpec, ViewKind, ViewSpec
from intent.targets.base import ResolvedTarget
from intent.targets.local import execution_key, resolve_local, select_device
from intent.targets.provider import provider


class JITKernel:
    """One author definition, with device-specific artifacts and fresh call bindings."""

    def __init__(self, definition, *, backend, device=None, compiler=None,
                 constexprs=None, tuning_config=None, options=None, workers=None,
                 backend_options=None):
        if not isinstance(definition, KernelDefinition):
            raise TypeError("intent.jit requires an @intent.kernel definition")
        provider(backend)
        self.definition, self.backend, self.device = definition, backend, device
        self.compiler, self.tuning_config, self.options = compiler, tuning_config, options
        self.constexprs = deepcopy(dict(constexprs or {}))
        self.workers, self.backend_options = workers, deepcopy(dict(backend_options or {}))
        self._targets, self._artifacts = {}, {}
        self._lock = RLock()
        self._closed = False
        self.__name__ = definition.__name__
        self.__doc__ = definition.__doc__

    @property
    def artifacts(self):
        """Artifacts produced so far; inspecting this property does not compile."""
        with self._lock:
            return tuple(self._artifacts.values())

    @property
    def interface(self):
        """Typed public interface after preparing one local specialization."""
        artifacts = self.artifacts
        if len(artifacts) != 1:
            raise ValueError("prepare one local specialization before inspecting its interface")
        return artifacts[0].interface

    def _artifact(self, arguments, outputs):
        device = select_device(self.backend, (*arguments, *(outputs or ())), self.device)
        execution = execution_key(device)
        with self._lock:
            if self._closed:
                raise RuntimeError("JIT kernel is closed")
            if execution not in self._targets:
                resolved = resolve_local(self.backend, device, workers=self.workers,
                                         backend_options=self.backend_options)
                if not isinstance(resolved, ResolvedTarget):
                    raise NotImplementedError(f"{self.backend} has no local executable target")
                self._targets[execution] = resolved
            target = self._targets[execution]
            specialization = ()
            if self.backend == "bangc":
                from intent.targets.bangc import specialize_call
                target, specialization = specialize_call(target, self.definition, arguments, outputs)
            key = execution, specialization
            if key not in self._artifacts:
                program = generate(self.definition, target=target, compiler=self.compiler,
                                   constexprs=self.constexprs, tuning_config=self.tuning_config,
                                   options=self.options)
                self._artifacts[key] = program.materialize()
            return self._artifacts[key]

    def prepare(self, *arguments, outputs=None):
        """Bind a local call without executing a kernel or selecting a tuning winner."""
        if outputs is not None and not isinstance(outputs, tuple):
            raise TypeError("outputs must be a tuple in declared Out order")
        artifact = self._artifact(arguments, outputs)
        return artifact.prepare(*arguments, outputs=outputs)

    def run(self, *arguments):
        call = self.prepare(*arguments)
        call.launch()
        return call.result()

    def as_torch_op(self, name):
        """Register the uniquely prepared local specialization with PyTorch."""
        artifacts = self.artifacts
        if len(artifacts) != 1:
            raise ValueError("prepare one device specialization before registering its PyTorch operator")
        return artifacts[0].as_torch_op(name)

    def __call__(self, *arguments):
        parameters = tuple(p for p in self.definition.signature.parameters.values()
                           if not isinstance(p.annotation, ConstexprSpec))
        if len(parameters) != len(arguments):
            raise TypeError(f"expected {len(parameters)} runtime arguments, got {len(arguments)}")
        inputs, outputs = [], []
        for parameter, value in zip(parameters, arguments, strict=True):
            destination = outputs if (isinstance(parameter.annotation, ViewSpec)
                                      and parameter.annotation.kind is ViewKind.OUT) else inputs
            destination.append(value)
        self.prepare(*inputs, outputs=tuple(outputs)).launch()

    def close(self):
        with self._lock:
            for artifact in self._artifacts.values():
                close = getattr(artifact.runtime, "close", None)
                if close is not None:
                    close()
            self._artifacts.clear()
            self._targets.clear()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def jit(definition, *, backend, device=None, compiler=None, constexprs=None,
        tuning_config=None, options=None, workers=None, backend_options=None):
    """Create a local JIT callable; backend is explicit and hardware is discovered on use."""
    return JITKernel(definition, backend=backend, device=device, compiler=compiler,
                     constexprs=constexprs, tuning_config=tuning_config, options=options,
                     workers=workers, backend_options=backend_options)
