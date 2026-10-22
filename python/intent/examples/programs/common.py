"""Author host orchestration using the product local JIT."""
from dataclasses import dataclass, field
import intent
from intent.language.annotations import ConstexprSpec, ViewKind, ViewSpec
from intent.runtime.invocation import ViewFacts, build_invocation_binders
from intent.targets.local import select_device, resolve_local
from .inputs import Array
from .native import HostBuffers, element_strides, empty_like, host_result, is_tensor

def _abstract_view(parameter, value):
    if isinstance(value, Array):
        if value.dtype != parameter.dtype.name:
            raise TypeError(f"{parameter.name} requires {parameter.dtype.name}, got {value.dtype}")
    elif is_tensor(value):
        from intent.integrations.torch import torch_dtype
        if value.dtype != torch_dtype(parameter.dtype):
            raise TypeError(f"{parameter.name} has an incompatible storage dtype")
    else:
        raise TypeError("source preparation requires host arrays or tensor geometry")
    if len(value.shape) != len(parameter.shape):
        raise ValueError(f"{parameter.name} has an incompatible rank")
    return ViewFacts(tuple(value.shape), element_strides(value), None, None, None,
                     None, parameter.dtype.name, None, None, None)

def _abstract_output(parameter, shape):
    value = Array.metadata(shape, parameter.dtype.name)
    return value, _abstract_view(parameter, value)

@dataclass
class _SourceCall:
    bound: object
    def result(self):
        return self.bound.result()
    def compile(self):
        raise RuntimeError("source inspection has no native call")
    def launch(self):
        raise RuntimeError("source inspection does not execute")

class _ExampleKernel:
    def __init__(self, context, definition, constexprs):
        self.context, self.definition, self.constexprs = context, definition, constexprs
        self.kernel = intent.jit(definition, backend=context.backend, device=context.device,
            compiler=context.compiler, constexprs=constexprs, workers=context.workers,
            backend_options=context.backend_options)
        context.kernels.append(self.kernel)

    def empty_like(self, reference, shape, *, dtype):
        if self.context.buffers is not None:
            return self.context.buffers.workspace(reference, shape, dtype=dtype)
        return empty_like(reference, shape, dtype=dtype)

    def prepare(self, *arguments, outputs=None):
        if self.context._closed:
            raise RuntimeError("example context is closed")
        if self.context.stage == "source":
            target = self.context.source_target()
            if self.context.backend == "bangc":
                from intent.targets.bangc import specialize_call
                target, _ = specialize_call(target, self.definition, arguments, outputs)
            program = intent.generate(self.definition, target=target,
                                      compiler=self.context.compiler, constexprs=self.constexprs)
            self.context.artifacts.append(program)
            binders = build_invocation_binders(program.interface, abstract=True,
                observe_view=lambda _, p, value: _abstract_view(p, value),
                allocate_output=lambda _, p, shape: _abstract_output(p, shape))
            explicit = outputs is not None
            supplied = program.interface.explicit_arguments(arguments, outputs) if explicit else arguments
            return _SourceCall(binders[explicit](None, supplied))
        parameters = [p.annotation for p in self.definition.signature.parameters.values()
                      if not isinstance(p.annotation, ConstexprSpec)
                      and not (isinstance(p.annotation, ViewSpec) and p.annotation.kind is ViewKind.OUT)]
        if len(parameters) != len(arguments):
            raise TypeError("example inputs disagree with the author declaration")
        arguments = tuple(self.context.buffers.input(value, element=p.dtype.name,
                            writable=p.kind is ViewKind.INOUT, preserve=p.kind is ViewKind.INOUT)
                          if isinstance(p, ViewSpec) else value
                          for p, value in zip(parameters, arguments, strict=True))
        call = self.kernel.prepare(*arguments, outputs=outputs)
        for artifact in self.kernel.artifacts:
            if all(artifact is not old for old in self.context.artifacts):
                self.context.artifacts.append(artifact)
        self.context.buffers.retain_outputs(call.result())
        return call

    def run(self, *arguments):
        call = self.prepare(*arguments)
        call.launch()
        return call.result()

    def __call__(self, *arguments):
        parameters = [p.annotation for p in self.definition.signature.parameters.values()
                      if not isinstance(p.annotation, ConstexprSpec)]
        if len(parameters) != len(arguments):
            raise TypeError("explicit example arguments disagree with the declaration")
        inputs, outputs = [], []
        for p, value in zip(parameters, arguments, strict=True):
            (outputs if isinstance(p, ViewSpec) and p.kind is ViewKind.OUT else inputs).append(value)
        self.prepare(*inputs, outputs=tuple(outputs)).launch()

@dataclass
class ExampleInvocation:
    program: object
    arguments: tuple
    prepared: object
    stage: str
    _resetters: tuple = field(default=(), repr=False)
    def reset_inputs(self):
        for restore in self._resetters:
            restore()

@dataclass
class ExampleContext:
    backend: str
    device: str
    compiler: str | None = None
    prepared: bool = False
    stage: str = "run"
    workers: int | None = None
    backend_options: dict = field(default_factory=dict)
    artifacts: list = field(default_factory=list, init=False)
    kernels: list = field(default_factory=list, init=False)
    invocations: list = field(default_factory=list, init=False)
    buffers: object = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _source_target: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.stage not in {"source", "native", "run"}:
            raise ValueError("example stage must be source, native, or run")
        if self.stage != "source":
            self.buffers = HostBuffers(self.backend, self.device, self.backend_options)

    def source_target(self):
        if self._source_target is None:
            self._source_target = resolve_local(self.backend, select_device(self.backend, (), self.device),
                workers=self.workers, backend_options=self.backend_options)
        return self._source_target

    def compile(self, definition, *, constexprs=None):
        if self._closed:
            raise RuntimeError("example context is closed")
        return _ExampleKernel(self, definition, constexprs)

    def call(self, artifact, *arguments):
        if self._closed:
            raise RuntimeError("example context is closed")
        if self.buffers is not None:
            self.buffers.begin()
        invocation = artifact.prepare(*arguments)
        resetters = tuple(self.buffers.resetters.values()) if self.buffers is not None else ()
        self.invocations.append(ExampleInvocation(artifact, arguments, invocation, self.stage, resetters))
        if self.stage == "native" or (self.stage == "run" and self.prepared):
            invocation.compile()
        if self.stage == "run":
            invocation.launch()
        result = invocation.result()
        if self.stage != "run":
            return host_result(result, contents=False)
        self.buffers.finish()
        return host_result(result, contents=True)

    def close(self):
        if self._closed:
            return
        for kernel in self.kernels:
            kernel.close()
        if self.buffers is not None:
            self.buffers.close()
        self.invocations.clear()
        self._closed = True

def describe_outputs(value, label="result"):
    if isinstance(value, Array) or is_tensor(value):
        print(f"{label}: shape={value.shape}, dtype={value.dtype}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            describe_outputs(item, f"{label}[{index}]")
    elif isinstance(value, dict):
        for name, item in value.items():
            describe_outputs(item, f"{label}.{name}")
    else:
        print(f"{label}: {value}")
