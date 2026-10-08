"""Complete example calls through public generation and invocation interfaces."""

from dataclasses import dataclass, field, replace

import intent

from intent.language.annotations import ConstexprSpec, ViewKind, ViewSpec
from intent.runtime.interface import ViewParameter
from intent.runtime.invocation import ViewFacts, build_invocation_binders
from intent.targets.bangc import BangCTarget
from intent.targets.weft import WeftTarget

from .inputs import Array
from .native import HostBuffers, element_strides, empty_like, host_result, is_tensor


def _call_target(target, definition, arguments, *, explicit_outputs):
    if not isinstance(target, BangCTarget):
        return target
    parameters = tuple(parameter for parameter in definition.signature.parameters.values()
                       if not isinstance(parameter.annotation, ConstexprSpec)
                       and (explicit_outputs or not isinstance(parameter.annotation, ViewSpec)
                            or parameter.annotation.kind is not ViewKind.OUT))
    if len(parameters) != len(arguments):
        raise TypeError(f"expected {len(parameters)} runtime arguments, got {len(arguments)}")
    shapes, strides = dict(target.shapes or {}), dict(target.strides or {})
    for parameter, argument in zip(parameters, arguments, strict=True):
        if not isinstance(parameter.annotation, ViewSpec):
            continue
        shape = tuple(argument.shape)
        stride = element_strides(argument)
        for mapping, actual in ((shapes, shape), (strides, stride)):
            if parameter.name in mapping and tuple(mapping[parameter.name]) != actual:
                raise ValueError(f"{parameter.name} disagrees with the explicitly selected BANG C geometry")
            mapping[parameter.name] = actual
    return replace(target, shapes=shapes, strides=strides)


def _abstract_view(parameter, value):
    if isinstance(value, Array):
        if value.dtype != parameter.dtype.name:
            raise TypeError(f"{parameter.name} requires {parameter.dtype.name}, got {value.dtype}")
    elif is_tensor(value):
        from intent.runtime.torch import torch_dtype
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
        raise RuntimeError("a source-only call has no native runtime; select stage='native'")

    def launch(self):
        raise RuntimeError("a source-only call cannot execute; select stage='run'")


class _ExampleKernel:
    def __init__(self, context, definition, constexprs):
        self.context, self.definition, self.constexprs = context, definition, constexprs
        self.target = context.target
        self._programs = {}

    def empty_like(self, reference, shape, *, dtype):
        if self.context.buffers is not None:
            return self.context.buffers.workspace(reference, shape, dtype=dtype)
        return empty_like(reference, shape, dtype=dtype)

    def _program(self, arguments, *, explicit_outputs=False):
        if self.context._closed:
            raise RuntimeError("example context is closed")
        target = _call_target(self.target, self.definition, arguments,
                              explicit_outputs=explicit_outputs)
        key = ((tuple(sorted((name, tuple(shape)) for name, shape in target.shapes.items())),
                tuple(sorted((name, tuple(strides)) for name, strides in target.strides.items())))
               if isinstance(target, BangCTarget) else ())
        if key not in self._programs:
            assets = self.context.assets
            if assets is not None and assets.mode == "load":
                program, artifact = assets.resolve(self.definition, self.constexprs,
                                                  target, self.context.stage)
            else:
                program = intent.generate(self.definition, target=target,
                                          compiler=self.context.compiler, constexprs=self.constexprs)
                if assets is not None:
                    program, artifact = assets.store(self.definition, self.constexprs,
                                                     target, program, stage=self.context.stage)
                else:
                    artifact = None if self.context.stage == "source" else program.materialize()
            self._programs[key] = program, artifact
            self.context.artifacts.append(program if artifact is None else artifact)
        return self._programs[key]

    def _prepare(self, arguments, *, explicit_outputs=False):
        program, artifact = self._program(arguments, explicit_outputs=explicit_outputs)
        interface = program.interface
        if self.context.stage == "source":
            binders = build_invocation_binders(interface, abstract=True,
                observe_view=lambda _, parameter, value: _abstract_view(parameter, value),
                allocate_output=lambda _, parameter, shape: _abstract_output(parameter, shape))
            return _SourceCall(binders[explicit_outputs](None, arguments))
        supplied = interface.parameters if explicit_outputs else interface.inputs
        if len(arguments) != len(supplied):
            raise TypeError(f"expected {len(supplied)} runtime arguments, got {len(arguments)}")
        if self.context.buffers is not None:
            arguments = tuple(self.context.buffers.input(value, element=parameter.dtype.name,
                                                        writable=parameter.writable,
                                                        preserve=parameter.access is ViewKind.INOUT)
                              if isinstance(parameter, ViewParameter) else value
                              for parameter, value in zip(supplied, arguments, strict=True))
        if not explicit_outputs:
            call = artifact.prepare(*arguments)
            if self.context.buffers is not None:
                self.context.buffers.retain_outputs(call.result())
            return call
        inputs, outputs = [], []
        for parameter, value in zip(interface.parameters, arguments, strict=True):
            (outputs if isinstance(parameter, ViewParameter) and parameter.output else inputs).append(value)
        return artifact.prepare(*inputs, outputs=tuple(outputs))

    def prepare(self, *arguments):
        return self._prepare(arguments)

    def run(self, *arguments):
        call = self.prepare(*arguments)
        call.launch()
        return call.result()

    def __call__(self, *arguments):
        call = self._prepare(arguments, explicit_outputs=True)
        call.launch()


@dataclass
class ExampleInvocation:
    program: object
    arguments: tuple
    prepared: object
    stage: str
    _resetters: tuple = field(default=(), repr=False)

    def reset_inputs(self):
        """Restore only supplied InOut allocations before another full launch."""
        for restore in self._resetters:
            restore()


@dataclass
class ExampleContext:
    target: object
    device: object
    compiler: str | None = None
    prepared: bool = False
    stage: str = "run"
    assets: object = None
    artifacts: list = field(default_factory=list)
    invocations: list = field(default_factory=list, init=False)
    buffers: object = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.stage not in {"source", "native", "run"}:
            raise ValueError("example stage must be source, native, or run")
        if self.stage != "source":
            self.buffers = HostBuffers(self.target, self.device)

    def compile(self, definition, *, constexprs=None):
        if self._closed:
            raise RuntimeError("example context is closed")
        return _ExampleKernel(self, definition, constexprs)

    def call(self, artifact, *arguments):
        # This scope includes staging and host-visible readback. Native handles
        # stay resident between the kernels of a composed author call.
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
        if self.buffers is not None:
            self.buffers.finish()
            return host_result(result, contents=True)
        return result

    def close(self):
        if self._closed:
            return
        for artifact in self.artifacts:
            runtime = getattr(artifact, "runtime", None)
            close = getattr(runtime, "close", None)
            if close is not None:
                close()
        if self.buffers is not None:
            self.buffers.close()
        self.invocations.clear()
        self._closed = True


def describe_outputs(value, label="result"):
    if isinstance(value, Array):
        state = "host data" if value.data is not None else "shape only; not executed"
        print(f"{label}: shape={value.shape}, dtype={value.dtype}, {state}")
    elif is_tensor(value):
        print(f"{label}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            describe_outputs(item, f"{label}[{index}]")
    elif isinstance(value, dict):
        for name, item in value.items():
            describe_outputs(item, f"{label}.{name}")
    else:
        print(f"{label}: {value}")
