from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import math
import statistics
from threading import RLock

from .buffer import DeviceBuffer, DeviceView, runtime
from .compilation import compile_library
from .tuning import TrialState
from ..interface import ViewParameter
from ..invocation import ViewFacts, invocation_result
from ..native import NativePreparedRuntime
from ..native_artifact import load_native_library
from ..diagnostics import (CacheObservation, CandidateObservation, ConfigurationAssessment,
                           ObservedCall, RequirementEvaluation, bindings, observation)
from intent.compiler.toolchain import CompilationStageError


def _measure(owner, queue, submit, *, prepare=None, repetitions=10):
    if repetitions <= 0:
        raise ValueError("BANG C measurement repetitions must be positive")
    start, end = ctypes.c_void_p(), ctypes.c_void_p()
    owner.runtime.invoke("cnrtNotifierCreate", ctypes.byref(start))
    try:
        owner.runtime.invoke("cnrtNotifierCreate", ctypes.byref(end))
        samples = []
        for _ in range(repetitions):
            if prepare is not None:
                prepare()
            owner.runtime.invoke("cnrtPlaceNotifier", start, queue)
            submit()
            owner.runtime.invoke("cnrtPlaceNotifier", end, queue)
            owner.runtime.invoke("cnrtQueueSync", queue)
            microseconds = ctypes.c_float()
            owner.runtime.invoke("cnrtNotifierDuration", start, end, ctypes.byref(microseconds))
            samples.append(microseconds.value * 1e-3)
        return statistics.median(samples)
    finally:
        owner.runtime.invoke("cnrtQueueSync", queue)
        if end.value:
            owner.runtime.invoke("cnrtNotifierDestroy", end)
        owner.runtime.invoke("cnrtNotifierDestroy", start)


@dataclass
class NativeCall(ObservedCall):
    program: NativeProgram
    arguments: tuple[object, ...]
    native_arguments: tuple[object, ...]
    outputs: tuple[DeviceBuffer | DeviceView, ...]
    description: tuple
    key: tuple
    assessments: tuple[ConfigurationAssessment, ...]
    winner: int | None = None
    history: dict[int, CandidateObservation] = field(default_factory=dict)
    tuning_reused: bool | None = None

    @property
    def eligible(self):
        return tuple(index for index, assessed in enumerate(self.assessments)
                     if assessed.status == "eligible")

    def inspect_configurations(self):
        return self.assessments

    def _observed(self, stage):
        candidates = []
        for index, assessed in enumerate(self.assessments):
            configuration = self.program.configuration_descriptions[index]
            if index in self.history:
                candidates.append(self.history[index])
            elif assessed.status != "eligible":
                candidates.append(CandidateObservation(configuration, "rejected", "candidate_binding",
                    message="; ".join(item.describe() for item in assessed.requirements if not item.accepted)))
            elif index in self.program.compilation_errors:
                error = self.program.compilation_errors[index]
                candidates.append(CandidateObservation(configuration, "failed", error.stage,
                    type(error).__name__, str(error)))
            elif self.program.compilations[index] is not None:
                candidates.append(CandidateObservation(configuration, "compiled", "native_compilation"))
        selected = None if self.winner is None else self.program.configuration_descriptions[self.winner]
        caches = self.program.compilation_observations()
        if self.tuning_reused is not None:
            caches = (*caches, CacheObservation("tuning", "process", self.tuning_reused,
                "BANG C NativeProgram.winners", "selection"))
        return observation("bangc", self.program.target_facts, self.description, selected, (),
                           tuple(candidates), stage=stage, caches=caches)

    def _failure(self, error):
        details = self._observed("failed")
        self._record_observation(details)
        if isinstance(error, CompilationStageError):
            error.observation = details
            return error
        return CompilationStageError("provider_invocation", str(error), observation=details)

    def compile(self) -> None:
        """Compile applicable candidates without loading, selecting or launching."""
        try:
            self.program.compile(self.eligible)
        except CompilationStageError as error:
            raise self._failure(error)
        self._record_observation(self._observed("compiled"))

    def _validate(self) -> None:
        self.program.check_open()
        if any(isinstance(value, (DeviceBuffer, DeviceView)) and not value.pointer for value in self.arguments):
            raise ValueError("BANG C call refers to a closed device allocation")
        self.program.runtime.select(self.program.target.device)

    def choose(self, queue=None) -> int:
        try:
            with self.program._state_lock:
                self._validate()
                if self.winner is not None:
                    return self.winner
                ready = self.program.ensure_loaded(self.eligible)
                if self.key in self.program.winners:
                    self.winner = self.program.winners[self.key]
                    if self.winner not in ready:
                        raise RuntimeError("cached BANG C winner is not applicable to this invocation")
                    self.tuning_reused = True
                elif len(ready) == 1:
                    self.winner = ready[0]
                    self.program.winners[self.key] = self.winner
                else:
                    self.tuning_reused = False
                    queue = self.program.ensure_queue() if queue is None else queue
                    # Earlier kernels in an explicit host sequence may have
                    # produced these arguments on the same queue.
                    self.program.runtime.invoke("cnrtQueueSync", queue)
                    timings = {}
                    with TrialState(self.program, self.arguments) as trial:
                        for index in ready:
                            try:
                                trial.reset()
                                self.program.submit(index, queue, trial.native_arguments)
                                self.program.runtime.invoke("cnrtQueueSync", queue)
                                elapsed = _measure(self.program, queue,
                                    lambda: self.program.submit(index, queue, trial.native_arguments),
                                    prepare=trial.reset, repetitions=3)
                                if not math.isfinite(elapsed) or elapsed <= 0:
                                    raise RuntimeError("CNRT timing did not return a positive finite duration")
                                timings[index] = elapsed
                                self.history[index] = CandidateObservation(
                                    self.program.configuration_descriptions[index], "trial_completed",
                                    "provider_tuning", elapsed_ms=elapsed)
                            except Exception as error:
                                self.history[index] = CandidateObservation(
                                    self.program.configuration_descriptions[index], "failed",
                                    "provider_tuning", type(error).__name__, str(error))
                                raise CompilationStageError("provider_tuning", str(error),
                                    candidate=self.program.candidates[index].entry) from error
                    self.winner = min(timings, key=timings.__getitem__)
                    self.program.winners[self.key] = self.winner
                self._record_observation(self._observed("selected"))
                return self.winner
        except Exception as error:
            raise self._failure(error)

    def _submit(self, queue: ctypes.c_void_p) -> None:
        if self.winner is None:
            raise RuntimeError("BANG C submission requires a selected candidate")
        self.program.submit(self.winner, queue, self.native_arguments)

    def enqueue(self, queue: ctypes.c_void_p | None = None) -> None:
        try:
            self.choose(queue)
            self._validate()
            self._submit(self.program.ensure_queue() if queue is None else queue)
        except Exception as error:
            raise self._failure(error)
        self._record_execution("submitted")

    def launch(self) -> None:
        self.enqueue()
        try:
            self.program.runtime.invoke("cnrtQueueSync", self.program.queue)
        except Exception as error:
            raise self._failure(error)
        self._record_execution("launched", replay=False)

    def result(self):
        return invocation_result(self.outputs)

    def benchmark(self) -> float:
        return benchmark_calls((self,))


def _queue_owner(calls: tuple[NativeCall, ...]):
    if not calls:
        raise ValueError("a BANG C launch sequence must contain at least one call")
    owner = calls[0].program
    if any(call.program.runtime is not owner.runtime or call.program.target.device != owner.target.device for call in calls):
        raise ValueError("a BANG C launch sequence must use one runtime and device")
    for call in calls:
        call.program.check_open()
    owner.ensure_queue()
    return owner


def launch_calls(calls: tuple[NativeCall, ...]) -> None:
    """Execute an explicitly supplied host sequence on one CNRT queue."""
    owner = _queue_owner(calls)
    for call in calls:
        call.enqueue(owner.queue)
    owner.runtime.invoke("cnrtQueueSync", owner.queue)


def benchmark_calls(calls: tuple[NativeCall, ...], *, prepare=None, repetitions: int = 10) -> float:
    owner = _queue_owner(calls)
    if any(call.winner is None for call in calls):
        if prepare is not None:
            prepare()
        # Select at each actual call point, after its input-producing kernel.
        launch_calls(calls)

    def reset():
        if prepare is not None:
            prepare()
        for call in calls:
            call._validate()

    def submit():
        for call in calls:
            call._submit(owner.queue)

    elapsed = _measure(owner, owner.queue, submit, prepare=reset, repetitions=repetitions)
    for call in calls:
        call._record_execution("benchmarked")
    return elapsed


class NativeProgram(NativePreparedRuntime):
    def __init__(self, source: str, contract, target) -> None:
        from intent.targets.specification import require_matching_target

        require_matching_target(contract.target, target.compilation)
        self.target = target
        self.source = source
        self._state_lock = RLock()
        self._closed = False
        self._runtime = None
        self.queue = ctypes.c_void_p()
        self.target_facts = bindings(contract.metadata["target"])
        self.facts = contract.facts
        self.candidates = self.facts.candidates
        self.configuration_descriptions = tuple(bindings(candidate.metadata()) for candidate in self.candidates)
        self.compilations = [None] * len(self.candidates)
        self.compilation_errors = {}
        self.libraries = [None] * len(self.candidates)
        self.functions = [None] * len(self.candidates)
        self.winners = {}
        self._abi = abi = contract.abi
        self._argument_types = (ctypes.c_void_p, *abi.argument_types())
        self.interface = abi.interface
        self.requirements = requirements = abi.requirements
        converters = {"bool": bool, "i8": int, "i16": int, "i32": int, "i64": int,
                      "f32": float, "f64": float}
        scalar_converters = {slot.parameter.position: converters[slot.carrier]
                             for slot in abi.slots if slot.role == "scalar"}
        self._binders = abi.binders(
            observe_view=type(self)._view, allocate_output=type(self)._allocate_output,
            check_alias=requirements.check_pair,
            scalar_argument=lambda parameter, value: scalar_converters[parameter.position](value),
            view_key=lambda facts, group: (facts.shape, facts.strides, facts.offset, facts.dtype, group),
            scalar_key=lambda parameter, value: (parameter.dtype.name, value),
            check_view_geometry=lambda owner, parameter, facts: requirements.check_geometry(parameter, facts),
            check_view_storage=lambda owner, parameter, facts: requirements.check_storage(parameter, facts),
        )
        parameters_by_name = {parameter.name: parameter for parameter in self.interface.parameters}
        for name, shape in target.compilation.shapes:
            parameter = parameters_by_name.get(name)
            if not isinstance(parameter, ViewParameter) or len(shape) != len(parameter.shape):
                raise ValueError("BANG C artifact and target disagree on bound parameter shapes")
            if any(extent >= 0 and extent != declared for extent, declared in zip(shape, parameter.shape)):
                raise ValueError("BANG C artifact and target disagree on bound extents")

    def check_open(self) -> None:
        if self._closed:
            raise ValueError("BANG C program is closed")

    @property
    def runtime(self):
        with self._state_lock:
            self.check_open()
            if self._runtime is None:
                self._runtime = runtime(self.target.neuware)
            return self._runtime

    def compile(self, indices=None):
        with self._state_lock:
            self.check_open()
            indices = tuple(range(len(self.candidates))) if indices is None else tuple(indices)
            for index in indices:
                if self.compilations[index] is not None or index in self.compilation_errors:
                    continue
                try:
                    self.compilations[index] = compile_library(self.candidates[index].source, self.target)
                except CompilationStageError as error:
                    error.candidate = self.candidates[index].entry
                    if error.stage != "native_compilation":
                        raise
                    self.compilation_errors[index] = error
            ready = tuple(index for index in indices if self.compilations[index] is not None)
            if not ready:
                reasons = "\n".join(f"{self.candidates[index].entry}: {self.compilation_errors[index]}"
                                    for index in indices if index in self.compilation_errors)
                raise CompilationStageError("native_compilation", "No BANG C candidate compiled successfully\n" + reasons)
            return tuple(self.compilations[index] for index in ready)

    def ensure_loaded(self, indices):
        with self._state_lock:
            self.compile(indices)
            ready = []
            for index in indices:
                compilation = self.compilations[index]
                if compilation is None:
                    continue
                if self.functions[index] is None:
                    library = load_native_library(compilation)
                    try:
                        function = library.bind(self.candidates[index].entry, self._argument_types, ctypes.c_int)
                    except Exception:
                        library.close()
                        raise
                    self.libraries[index], self.functions[index] = library, function
                ready.append(index)
            return tuple(ready)

    def ensure_queue(self) -> ctypes.c_void_p:
        with self._state_lock:
            self.check_open()
            if not self.queue.value:
                self.runtime.select(self.target.device)
                self.runtime.invoke("cnrtQueueCreate", ctypes.byref(self.queue))
            return self.queue

    def submit(self, index, queue, arguments):
        status = self.functions[index](queue, *arguments)
        if status:
            raise RuntimeError(f"BANG C kernel submission failed with CNRT status {status}")

    def compilation_observations(self):
        return tuple(CacheObservation("native_compilation", "persistent_artifact", artifact.cache_hit,
                         "BANG C native artifact", "native_compilation", artifact.cache_reason,
                         self.candidates[index].entry)
                     for index, artifact in enumerate(self.compilations) if artifact is not None)

    def prepare(self, arguments: tuple[object, ...], *, explicit_outputs: bool = False) -> NativeCall:
        self.check_open()
        bound = self._binders[bool(explicit_outputs)](self, arguments)
        dimensions = {identity: view.shape[axis]
                      for parameter, view in zip(self.interface.parameters, bound.views)
                      if isinstance(parameter, ViewParameter)
                      for axis, identity in enumerate(parameter.dimensions) if identity > 0}
        assessments = tuple(ConfigurationAssessment(description, tuple(
            RequirementEvaluation("legality", "fragment_elements", "less_equal",
                f"DSA dimension {identity} must fit this candidate's full tile",
                "satisfied" if dimensions[identity] <= candidate.tile else "violated",
                dimensions[identity], candidate.tile)
            for identity in candidate.full_extent_dimensions))
            for candidate, description in zip(self.candidates, self.configuration_descriptions, strict=True))
        if not any(assessed.status == "eligible" for assessed in assessments):
            raise CompilationStageError("candidate_binding", "No BANG C candidate covers this invocation's full extents")
        return NativeCall(self, bound.arguments, bound.native_arguments, bound.outputs,
                          self.describe_arguments(bound, f"mlu:{self.target.device}"), bound.key, assessments)

    def _view(self, parameter: ViewParameter, value) -> ViewFacts:
        if not isinstance(value, (DeviceBuffer, DeviceView)) or not value.pointer:
            raise TypeError(f"{parameter.name} requires a live MLU buffer or view")
        if value.device != self.target.device or value.runtime is not self.runtime:
            raise ValueError("BANG C arguments must belong to the artifact's MLU device and runtime")
        if value.dtype != parameter.dtype.name or len(value.shape) != len(parameter.shape):
            raise ValueError(f"{parameter.name} has incompatible shape or dtype")
        lower, upper = value.byte_bounds
        owner = value.owner if isinstance(value, DeviceView) else value
        return ViewFacts(value.shape, value.strides, value.pointer, value.allocation_pointer,
                         owner.pointer + owner.nbytes,
                         value.pointer - value.allocation_pointer, value.dtype, lower, upper, value.allocation_pointer)

    def _allocate_output(self, parameter: ViewParameter, shape: tuple[int, ...]) -> tuple[DeviceBuffer, ViewFacts]:
        value = DeviceBuffer(shape, parameter.dtype.name, device=self.target.device, neuware=self.target.neuware)
        lower, upper = value.byte_bounds
        return value, ViewFacts(value.shape, value.strides, value.pointer, value.allocation_pointer,
                                value.pointer + value.nbytes,
                                0, value.dtype, lower, upper, value.allocation_pointer)

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            if self.queue.value:
                self.runtime.invoke("cnrtSetDevice", self.target.device)
                self.runtime.invoke("cnrtQueueSync", self.queue)
                self.runtime.invoke("cnrtQueueDestroy", self.queue)
                self.queue = ctypes.c_void_p()
            for index, library in enumerate(self.libraries):
                if library is not None:
                    library.close()
                    self.libraries[index] = self.functions[index] = None
            self._closed = True

    def __del__(self):
        queue = getattr(self, "queue", None)
        if queue and queue.value:
            self._runtime.library.cnrtSetDevice(self.target.device)
            self._runtime.library.cnrtQueueSync(queue)
            self._runtime.library.cnrtQueueDestroy(queue)
        for library in getattr(self, "libraries", ()):
            if library is not None:
                library.close()
