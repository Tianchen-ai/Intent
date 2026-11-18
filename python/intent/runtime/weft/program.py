from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import math
import statistics
from threading import RLock
from time import perf_counter

from .buffer import Buffer
from .compilation import compile_program
from .binding import CallState, NativeEntries
from ..interface import ViewParameter
from ..invocation import ViewFacts, invocation_result
from ..native import NativePreparedRuntime
from ..native_artifact import load_native_library
from ..diagnostics import CacheObservation, CandidateObservation, ObservedCall, bindings, observation
from intent.compiler.toolchain import CompilationStageError


_winners: dict[tuple, int] = {}


def _measure(measure, arguments: tuple, repetitions: int = 1) -> float:
    elapsed = measure(*arguments, repetitions)
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("native benchmark returned no valid positive duration")
    return elapsed


@dataclass
class NativeCall(ObservedCall):
    program: NativeProgram
    arguments: tuple
    native_arguments: tuple
    outputs: tuple[Buffer, ...]
    key: tuple
    description: tuple
    trial_storage: tuple[memoryview, ...]
    state: CallState
    winner: int | None = None

    @contextmanager
    def _native_stage(self, stage: str, *, candidate=None, observed=()):
        try:
            yield
        except Exception as cause:
            error = cause if isinstance(cause, CompilationStageError) else CompilationStageError(
                stage, str(cause), cache_directory=self.program.directory,
                candidate=None if candidate is None else candidate.entry)
            failed_entry = error.candidate if error.candidate is not None else (
                candidate.entry if candidate is not None else None)
            configuration = next((description for entry, description in zip(
                self.program.candidates, self.program.configuration_descriptions, strict=True)
                if entry.entry == failed_entry), None)
            failed = CandidateObservation(configuration, "failed", error.stage,
                                          type(cause).__name__, str(cause))
            if self.observation is not None and not observed:
                details = replace(self.observation, stage="failed",
                                  candidates=(*self.observation.candidates, failed))
            else:
                details = observation("weft", self.program.target, self.description, configuration, (),
                                      (*observed, failed), stage="failed", caches=self.program.compilation_cache)
            self._record_observation(details)
            error.observation = details
            if error is cause:
                raise
            raise error from cause

    def compile(self) -> None:
        """Compile or validate the native portfolio without loading or executing it."""
        with self._native_stage("provider_native_compilation"):
            self.program.compile()
        self._record_observation(observation(
            "weft", self.program.target, self.description, None, (),
            tuple(CandidateObservation(configuration, "compiled", "provider_native_compilation")
                  for configuration in self.program.configuration_descriptions),
            stage="compiled", caches=self.program.compilation_cache,
            history_unavailable="Native compilation does not load or time a native candidate"))

    def _prepare_execution(self) -> None:
        with self._native_stage("provider_invocation"):
            self.program.check_execution()
            self.state.check()
            self.state.entries = self.program._entries.entries

    def inspect_configurations(self):
        return self.program.inspect_configurations()

    @contextmanager
    def _trial_state(self):
        snapshots = tuple((storage, bytes(storage)) for storage in self.trial_storage)

        def restore() -> None:
            for storage, snapshot in snapshots:
                storage[:] = snapshot

        try:
            yield restore
        finally:
            restore()

    def choose(self, prepare=None) -> int:
        self._prepare_execution()
        self.key = (self.program.identity, self.key[1])
        if self.winner is not None:
            return self.winner
        observed = []
        single = len(self.program.measurements) == 1
        reused = not single and self.key in _winners
        if single:
            self.winner = 0
        elif not reused:
            timings = []
            with self._trial_state() as restore:
                for candidate, measure, configuration in zip(self.program.candidates, self.program.measurements,
                                                             self.program.configuration_descriptions, strict=True):
                    samples = []
                    with self._native_stage("provider_tuning", candidate=candidate, observed=observed):
                        deadline = perf_counter() + 0.100
                        for _ in range(3):
                            restore()
                            if prepare is not None:
                                prepare()
                            samples.append(_measure(measure, self.native_arguments))
                            self.state.check()
                            if perf_counter() >= deadline:
                                break
                    elapsed = statistics.median(samples)
                    timings.append(elapsed)
                    observed.append(CandidateObservation(configuration, "trial_completed", "provider_tuning",
                                                         elapsed_ms=elapsed))
                _winners[self.key] = min(range(len(timings)), key=timings.__getitem__)
        if not single:
            self.winner = _winners[self.key]
        self._record_observation(observation("weft", self.program.target, self.description,
            self.program.configuration_descriptions[self.winner], (), observed, stage="selected",
            history_unavailable="Only one native candidate; tuning was not required" if single else
                                "The process-local winner was reused; no candidates were retried" if reused else None,
            caches=(*self.program.compilation_cache,
                    CacheObservation("tuning", "process", None if single else reused, "Weft runtime winner cache",
                                     "selection",
                                     "Only one native candidate" if single else None))))
        return self.winner

    def launch(self) -> None:
        selected = self.choose()
        with self._native_stage("provider_invocation", candidate=self.program.candidates[selected]):
            self.program.functions[selected](*self.native_arguments)
            self.state.check()
        self._record_execution("launched")

    def benchmark(self, *, prepare=None, samples: int = 10) -> float:
        selected = self.choose(prepare)
        measure = self.program.measurements[selected]
        values = []
        with self._native_stage("provider_invocation", candidate=self.program.candidates[selected]):
            with self._trial_state() as restore:
                for _ in range(samples):
                    restore()
                    if prepare is not None:
                        prepare()
                    values.append(_measure(measure, self.native_arguments))
                    self.state.check()
            elapsed = statistics.median(values)
        self._record_execution("benchmarked")
        return elapsed

    def result(self):
        return invocation_result(self.outputs)

    def native_call(self):
        """Select a configuration and export a checked local CPU invocation."""
        selected = self.choose()
        return self.program.functions[selected].bind_call(
            self.native_arguments, owner=self, validate=self._prepare_execution,
            complete=self.state.check)


class NativeProgram(NativePreparedRuntime):
    def __init__(self, program, target, *, artifact=None) -> None:
        self.program, self.resolved_target = program, target
        self.artifact = artifact
        self.directory = program.cache_directory
        contract = program._contract
        self.facts = contract.facts
        self.target = bindings({"family": "cpu", **{name: value for name, value in asdict(contract.target).items()
                                                  if name != "provider"}})
        self.compilation_cache = ()
        abi = contract.abi
        self.interface = abi.interface
        self.trial_regions = abi.trial_regions()
        self.requirements = requirements = abi.requirements
        self._binders = abi.binders(
            observe_view=type(self)._view, allocate_output=type(self)._allocate_output,
            check_alias=requirements.check_pair,
            check_view_geometry=lambda owner, parameter, facts: requirements.check_geometry(parameter, facts),
            check_view_storage=lambda owner, parameter, facts: requirements.check_storage(parameter, facts),
            view_key=lambda facts, group: (facts.shape, facts.strides, facts.dtype, facts.offset, group),
            scalar_key=lambda parameter, value: (parameter.dtype.name, value),
        )
        self.candidates = self.facts.candidates
        self.configuration_descriptions = tuple(bindings(candidate.metadata()) for candidate in self.candidates)
        self.identity = artifact.native.identity if artifact is not None else None
        self._argument_types = [ctypes.c_void_p, *abi.argument_types()]
        self._library = None
        self._entries = None
        self._closed = False
        self._load_lock = RLock()
        self.functions, self.measurements = [], []

    def compile(self) -> None:
        with self._load_lock:
            if self._closed:
                raise CompilationStageError("provider_native_compilation", "native artifact is closed")
            if self.artifact is None:
                self.artifact = compile_program(self.program, self.resolved_target)
                self.directory = self.artifact.native.directory
                self.identity = self.artifact.native.identity
                self.compilation_cache = (CacheObservation("native_compilation", "provider",
                    None, "Triton–Weft JIT cache", "native_compilation",
                    "Task compilation uses the provider cache; hit status is not exposed"),
                    CacheObservation("native_compilation", "disk",
                    self.artifact.native.cache_hit, str(self.directory), "native_compilation",
                    self.artifact.native.cache_reason),)
            self.artifact.native.validate()

    def ensure_loaded(self):
        with self._load_lock:
            self.compile()
            if self._closed:
                raise CompilationStageError("native_loading", "native artifact is closed")
            if self._library is not None:
                return self._library
            self.resolved_target.validate_execution()
            entries = NativeEntries(self.artifact.tasks)
            try:
                library = load_native_library(self.artifact.native)
            except Exception:
                entries.close()
                raise
            try:
                functions = [library.bind(candidate.entry + "_invoke", self._argument_types, None)
                             for candidate in self.candidates]
                measurements = [library.bind(candidate.entry + "_benchmark",
                                              [*self._argument_types, ctypes.c_int64], ctypes.c_double)
                                for candidate in self.candidates]
            except Exception:
                entries.close()
                library.close()
                raise
            self._entries, self._library = entries, library
            self.functions, self.measurements = functions, measurements
            return library

    @property
    def library(self):
        return self.ensure_loaded()

    def export(self, directory):
        self.compile()
        return self.artifact.export(directory)

    def check_execution(self) -> None:
        self.ensure_loaded()
        self.resolved_target.validate_execution()

    def _view(self, parameter: ViewParameter, value) -> ViewFacts:
        if not isinstance(value, Buffer) or value.dtype != parameter.dtype.name:
            raise TypeError(f"{parameter.name} requires a native {parameter.dtype} Buffer")
        if len(value.shape) != len(parameter.shape):
            raise ValueError("native view rank disagrees with the compiler ABI")
        return ViewFacts(value.shape, value.strides, value.pointer, value.allocation, value.allocation_end,
                         value.offset, value.dtype, value.pointer,
                         value.pointer + value.storage_span.nbytes, value.allocation_identity)

    def _allocate_output(self, parameter: ViewParameter, shape: tuple[int, ...]) -> tuple[Buffer, ViewFacts]:
        value = Buffer.empty(shape, parameter.dtype.name)
        return value, self._view(parameter, value)

    def prepare(self, arguments: tuple, *, explicit_outputs: bool = False) -> NativeCall:
        if self._closed:
            raise CompilationStageError("native_loading", "native artifact is closed",
                                        cache_directory=self.directory)
        bound = self._binders[bool(explicit_outputs)](self, arguments)
        trial_storage = []
        for region in self.trial_regions:
            position = region.pointer.parameter.position
            storage = bound.arguments[position].storage
            if storage.nbytes != region.byte_count(bound.views[position].shape):
                raise ValueError("native trial byte range disagrees with the bound buffer storage")
            trial_storage.append(storage)
        state = CallState()
        return NativeCall(self, bound.arguments,
                          (ctypes.c_void_p(ctypes.addressof(state)), *bound.native_arguments), bound.outputs,
                          (self.identity, bound.key), self.describe_arguments(bound, "cpu"),
                          tuple(trial_storage), state)

    def close(self) -> None:
        with self._load_lock:
            self._closed = True
            self.functions.clear()
            self.measurements.clear()
            if self._entries is not None:
                self._entries.close()
                self._entries = None
            if self._library is not None:
                self._library.close()
                self._library = None
