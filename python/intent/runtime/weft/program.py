from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import math
import os
from pathlib import Path
import resource
import statistics
from threading import RLock

from .buffer import Buffer
from .compilation import WeftArtifact
from .target import TargetProfile
from ..interface import ViewParameter
from ..invocation import ViewFacts, invocation_result
from ..native import NativePreparedRuntime
from ..native_artifact import load_native_library
from ..diagnostics import CacheObservation, CandidateObservation, ObservedCall, bindings, observation
from intent.compiler.toolchain import CompilationStageError


class _ExecutionContract:
    """Artifact-local hardware facts, with calling-thread state checked on use."""

    def __init__(self, profile: TargetProfile, kernels, native=None) -> None:
        from weft.runtime.driver import Toolchain
        from weft.runtime.target import NativeTarget, Target
        self.profile = profile
        self.native = native or NativeTarget.discover(Toolchain(compiler=os.environ.get("INTENT_WEFT_COMPILER")))
        target = Target(profile.march, profile.abi, profile.vlen_bits)
        self.extensions = tuple(sorted({name for kernel in kernels for name in kernel["used_extensions"]}))
        for kernel in kernels:
            self.native.require(target, kernel["used_extensions"], kernel["isa_extensions"])
        self.vlen_probe = None

    def bind_library(self, library) -> None:
        self.vlen_probe = library.bind("intent_weft_vlen_bits", [], ctypes.c_int64)

    def check(self) -> None:
        stack, _ = resource.getrlimit(resource.RLIMIT_STACK)
        if stack != resource.RLIM_INFINITY and self.profile.private_stack_bytes >= stack:
            raise ValueError("artifact private stack budget exceeds the native thread's stack limit")
        cpus = os.sched_getaffinity(0)
        if not cpus.issubset(self.profile.cpus):
            raise ValueError("current CPU affinity exceeds the artifact execution set")
        self.native.check_execution(self.extensions)
        # The calling thread may have migrated, so VLEN remains a live check.
        if self.vlen_probe is not None and self.vlen_probe() != self.profile.vlen_bits:
            raise ValueError("native VLEN disagrees with the artifact")


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
                        for _ in range(3):
                            restore()
                            if prepare is not None:
                                prepare()
                            samples.append(_measure(measure, self.native_arguments))
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
            elapsed = statistics.median(values)
        self._record_execution("benchmarked")
        return elapsed

    def result(self):
        return invocation_result(self.outputs)

    def native_call(self):
        """Select a configuration and export a checked local CPU invocation."""
        selected = self.choose()
        return self.program.functions[selected].bind_call(
            self.native_arguments, owner=self, validate=self.program.check_execution)


class NativeProgram(NativePreparedRuntime):
    def __init__(self, artifact: WeftArtifact | str | Path, *, native_builder=None,
                 source_directory=None, native_target=None) -> None:
        if not isinstance(artifact, WeftArtifact):
            artifact = WeftArtifact.read(artifact, load_native=True)
        if artifact.native is None and native_builder is None:
            raise ValueError("Weft native loading requires compile_artifact to finish first")
        self.artifact = artifact
        self._native_builder = native_builder
        self._native_target = native_target
        self.directory = artifact.native.directory if artifact.native is not None else source_directory
        contract = artifact.contract
        self.profile = artifact.profile
        self.facts = contract.facts
        self.target = bindings({"family": "cpu", **{name: value for name, value in asdict(contract.target).items()
                                                  if name != "provider"}})
        self.compilation_cache = () if artifact.native is None else (CacheObservation("native_compilation",
            "disk" if artifact.native_cache_observed else "external_aot",
            artifact.native.cache_hit if artifact.native_cache_observed else None,
            str(artifact.native.directory), "materialization", artifact.native.cache_reason if artifact.native_cache_observed else
            "A saved native artifact was loaded; its compilation cache use was not observed"),)
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
        kernels = {kernel["symbol"]: kernel for kernel in artifact.kernels}
        self.candidate_extensions = tuple(frozenset(
            extension for task in self.facts.tasks if task.cpu_entry == candidate.entry
            for extension in kernels[task.abi.symbol]["used_extensions"])
            for candidate in self.candidates)
        self.used_extensions = frozenset(extension for kernel in kernels.values()
                                         for extension in kernel["used_extensions"])
        self.identity = artifact.native.identity if artifact.native is not None else None
        self._argument_types = abi.argument_types()
        self._library = None
        self._execution = None
        self._closed = False
        self._load_lock = RLock()
        self.functions, self.measurements = [], []

    def compile(self) -> None:
        with self._load_lock:
            if self._closed:
                raise CompilationStageError("provider_native_compilation", "native artifact is closed")
            if self.artifact.native is None:
                self.artifact = self._native_builder(self.artifact)
                self.directory = self.artifact.native.directory
                self.identity = self.artifact.native.identity
                self.compilation_cache = (CacheObservation("native_compilation", "disk",
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
            execution = _ExecutionContract(self.profile, self.artifact.kernels, self._native_target)
            execution.check()
            library = load_native_library(self.artifact.native)
            try:
                execution.bind_library(library)
                execution.check()
                functions = [library.bind(candidate.entry + "_invoke", self._argument_types, None)
                             for candidate in self.candidates]
                measurements = [library.bind(candidate.entry + "_benchmark",
                                              [*self._argument_types, ctypes.c_int64], ctypes.c_double)
                                for candidate in self.candidates]
            except Exception:
                library.close()
                raise
            self._execution, self._library = execution, library
            self.functions, self.measurements = functions, measurements
            return library

    @property
    def library(self):
        return self.ensure_loaded()

    def check_execution(self) -> None:
        self.ensure_loaded()
        self._execution.check()

    def _view(self, parameter: ViewParameter, value) -> ViewFacts:
        if not isinstance(value, Buffer) or value.dtype != parameter.dtype.name:
            raise TypeError(f"{parameter.name} requires a native {parameter.dtype} Buffer")
        if len(value.shape) != len(parameter.shape):
            raise ValueError("native view rank disagrees with the compiler ABI")
        return ViewFacts(value.shape, value.strides, value.pointer, value.allocation, value.allocation_end,
                         value.pointer - value.allocation, value.dtype,
                         value.pointer, value.pointer + value.nbytes, value.allocation)

    def _allocate_output(self, parameter: ViewParameter, shape: tuple[int, ...]) -> tuple[Buffer, ViewFacts]:
        value = Buffer.empty(shape, parameter.dtype.name)
        facts = ViewFacts(value.shape, value.strides, value.pointer, value.allocation, value.allocation_end,
                          value.pointer - value.allocation, value.dtype,
                          value.pointer, value.pointer + value.nbytes, value.allocation)
        return value, facts

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
        return NativeCall(self, bound.arguments, bound.native_arguments, bound.outputs,
                          (self.identity, bound.key), self.describe_arguments(bound, "cpu"),
                          tuple(trial_storage))

    def close(self) -> None:
        with self._load_lock:
            self._closed = True
            self.functions.clear()
            self.measurements.clear()
            if self._execution is not None:
                self._execution.vlen_probe = None
            if self._library is not None:
                self._library.close()
                self._library = None
