from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
from threading import Lock

from .buffer import Buffer
from .compilation import WeftArtifact
from .target import TargetProfile, matrix_capability
from ..interface import ViewParameter
from ..invocation import ViewFacts, invocation_result
from ..native import NativePreparedRuntime
from ..native_artifact import load_native_library
from ..diagnostics import CacheObservation, CandidateObservation, ObservedCall, bindings, observation
from intent.compiler.toolchain import CompilationStageError


def _isa_extensions(isa: str) -> set[str]:
    base, *extensions = isa.split("_")
    if not base.startswith("rv64"):
        raise ValueError("native CPU does not report an RV64 ISA")
    result = set(base[4:]) | set(extensions)
    if "g" in result:
        result.remove("g")
        result |= {"i", "m", "a", "f", "d", "zicsr", "zifencei"}
    return result


def _cpu_facts() -> dict[int, dict[str, str]]:
    facts = {}
    for paragraph in Path("/proc/cpuinfo").read_text().split("\n\n"):
        fields = dict(line.split(":", 1) for line in paragraph.splitlines() if ":" in line)
        fields = {key.strip(): value.strip() for key, value in fields.items()}
        if "processor" in fields:
            facts[int(fields["processor"])] = fields
    return facts


class _ExecutionContract:
    """Artifact-local hardware facts, with calling-thread state checked on use."""

    def __init__(self, profile: TargetProfile, used_extensions: frozenset[str]) -> None:
        if sys.platform != "linux" or platform.machine() != "riscv64" or sys.byteorder != "little":
            raise NotImplementedError("Weft native loading requires little-endian RISC-V Linux")
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            raise ValueError("native pointer size disagrees with RV64/lp64d")
        self.profile = profile
        self.required = _isa_extensions(profile.march)
        self.capabilities = tuple((extension, matrix_capability(extension, profile.vlen_bits))
                                  for extension in used_extensions)
        self.facts = _cpu_facts()
        self.validated_cpus: set[int] = set()
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.prctl = self.libc.prctl
        self.prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
        self.prctl.restype = ctypes.c_int
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
        for cpu in cpus - self.validated_cpus:
            # ISA and vendor identity belong to the CPU. A newly online CPU may
            # be absent from the artifact's initial /proc/cpuinfo snapshot.
            if cpu not in self.facts:
                self.facts = _cpu_facts()
            facts = self.facts[cpu]
            extensions = _isa_extensions(facts["isa"])
            missing = self.required - extensions
            if missing:
                raise ValueError(f"CPU {cpu} lacks requested ISA extensions: {sorted(missing)}")
            for extension, capability in self.capabilities:
                if (capability.isa_feature not in extensions or
                        int(facts["mvendorid"], 0) != capability.vendor_id or
                        int(facts["marchid"], 0) != capability.architecture_id):
                    raise ValueError(f"CPU {cpu} does not support the artifact's {extension} instructions")
            self.validated_cpus.add(cpu)
        control = self.prctl(70, 0, 0, 0, 0)
        if control < 0:
            raise OSError(ctypes.get_errno(), "cannot query the calling thread's RVV state")
        if control & 3 == 1:
            raise ValueError("RVV is disabled for the calling thread")
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
        """Validate the AOT portfolio without loading code or probing a device."""
        with self._native_stage("provider_native_compilation"):
            self.program.compile()
        self._record_observation(observation(
            "weft", self.program.target, self.description, None, (),
            tuple(CandidateObservation(configuration, "compiled", "provider_native_compilation")
                  for configuration in self.program.configuration_descriptions),
            stage="compiled", caches=self.program.compilation_cache,
            history_unavailable="AOT artifact validation does not load or time a native candidate"))

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


class NativeProgram(NativePreparedRuntime):
    def __init__(self, artifact: WeftArtifact | str | Path) -> None:
        if not isinstance(artifact, WeftArtifact):
            artifact = WeftArtifact.read(artifact, load_native=True)
        if artifact.native is None:
            raise ValueError("Weft native loading requires compile_artifact to finish first")
        self.artifact = artifact
        self.directory = artifact.native.directory
        contract = artifact.contract
        self.profile = artifact.profile
        self.facts = contract.facts
        self.target = bindings({"family": "cpu", **{name: value for name, value in asdict(contract.target).items()
                                                  if name != "provider"}})
        self.compilation_cache = (CacheObservation("native_compilation",
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
        self.identity = artifact.native.identity
        self._argument_types = abi.argument_types()
        self._library = None
        self._execution = None
        self._closed = False
        self._load_lock = Lock()
        self.functions, self.measurements = [], []

    def compile(self) -> None:
        if self._closed:
            raise CompilationStageError("provider_native_compilation", "native artifact is closed")
        self.artifact.native.validate()

    def ensure_loaded(self):
        with self._load_lock:
            if self._closed:
                raise CompilationStageError("native_loading", "native artifact is closed")
            if self._library is not None:
                return self._library
            execution = _ExecutionContract(self.profile, self.used_extensions)
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
