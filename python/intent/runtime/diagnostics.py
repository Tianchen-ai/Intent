"""Observations from an existing provider invocation, never compilation policy."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any, Callable, Mapping, cast

from .interface import PublicInterface, ViewParameter
from .invocation import ViewFacts


def bindings(values: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    def freeze(value):
        if isinstance(value, Mapping):
            return tuple((name, freeze(item)) for name, item in sorted(value.items()))
        if isinstance(value, (tuple, list)):
            return tuple(freeze(item) for item in value)
        return value
    return freeze(values)


@dataclass(frozen=True, slots=True)
class CacheObservation:
    """A reuse decision at the named stage/layer, never an inferred SDK cache hit.

    Materialization and selection facts remain attached to the prepared call
    they produced. Later dispatch can reuse that call without implying a native
    compiler or provider tuning cache hit.
    """

    layer: str
    scope: str
    hit: bool | None
    source: str
    stage: str
    reason: str | None = None
    entry: str | None = None


@dataclass(frozen=True, slots=True)
class NativeResource:
    name: str
    value: int | None
    unit: str
    source: str
    stage: str
    unavailable_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    """A provider compilation or trial; elapsed_ms is existing tuning data only."""

    configuration: tuple[tuple[str, object], ...] | None
    status: str
    stage: str
    error_type: str | None = None
    message: str | None = None
    elapsed_ms: float | None = None


@dataclass(frozen=True, slots=True)
class RequirementEvaluation:
    """Evaluation of one compiler-declared candidate condition."""

    kind: str
    metric: str
    predicate: str
    message: str
    status: str
    usage: int | None
    limit: int | None
    detail: str = ""

    def describe(self) -> str:
        quantities = []
        if self.usage is not None:
            quantities.append(f"usage={self.usage}")
        if self.limit is not None:
            quantities.append(f"limit={self.limit}")
        if self.detail:
            quantities.append(self.detail)
        suffix = "; " + ", ".join(quantities) if quantities else ""
        return f"{self.kind}/{self.metric}/{self.predicate}: {self.message} ({self.status}{suffix})"

    @property
    def accepted(self) -> bool:
        return self.status in {"satisfied", "inactive"}


@dataclass(frozen=True, slots=True)
class ConfigurationAssessment:
    """A declared configuration's current eligibility, without compiling or timing it."""

    configuration: tuple[tuple[str, object], ...]
    requirements: tuple[RequirementEvaluation, ...]
    provider_reason: str | None = None

    @property
    def status(self) -> str:
        if self.provider_reason is not None or any(
                item.status in {"violated", "invalid"} for item in self.requirements):
            return "rejected"
        if any(item.status == "unknown" for item in self.requirements):
            return "unknown"
        return "eligible"


@dataclass(frozen=True, slots=True)
class InvocationArgument:
    name: str
    dtype: str
    shape: tuple[int, ...] | None
    strides: tuple[int, ...] | None
    device: str | None
    scalar: int | float | bool | None = None


@dataclass(frozen=True, slots=True)
class NativeObservation:
    """Latest native compilation or invocation; missing SDK facts stay unknown.

    Arguments describe this invocation, not a promise that every scalar belongs
    to the provider's specialization key. Candidate bindings and native resource
    fields describe the selected compiled program, if one was selected. This is not an occupancy
    estimate, a correctness result, or an Intent optimization decision.
    """

    provider: str
    stage: str
    target: tuple[tuple[str, object], ...]
    invocation: tuple[InvocationArgument, ...]
    configuration: tuple[tuple[str, object], ...] | None
    resources: tuple[NativeResource, ...]
    candidates: tuple[CandidateObservation, ...]
    candidate_history_unavailable: str | None = None
    caches: tuple[CacheObservation, ...] = ()


class _ObservationSource:
    """Materialize immutable invocation facts without revisiting execution."""

    __slots__ = ("_factory", "_value")

    def __init__(self, factory: Callable[[], NativeObservation]):
        self._factory = factory
        self._value: NativeObservation | None = None

    def resolve(self) -> NativeObservation:
        factory = self._factory
        if factory is not None:
            value = factory()
            self._value = value
            self._factory = None
        return cast(NativeObservation, self._value)


@dataclass(slots=True)
class _DeferredObservation:
    _source: _ObservationSource
    stage: str
    _replayed: bool = False
    _value: NativeObservation | None = field(default=None, init=False)

    def executed(self, stage: str, replayed: bool) -> _DeferredObservation:
        # Repeated launch/benchmark transitions share one immutable source,
        # rather than building a chain of pending snapshots.
        return _DeferredObservation(self._source, stage, self._replayed or replayed)

    def resolve(self) -> NativeObservation:
        if self._value is None:
            source = self._source.resolve()
            caches = source.caches
            if self._replayed:
                caches = (*caches, _prepared_replay_observation())
            self._value = replace(source, stage=self.stage, caches=caches)
        return self._value


def _prepared_replay_observation() -> CacheObservation:
    return CacheObservation("prepared_call", "prepared_call", True, "Stored prepared invocation",
                            "dispatch", "The selected invocation was reused; SDK cache behavior is separate")


def resolve_observation(observed: NativeObservation | _DeferredObservation | None) -> NativeObservation | None:
    return observed.resolve() if isinstance(observed, _DeferredObservation) else observed


class ObservedCall:
    """Share a prepared call's latest snapshot with its owning runtime."""

    _observation: NativeObservation | _DeferredObservation | None = None
    _replay_observation: NativeObservation | _DeferredObservation | None = None
    _executed = False

    @property
    def observation(self) -> NativeObservation | None:
        """Read the latest native snapshot without choosing, compiling or launching."""
        return resolve_observation(self._observation)

    def _record_observation(self, observed: NativeObservation | _DeferredObservation | None) -> None:
        self._observation = observed
        self.program._observation = observed

    def _defer_observation(self, factory: Callable[[], NativeObservation], *, stage: str) -> None:
        self._record_observation(_DeferredObservation(_ObservationSource(factory), stage))

    def _record_execution(self, stage: str, observed: NativeObservation | None = None,
                          *, replay: bool | None = None) -> None:
        current = observed if observed is not None else self._observation
        reused = self._executed if replay is None else replay
        self._executed = True
        if not reused:
            self._replay_observation = None
        if current is None:
            self._record_observation(None)
            return

        def dispatched(base, *, reused):
            if isinstance(base, _DeferredObservation):
                return base.executed(stage, reused)
            caches = (*base.caches, _prepared_replay_observation()) if reused else base.caches
            return replace(base, stage=stage, caches=caches)

        if reused:
            if self._replay_observation is None:
                self._replay_observation = dispatched(current, reused=True)
            elif self._replay_observation.stage != stage:
                self._replay_observation = dispatched(self._replay_observation, reused=False)
            current = self._replay_observation
        elif current.stage != stage:
            current = dispatched(current, reused=False)
        self._record_observation(current)


class CandidateRecorder:
    """One bounded record per candidate considered by the existing tuner."""

    def __init__(self) -> None:
        self._records: dict[tuple | None, CandidateObservation] = {}
        self._lock = Lock()

    def record(self, configuration: Mapping[str, object] | None, status: str, stage: str,
               error: Exception | None = None) -> None:
        row = bindings(configuration) if configuration is not None else None
        item = CandidateObservation(row, status, stage,
                                    type(error).__name__ if error is not None else None,
                                    str(error) if error is not None else None)
        with self._lock:
            self._records[row] = item

    def timing(self, configuration: Mapping[str, object], elapsed_ms: float) -> None:
        """Attach a completed provider measurement to an already observed trial."""
        row = bindings(configuration)
        with self._lock:
            previous = self._records.get(row)
            if previous is not None and previous.status != "failed":
                self._records[row] = replace(previous, elapsed_ms=elapsed_ms)

    def snapshot(self) -> tuple[CandidateObservation, ...]:
        with self._lock:
            return tuple(self._records.values())


def invocation_arguments(interface: PublicInterface, arguments: tuple,
                         facts: tuple[ViewFacts | None, ...], device: str) -> tuple[InvocationArgument, ...]:
    """Snapshot the common binding's observations without querying a device or tensor."""
    result = []
    for parameter, value, view in zip(interface.parameters, arguments, facts, strict=True):
        if isinstance(parameter, ViewParameter):
            result.append(InvocationArgument(parameter.name, parameter.dtype.name,
                                             view.shape, view.strides, device))
        else:
            result.append(InvocationArgument(parameter.name, parameter.dtype.name, None, None, None, value))
    return tuple(result)


def observation(provider: str, target: tuple[tuple[str, object], ...],
                invocation: tuple[InvocationArgument, ...],
                configuration: tuple[tuple[str, object], ...] | None,
                resources: tuple[NativeResource, ...], candidates=(), *,
                stage: str = "launched", history_unavailable: str | None = None,
                caches: tuple[CacheObservation, ...] = ()) -> NativeObservation:
    """Assemble a call snapshot from explicitly frozen target/configuration descriptions."""
    return NativeObservation(provider, stage, target, invocation, configuration,
                             resources, tuple(candidates), history_unavailable, caches)


def resource(name: str, value: Any, unit: str, source: str, stage: str, *,
             unavailable: str) -> NativeResource:
    if value is None:
        return NativeResource(name, None, unit, source, stage, unavailable)
    if type(value) is not int:
        return NativeResource(name, None, unit, source, stage,
                              f"SDK returned {type(value).__name__}, not an integer resource count")
    return NativeResource(name, value, unit, source, stage)


def unavailable_resources(provider: str, reason: str) -> tuple[NativeResource, ...]:
    return tuple(resource(name, None, unit, provider, "native", unavailable=reason) for name, unit in (
        ("registers_per_thread", "registers"),
        ("local_memory_words_per_thread", "32-bit words"),
        ("shared_memory_bytes", "bytes"),
        ("max_threads_per_block", "threads")))
