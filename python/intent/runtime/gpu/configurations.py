from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ..artifact import ParameterRole, TuningConfiguration, TuningParameter
from ..diagnostics import ConfigurationAssessment, RequirementEvaluation, bindings
from .expressions import Expression


@dataclass(frozen=True, slots=True)
class ConfigurationRequirement:
    kind: str
    metric: str
    predicate: str
    usage: Expression
    limit: Expression | None
    activation: str | None
    message: str
    _reference_keys: tuple[int | str, ...] = field(init=False, repr=False, compare=False)
    _last_evaluation: tuple[tuple[int, ...], RequirementEvaluation] | None = field(
        init=False, default=None, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_reference_keys", tuple(self.references))

    @classmethod
    def read(cls, entry: dict) -> ConfigurationRequirement:
        kind, metric, message = entry["kind"], entry["metric"], entry["message"]
        if kind not in {"legality", "nominal_budget"}:
            raise ValueError("unknown configuration requirement kind")
        if metric not in {"fragment_elements", "fragment_register_words", "fragment_bytes", "resident_workers"}:
            raise ValueError("unknown configuration requirement metric")
        predicate = entry["predicate"]
        if predicate not in {"less_equal", "equal", "positive", "power_of_two", "multiple_of"}:
            raise ValueError("unknown configuration requirement predicate")
        if kind == "nominal_budget" and predicate != "less_equal":
            raise ValueError("nominal budget requires an upper bound")
        limit = entry["limit"]
        if (limit is not None) != (predicate in {"less_equal", "equal", "multiple_of"}):
            raise ValueError("configuration requirement limit disagrees with its predicate")
        activation = entry["activation"]
        if activation is not None and (not isinstance(activation, str) or not activation):
            raise ValueError("configuration activation must name a boolean provider parameter")
        if not isinstance(message, str) or not message:
            raise ValueError("configuration requirement must carry its diagnostic message")
        return cls(kind, metric, predicate, Expression.read(entry["usage"], checked=True),
                   Expression.read(limit, checked=True) if limit is not None else None, activation, message)

    @property
    def expressions(self) -> tuple[Expression, ...]:
        return (self.usage,) if self.limit is None else (self.usage, self.limit)

    @property
    def references(self) -> frozenset[int | str]:
        references = frozenset().union(*(expression.references for expression in self.expressions))
        return references if self.activation is None else references | {self.activation}

    def assess(self, values: Mapping[int | str, object]) -> RequirementEvaluation:
        # Requirements are pure functions of these scalar facts. Keep only the
        # most recent evaluation: tuning repeatedly launches one candidate, but
        # another invocation may have different dimensions, aliases or coverage.
        active = values.get(self.activation) if self.activation is not None else None
        if type(active) is int and active == 0:
            key = (0,)
        else:
            facts = tuple(values.get(name) for name in self._reference_keys)
            if any(type(value) is not int for value in facts):
                return self._assess(values)
            key = (1, *facts)
        previous = self._last_evaluation
        if previous is not None and previous[0] == key:
            return previous[1]
        evaluation = self._assess(values)
        # Publish key and result together; interleaved callers must never pair
        # one invocation's facts with another invocation's evaluation.
        object.__setattr__(self, "_last_evaluation", (key, evaluation))
        return evaluation

    def _assess(self, values: Mapping[int | str, object]) -> RequirementEvaluation:
        def result(status, usage=None, limit=None, detail=""):
            return RequirementEvaluation(self.kind, self.metric, self.predicate,
                                         self.message, status, usage, limit, detail)

        # An unselected local form has no quantity obligation. In particular,
        # its inactive branch need not resolve invocation-derived expressions.
        if self.activation is not None:
            if self.activation not in values:
                return result("unknown", detail=f"missing activation {self.activation!r}")
            active = values[self.activation]
            if type(active) is not int or active not in (0, 1):
                return result("invalid", detail=f"activation {self.activation!r} must be 0 or 1")
            if not active:
                return result("inactive", detail=f"{self.activation}=0")
        errors = []

        def quantity(expression):
            if expression is None:
                return None
            try:
                return expression(values)
            except (ArithmeticError, KeyError) as error:
                errors.append(f"{type(error).__name__}: {error}")
                return None

        usage, limit = quantity(self.usage), quantity(self.limit)
        if (usage is not None and usage <= 0) or (limit is not None and
                          (limit < 0 or self.predicate in {"equal", "multiple_of"} and limit == 0)):
            return result("invalid", usage, limit)
        if self.predicate == "less_equal" and usage is None and limit is not None:
            bound = self.usage.nonnegative_bound(values, limit)
            if bound is not None and bound > limit:
                return result("violated", limit=limit, detail="positive quantity exceeds the finite bound")
        if usage is None or self.limit is not None and limit is None:
            return result("unknown", usage, limit, "; ".join(errors))
        if self.predicate == "less_equal":
            satisfied = usage <= limit
        elif self.predicate == "equal":
            satisfied = usage == limit
        elif self.predicate == "positive":
            satisfied = True
        elif self.predicate == "power_of_two":
            satisfied = (usage & (usage - 1)) == 0
        else:
            satisfied = usage % limit == 0
        return result("satisfied" if satisfied else "violated", usage, limit)


@dataclass(frozen=True, slots=True)
class CoverageBinding:
    name: str
    bound: Expression
    candidates: tuple[int, ...]

    def select(self, values: Mapping[int | str, object]) -> int:
        required = self.bound(values)
        selected = next((candidate for candidate in self.candidates if candidate >= required), None)
        if selected is None:
            raise ValueError(f"no legal full-coverage extent for {self.name}: required {required}")
        return selected


class ConfigurationSpace:
    """Final IR candidate bindings and their invocation-dependent constraints.

    Rows retain compiler order. Coverage is bound from invocation facts; provider
    adapters only project a row into their native JIT and tuner interfaces.
    """

    def __init__(self, interface: dict) -> None:
        self.rows = tuple(interface["configurations"])
        self.parameters = tuple(TuningParameter(
            entry["name"], ParameterRole(entry["role"]), entry["category"],
            tuple(entry["candidates"]), entry.get("dimension"),
            tuple(entry["source"]) if entry.get("source") is not None else None,
            tuple(entry["argument_axis"]) if entry.get("argument_axis") is not None else None,
        ) for entry in interface["parameters"])
        self.requirements = tuple(ConfigurationRequirement.read(entry) for entry in interface["requirements"])
        deferred = tuple(CoverageBinding(entry["name"], Expression.read(entry["coverage"]),
                                         tuple(entry["candidates"]))
                         for entry in interface["parameters"] if entry.get("coverage") is not None)
        deferred_names = frozenset(binding.name for binding in deferred)
        self.bound_names = frozenset(parameter.name for parameter in self.parameters
                                     if parameter.name not in deferred_names)
        names = {parameter.name for parameter in self.parameters}
        if len(names) != len(self.parameters) or not self.rows:
            raise ValueError("configuration space requires unique parameters and nonempty candidate rows")
        domains = {parameter.name: parameter.candidates for parameter in self.parameters}
        value_types = {entry["name"]: entry["value_type"] for entry in interface["parameters"]}
        for parameter in self.parameters:
            value_type = value_types[parameter.name]
            if value_type not in {"bool", "index"}:
                raise ValueError("configuration parameter must have a boolean or index value type")
            if value_type == "bool" and (parameter.category != 6 or parameter.name in deferred_names):
                raise ValueError("boolean configuration choices must be bound provider parameters")
            if not parameter.candidates or any(
                type(value) is not int or (value not in (0, 1) if value_type == "bool" else value <= 0)
                for value in parameter.candidates
            ):
                raise ValueError("configuration parameter candidates disagree with its value type")
        self.coverage, self.derived = self._order_deferred(deferred, names, value_types)
        self.coverage_names = tuple(binding.name for binding in self.coverage)
        self.derived_names = frozenset(binding.name for binding in self.derived)
        for row in self.rows:
            if set(row) != self.bound_names or any(type(value) is not int or value not in domains[name]
                                                   for name, value in row.items()):
                raise ValueError("configuration row must bind each non-deferred parameter within its declared domain")
        for requirement in self.requirements:
            if any(isinstance(reference, str) and reference not in names for reference in requirement.references):
                raise ValueError("configuration requirement references an undeclared parameter")
            if requirement.activation is not None and value_types[requirement.activation] != "bool":
                raise ValueError("configuration activation requires a boolean provider parameter")
            for expression in requirement.expressions:
                if any(isinstance(reference, str) and value_types[reference] != "index"
                       for reference in expression.references):
                    raise ValueError("configuration quantity expressions require index parameters")

    def _order_deferred(self, deferred, names, value_types):
        """Separate invocation extents from extents depending on a selected row."""
        pending = list(deferred)
        resolved = set(self.bound_names)
        candidate_dependent = set(self.bound_names)
        coverage, derived = [], []
        for binding in pending:
            for reference in binding.bound.references:
                if isinstance(reference, str) and (
                    reference not in names or value_types[reference] != "index"
                ):
                    raise ValueError("deferred extent must reference declared index parameters")
        while pending:
            ready = next((binding for binding in pending if all(
                not isinstance(reference, str) or reference in resolved
                for reference in binding.bound.references)), None)
            if ready is None:
                raise ValueError("deferred configuration extents contain a dependency cycle")
            pending.remove(ready)
            if ready.bound.references & candidate_dependent:
                derived.append(ready)
                candidate_dependent.add(ready.name)
            else:
                coverage.append(ready)
            resolved.add(ready.name)
        return tuple(coverage), tuple(derived)

    def assess(self, values: Mapping[int | str, object]) -> tuple[RequirementEvaluation, ...]:
        """Evaluate declared candidate conditions, independently of native allocation."""
        return tuple(requirement.assess(values) for requirement in self.requirements)

    @staticmethod
    def _within(evaluations: tuple[RequirementEvaluation, ...]) -> bool:
        if any(entry.status in {"violated", "invalid"} for entry in evaluations):
            return False
        unresolved = tuple(entry for entry in evaluations if entry.status == "unknown")
        if unresolved:
            from ...compiler.toolchain import CompilationStageError
            raise CompilationStageError("candidate_binding", "; ".join(entry.describe() for entry in unresolved))
        return all(entry.accepted for entry in evaluations)

    def satisfies_requirements(self, values: Mapping[int | str, object]) -> bool:
        return self._within(self.assess(values))

    def candidates(self, values: Mapping[int | str, object], *,
                   rows: Iterable[Mapping[str, int]] | None = None) -> tuple[dict, ...]:
        return self.select(self.inspect(values, rows=rows))

    def bound_configuration(self, values: Mapping[int | str, object], row: Mapping[str, int]) -> dict:
        result = {**row, **{name: values[name] for name in self.coverage_names}}
        for binding in self.derived:
            result[binding.name] = binding.select({**values, **result})
        return result

    def inspect(self, values: Mapping[int | str, object], *,
                rows: Iterable[Mapping[str, int]] | None = None) -> tuple[ConfigurationAssessment, ...]:
        """Evaluate current rows once, retaining rejected conditions for inspection."""
        selected = self.rows if rows is None else tuple(rows)
        if any(row not in self.rows for row in selected):
            from ...compiler.toolchain import CompilationStageError
            raise CompilationStageError("candidate_binding", "candidate is absent from the generated configuration space")
        result = []
        for row in selected:
            bound = self.bound_configuration(values, row)
            result.append(ConfigurationAssessment(bindings(bound), self.assess({**values, **bound})))
        return tuple(result)

    def select(self, assessed: tuple[ConfigurationAssessment, ...]) -> tuple[dict, ...]:
        """Select evaluated rows without reinterpreting or re-evaluating conditions."""
        result = tuple({name: value for name, value in entry.configuration if name in self.bound_names}
                       for entry in assessed if entry.provider_reason is None and self._within(entry.requirements))
        if not result:
            from ...compiler.toolchain import CompilationStageError
            reasons = []
            for entry in assessed[:4]:
                rejected = [value.describe() for value in entry.requirements if not value.accepted]
                if entry.provider_reason is not None:
                    rejected.append(entry.provider_reason)
                reasons.append(f"{dict(entry.configuration)}: " + "; ".join(rejected[:3]))
            if len(assessed) > 4:
                reasons.append(f"{len(assessed) - 4} additional candidate rows rejected")
            stage = "provider_eligibility" if any(
                all(value.accepted for value in entry.requirements) for entry in assessed) else "candidate_selection"
            raise CompilationStageError(stage,
                "no candidate satisfies the declared configuration requirements\n" + "\n".join(reasons))
        return result

    def enumerate(self, values: Mapping[int | str, object], rows: Iterable[Mapping[str, int]]) -> tuple[TuningConfiguration, ...]:
        result = []
        for row in rows:
            bindings = {**values, **self.bound_configuration(values, row)}
            result.append(TuningConfiguration(self.parameters,
                                               tuple(bindings[parameter.name] for parameter in self.parameters)))
        return tuple(result)
