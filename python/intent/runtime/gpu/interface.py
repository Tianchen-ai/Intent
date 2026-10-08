from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

from ...language.dtypes import DType, dtype
from ..interface import PublicInterface, ScalarParameter, ViewParameter, byte_spans_overlap
from ..invocation import build_invocation_binders, invocation_result
from ..torch import torch_dtype
from ..torch_views import allocate_output, check_abstract_relation, observe_view, view_byte_span
from .configurations import ConfigurationSpace, CoverageBinding
from .expressions import Expression, evaluate_shape, read_expressions
from .contract import require_references


@dataclass(frozen=True, slots=True)
class Argument:
    abi: int
    id: int
    kernel_name: str


@dataclass(frozen=True, slots=True)
class PublicArgument(Argument):
    parameter: ScalarParameter | ViewParameter

    @property
    def writable(self) -> bool:
        return isinstance(self.parameter, ViewParameter) and self.parameter.writable

    @property
    def access(self):
        return self.parameter.access


@dataclass(frozen=True, slots=True)
class MetadataArgument(Argument):
    kind: str
    source: int
    axis: int
    dimension: int | None


@dataclass(frozen=True, slots=True)
class WorkspaceArgument(Argument):
    dtype: DType
    shape: tuple[Expression, ...]

    @property
    def writable(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class OverlapBinding:
    name: str
    lhs: int
    rhs: int


@dataclass(slots=True)
class BoundInvocation:
    interface: GPUInterface
    arguments: tuple
    outputs: tuple
    values: dict[int | str, object]
    description: tuple

    @property
    def views(self) -> tuple:
        return tuple(self.values[view.id] for view in self.interface.views)

    @property
    def public_views(self) -> tuple:
        return tuple(self.values[view.id] for view in self.interface.public_views)

    def result(self):
        return invocation_result(self.outputs)


class GPUInterface:
    """Bind public parameters to the final GPU program's physical arguments."""

    def __init__(self, metadata: dict) -> None:
        self.public = PublicInterface.read(metadata["interface"])
        physical = metadata["gpu"]
        arguments = []
        for entry in physical["arguments"]:
            common = entry["abi"], entry["id"], entry["kernel_name"]
            if type(common[0]) is not int or common[0] < 0:
                raise ValueError("GPU argument slots must be nonnegative integers")
            if type(common[1]) is not int or not 0 < common[1] < 1 << 64:
                raise ValueError("GPU argument identities must be positive unsigned 64-bit integers")
            if not isinstance(common[2], str) or not common[2]:
                raise ValueError("GPU native argument names must be nonempty strings")
            if entry["kind"] == "public":
                position = entry["parameter"]
                if type(position) is not int or not 0 <= position < len(self.public.parameters):
                    raise ValueError("GPU binding references an unknown public parameter")
                arguments.append(PublicArgument(*common, self.public.parameters[position]))
            elif entry["kind"] in {"dimension", "stride"}:
                if type(entry["source"]) is not int or not 0 < entry["source"] < 1 << 64:
                    raise ValueError("GPU metadata source must be a stable argument identity")
                dimension = entry["dimension"] if entry["kind"] == "dimension" else None
                if entry["kind"] == "dimension" and (type(dimension) is not int or dimension <= 0):
                    raise ValueError("GPU logical dimension identity must be a positive integer")
                arguments.append(MetadataArgument(*common, entry["kind"], entry["source"], entry["axis"], dimension))
            elif entry["kind"] == "workspace":
                arguments.append(WorkspaceArgument(*common, dtype(entry["dtype"]), read_expressions(entry["shape"])))
            else:
                raise ValueError("unknown GPU argument binding kind")
        self.arguments = tuple(arguments)
        self.by_name = {entry.kernel_name: entry for entry in arguments}
        self.by_id = {entry.id: entry for entry in arguments}
        if len(self.by_id) != len(arguments) or len(self.by_name) != len(arguments):
            raise ValueError("GPU argument ids and native names must be unique")
        if tuple(entry.abi for entry in arguments) != tuple(range(len(arguments))):
            raise ValueError("GPU arguments must be ordered by current ABI slot")
        self._public_bindings = {entry.parameter.position: entry for entry in arguments if isinstance(entry, PublicArgument)}
        if (len(self._public_bindings) != len(self.public.parameters) or
                sum(isinstance(entry, PublicArgument) for entry in arguments) != len(self.public.parameters)):
            raise ValueError("GPU physical arguments must bind each public parameter exactly once")
        self.public_views = tuple(entry for entry in arguments
                                  if isinstance(entry, PublicArgument) and isinstance(entry.parameter, ViewParameter))
        self.scalars = tuple(entry for entry in arguments
                             if isinstance(entry, PublicArgument) and isinstance(entry.parameter, ScalarParameter))
        self.workspaces = tuple(entry for entry in arguments if isinstance(entry, WorkspaceArgument))
        self.views = tuple(entry for entry in arguments if isinstance(entry, WorkspaceArgument) or entry in self.public_views)
        self.metadata = tuple(entry for entry in arguments if isinstance(entry, MetadataArgument))
        self.grid = read_expressions(physical["grid"])
        self.configuration_space = ConfigurationSpace(physical)
        overlap_names = set()
        overlaps = []
        view_ids = {view.id for view in self.views}
        parameter_names = {parameter.name for parameter in self.configuration_space.parameters}
        for entry in physical["overlaps"]:
            name, lhs, rhs = entry["name"], entry["lhs"], entry["rhs"]
            if not isinstance(name, str) or not name or name in overlap_names | parameter_names | self.by_name.keys():
                raise ValueError("GPU overlap binding requires a unique native name")
            if any(type(identity) is not int or identity not in view_ids for identity in (lhs, rhs)):
                raise ValueError("GPU overlap binding must reference two declared views")
            overlap_names.add(name)
            overlaps.append(OverlapBinding(name, lhs, rhs))
        self.overlaps = tuple(overlaps)
        require_references(self.grid, self)
        require_references(tuple(binding.bound for binding in (
            *self.configuration_space.coverage, *self.configuration_space.derived)), self)
        for requirement in self.configuration_space.requirements:
            require_references(requirement.expressions, self)
        self._binders = build_invocation_binders(
            self.public, observe_view=self._observe_view, allocate_output=self._allocate_output)
        self._abstract_binders = build_invocation_binders(
            self.public, observe_view=self._observe_abstract_view, allocate_output=self._allocate_abstract_output,
            check_relation=check_abstract_relation, abstract=True)
        self._host_order = self._host_bindings()

    def _host_bindings(self) -> tuple[MetadataArgument | CoverageBinding | WorkspaceArgument, ...]:
        """Order existing producers; ABI slot order is not a host dependency."""
        nodes = (*self.metadata, *self.configuration_space.coverage, *self.workspaces)
        roots = {entry.id for entry in self._public_bindings.values()}
        positions, dependencies = {}, []
        dimensions, strides = set(), set()
        for index, entry in enumerate(nodes):
            key = entry.name if isinstance(entry, CoverageBinding) else entry.id
            if key in positions or key in roots:
                raise ValueError(f"duplicate GPU host binding {key!r}")
            positions[key] = index
            if isinstance(entry, MetadataArgument):
                source = self.by_id.get(entry.source)
                if entry.kind == "dimension" and not (
                    isinstance(source, PublicArgument) and isinstance(source.parameter, ViewParameter)
                ):
                    raise ValueError("logical dimensions must be bound from public views")
                if isinstance(source, PublicArgument) and isinstance(source.parameter, ViewParameter):
                    rank = len(source.parameter.shape)
                elif isinstance(source, WorkspaceArgument):
                    rank = len(source.shape)
                else:
                    raise ValueError("GPU metadata must refer to a declared view or workspace")
                if type(entry.axis) is not int or not 0 <= entry.axis < rank:
                    raise ValueError("GPU metadata must refer to a valid resource axis")
                if entry.kind == "dimension":
                    if source.parameter.dimensions[entry.axis] != entry.dimension or entry.dimension in dimensions:
                        raise ValueError("GPU dimension metadata must uniquely bind its declared logical dimension")
                    dimensions.add(entry.dimension)
                else:
                    identity = entry.source, entry.axis
                    if identity in strides:
                        raise ValueError("GPU stride metadata must uniquely bind a resource axis")
                    strides.add(identity)
                dependencies.append(frozenset((entry.source,)))
            elif isinstance(entry, CoverageBinding):
                if any(isinstance(reference, str) and reference not in self.configuration_space.coverage_names
                       for reference in entry.bound.references):
                    raise ValueError("invocation extents cannot depend on unselected candidate parameters")
                dependencies.append(entry.bound.references)
            else:
                dependencies.append(frozenset(reference for extent in entry.shape for reference in extent.references))
        incoming, followers = [0] * len(nodes), [[] for _ in nodes]
        for index, references in enumerate(dependencies):
            for reference in references:
                if reference in roots:
                    continue
                producer = positions.get(reference)
                if producer is None:
                    if reference in self.configuration_space.bound_names | self.configuration_space.derived_names:
                        raise ValueError("GPU host allocation cannot depend on an unselected candidate parameter")
                    raise ValueError(f"GPU host binding references an unavailable argument or parameter: {reference!r}")
                incoming[index] += 1
                followers[producer].append(index)
        ready = []
        for index, count in enumerate(incoming):
            if count == 0:
                heappush(ready, index)
        ordered = []
        while ready:
            index = heappop(ready)
            ordered.append(nodes[index])
            for consumer in followers[index]:
                incoming[consumer] -= 1
                if incoming[consumer] == 0:
                    heappush(ready, consumer)
        if len(ordered) != len(nodes):
            raise ValueError("GPU host bindings contain a cycle between metadata, coverage or workspace producers")
        return tuple(ordered)

    def native_value(self, name: str, values: dict):
        """Resolve spelling only at the provider's native call boundary."""
        argument = self.by_name.get(name)
        return values[argument.id if argument is not None else name]

    def callback_values(self, arguments: dict) -> dict:
        values = dict(arguments)
        for name, argument in self.by_name.items():
            if name in arguments:
                values[argument.id] = values.pop(name)
        return values

    @staticmethod
    def _observe_view(device: int, parameter: ViewParameter, value):
        import torch
        return observe_view(parameter, value, device=torch.device("cuda", device))

    @staticmethod
    def _allocate_output(device: int, parameter: ViewParameter, shape: tuple):
        import torch
        return allocate_output(parameter, shape, device=torch.device("cuda", device))

    @staticmethod
    def _observe_abstract_view(device: int, parameter: ViewParameter, value):
        import torch
        return observe_view(parameter, value, device=torch.device("cuda", device), abstract=True)

    @staticmethod
    def _allocate_abstract_output(device: int, parameter: ViewParameter, shape: tuple):
        import torch
        return allocate_output(parameter, shape, device=torch.device("cuda", device), abstract=True)

    def bind(self, arguments: tuple, *, device: int, outputs: tuple | None = None,
             explicit_outputs: bool = False, abstract: bool = False) -> BoundInvocation:
        import torch

        if explicit_outputs and outputs is not None:
            raise TypeError("explicit arguments already contain the output buffers")
        if outputs is not None:
            arguments = self.public.explicit_arguments(arguments, outputs)
            explicit_outputs = True
        binders = self._abstract_binders if abstract else self._binders
        public = binders[bool(explicit_outputs)](device, arguments)
        values = {self._public_bindings[position].id: value for position, value in enumerate(public.arguments)}
        from ..diagnostics import invocation_arguments
        description = () if abstract else invocation_arguments(
            self.public, public.arguments, public.views, f"cuda:{device}")
        result = BoundInvocation(self, public.arguments, public.outputs, values, description)
        if abstract:
            return result
        for entry in self._host_order:
            if isinstance(entry, MetadataArgument):
                source = values[entry.source]
                values[entry.id] = source.shape[entry.axis] if entry.kind == "dimension" else source.stride(entry.axis)
            elif isinstance(entry, CoverageBinding):
                values[entry.name] = entry.select(values)
            else:
                values[entry.id] = torch.empty(evaluate_shape(entry.shape, values), dtype=torch_dtype(entry.dtype),
                                               device=torch.device("cuda", device))
        spans = {}
        for overlap in self.overlaps:
            pair = []
            for identity in (overlap.lhs, overlap.rhs):
                if identity not in spans:
                    spans[identity] = view_byte_span(values[identity])
                pair.append(spans[identity])
            values[overlap.name] = byte_spans_overlap(*pair)
        return result
