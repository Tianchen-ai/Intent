from __future__ import annotations

from dataclasses import dataclass

from ..language.annotations import ViewKind
from ..language.dtypes import DType, dtype


def byte_spans_overlap(lhs: tuple[int, int], rhs: tuple[int, int]) -> bool:
    left_start, left_end = lhs
    right_start, right_end = rhs
    return (left_start < left_end and right_start < right_end and
            left_start < right_end and right_start < left_end)


@dataclass(frozen=True, slots=True)
class ScalarParameter:
    position: int
    name: str
    dtype: DType


@dataclass(frozen=True, slots=True)
class StrideSymbol:
    name: str


@dataclass(frozen=True, slots=True)
class ViewParameter:
    position: int
    name: str
    dtype: DType
    shape: tuple[int, ...]
    dimensions: tuple[int, ...]
    strides: tuple[int | StrideSymbol | None, ...]
    access: ViewKind
    alias: str
    noalias: bool

    @property
    def writable(self) -> bool:
        return self.access is not ViewKind.IN

    @property
    def output(self) -> bool:
        return self.access is ViewKind.OUT


@dataclass(frozen=True, slots=True)
class ViewAxis:
    parameter: int
    axis: int


@dataclass(frozen=True, slots=True)
class ViewBinding:
    parameter: ViewParameter
    shape_checks: tuple[tuple[int, int | ViewAxis], ...]
    stride_checks: tuple[tuple[int, int | ViewAxis], ...]
    output_shape: tuple[int | ViewAxis | None, ...]


@dataclass(frozen=True, slots=True)
class BindingRelations:
    """Static relations and output-construction eligibility for one call mode.

    A declaration may require explicit output buffers without preventing its
    program from being compiled, saved, loaded or called with those buffers.
    """

    supplied: tuple[ScalarParameter | ViewParameter, ...]
    views: tuple[ViewBinding, ...]
    dimensions: tuple[tuple[int, ViewAxis], ...]
    allocation_errors: tuple[str, ...]

    def require_output_allocation(self) -> None:
        """Require that this call can construct its omitted Out buffers."""
        if self.allocation_errors:
            raise NotImplementedError(
                "Cannot allocate the declared outputs: " + "; ".join(self.allocation_errors) +
                ". Provide explicit output buffers with prepare(..., outputs=(...)) or launch(...).")


@dataclass(frozen=True, slots=True)
class AliasCheck:
    left: int
    right: int
    writable: bool
    noalias: bool
    same_allocation: bool

    def noalias_violation(self, left: tuple[int, int], right: tuple[int, int]) -> bool:
        return self.noalias and byte_spans_overlap(left, right)

    def allocation_violation(self, left: object, right: object) -> bool:
        return self.same_allocation and left != right


@dataclass(frozen=True, slots=True)
class PublicInterface:
    """The author's ordered runtime parameters, independent of a device ABI.

    Shapes, element strides and allocation constraints describe public views.
    Workspaces, launch parameters and native calling conventions are separate.
    """

    parameters: tuple[ScalarParameter | ViewParameter, ...]

    @classmethod
    def read(cls, metadata: dict) -> PublicInterface:
        parameters = []
        names = set()
        for position, entry in enumerate(metadata["parameters"]):
            name = entry["name"]
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("public parameter names must be nonempty and unique")
            names.add(name)
            element = dtype(entry["dtype"])
            if entry["kind"] == "scalar":
                parameters.append(ScalarParameter(position, name, element))
                continue
            if entry["kind"] != "view":
                raise ValueError("public interface parameter must be a view or scalar")
            shape, dimensions = tuple(entry["shape"]), tuple(entry["dimensions"])
            if any(type(extent) is not int or extent < -1 for extent in shape):
                raise ValueError("public shape requires static extents or dynamic -1")
            if len(dimensions) != len(shape) or any(type(identity) is not int for identity in dimensions):
                raise ValueError("public dimension identities must match view rank")
            strides = []
            for constraint in entry["strides"]:
                if constraint is None or type(constraint) is int:
                    strides.append(constraint)
                elif isinstance(constraint, dict) and constraint.get("kind") == "stride_symbol":
                    symbol = constraint["symbol"]
                    if not isinstance(symbol, str) or not symbol:
                        raise ValueError("stride symbol must be a nonempty string")
                    strides.append(StrideSymbol(symbol))
                else:
                    raise ValueError("unsupported public stride constraint")
            if len(strides) != len(shape):
                raise ValueError("public stride constraints must match view rank")
            if type(entry["access"]) is not int or type(entry["noalias"]) is not bool:
                raise TypeError("public access and noalias require integer and bool values")
            alias = entry["alias"]
            if not isinstance(alias, str) or (alias and entry["noalias"]):
                raise ValueError("alias and noalias are mutually exclusive")
            parameters.append(ViewParameter(position, name, element, shape, dimensions,
                                            tuple(strides), ViewKind(entry["access"]),
                                            alias, entry["noalias"]))
        return cls(tuple(parameters))

    @property
    def views(self) -> tuple[ViewParameter, ...]:
        return tuple(parameter for parameter in self.parameters if isinstance(parameter, ViewParameter))

    @property
    def inputs(self) -> tuple[ScalarParameter | ViewParameter, ...]:
        return tuple(parameter for parameter in self.parameters
                     if not isinstance(parameter, ViewParameter) or not parameter.output)

    @property
    def outputs(self) -> tuple[ViewParameter, ...]:
        """Declared Out buffers; an InOut remains a supplied input."""
        return tuple(view for view in self.views if view.output)

    @property
    def mutable_inputs(self) -> tuple[ViewParameter, ...]:
        return tuple(view for view in self.views if view.access is ViewKind.INOUT)

    @property
    def alias_checks(self) -> tuple[AliasCheck, ...]:
        return tuple(AliasCheck(left.position, right.position, left.writable or right.writable,
                                left.noalias or right.noalias, bool(left.alias and left.alias == right.alias))
                     for index, left in enumerate(self.views) for right in self.views[index + 1:]
                     if left.writable or right.writable or left.noalias or right.noalias or
                     (left.alias and left.alias == right.alias))

    def explicit_arguments(self, arguments: tuple, outputs: tuple) -> tuple:
        if len(arguments) != len(self.inputs):
            raise TypeError(f"expected {len(self.inputs)} input arguments, got {len(arguments)}")
        if len(outputs) != len(self.outputs):
            raise TypeError(f"expected {len(self.outputs)} explicit output buffers, got {len(outputs)}")
        supplied, allocated = iter(arguments), iter(outputs)
        return tuple(next(allocated) if isinstance(parameter, ViewParameter) and parameter.output
                     else next(supplied) for parameter in self.parameters)

    def binding_relations(self, *, explicit_outputs: bool) -> BindingRelations:
        supplied = self.parameters if explicit_outputs else self.inputs
        ordered = tuple(parameter for parameter in supplied if isinstance(parameter, ViewParameter))
        if not explicit_outputs:
            ordered += self.outputs
        # Static declarations constrain the entire shared dimension, including
        # a dynamic Out that precedes its statically sized sibling in the ABI.
        static_extents: dict[int, int] = {}
        allocation_errors = []
        for view in self.views:
            for extent, identity in zip(view.shape, view.dimensions, strict=True):
                if extent >= 0 and identity > 0:
                    known = static_extents.setdefault(identity, extent)
                    if known != extent and not explicit_outputs:
                        allocation_errors.append(
                            f"dimension {identity} has conflicting static extents {known} and {extent}")
        dimensions: dict[int, ViewAxis] = {}
        stride_symbols: dict[str, ViewAxis] = {}
        bindings = []
        for view in ordered:
            shape_checks, stride_checks, output_shape = [], [], []
            earlier_dimensions = dict(dimensions)
            for axis, (extent, identity) in enumerate(zip(view.shape, view.dimensions, strict=True)):
                reference = dimensions.get(identity) if identity > 0 else None
                known = extent if extent >= 0 else static_extents.get(identity)
                output_shape.append(known if known is not None else earlier_dimensions.get(identity))
                if extent >= 0:
                    shape_checks.append((axis, extent))
                elif known is not None:
                    shape_checks.append((axis, known))
                if reference is not None:
                    shape_checks.append((axis, reference))
                elif identity > 0:
                    dimensions[identity] = ViewAxis(view.position, axis)
            for axis, constraint in enumerate(view.strides):
                if isinstance(constraint, StrideSymbol):
                    reference = stride_symbols.get(constraint.name)
                    if reference is not None:
                        stride_checks.append((axis, reference))
                    else:
                        stride_symbols[constraint.name] = ViewAxis(view.position, axis)
                elif constraint is not None:
                    stride_checks.append((axis, constraint))
            bindings.append(ViewBinding(view, tuple(shape_checks), tuple(stride_checks), tuple(output_shape)))
        if not explicit_outputs:
            for binding in bindings:
                if binding.parameter.output:
                    for axis, extent in enumerate(binding.output_shape):
                        if extent is None:
                            allocation_errors.append(
                                f"{binding.parameter.name}.shape[{axis}] has no extent supplied by inputs "
                                "or fixed by the public shape declarations")
            for check in self.alias_checks:
                left, right = self.parameters[check.left], self.parameters[check.right]
                if check.same_allocation and (left.output or right.output):
                    allocation_errors.append(
                        f"{left.name} and {right.name} must share an allocation; independently allocated "
                        "outputs cannot establish that relation")
        return BindingRelations(supplied, tuple(bindings), tuple(dimensions.items()), tuple(allocation_errors))
