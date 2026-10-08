from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .interface import AliasCheck, PublicInterface, ViewAxis, ViewParameter


@dataclass(slots=True)
class ViewFacts:
    """Fresh observations for one binding; abstract views have no address facts."""

    shape: tuple
    strides: tuple
    pointer: int | None
    allocation: int | None
    allocation_end: int | None
    offset: int | None
    dtype: object
    begin: int | None
    end: int | None
    allocation_identity: object


def invocation_result(outputs: tuple):
    """Return only declared Out buffers: None, one value, or an ordered tuple."""
    if not outputs:
        return None
    return outputs[0] if len(outputs) == 1 else outputs


@dataclass(slots=True)
class BoundPublicArguments:
    arguments: tuple
    outputs: tuple
    views: tuple[ViewFacts | None, ...]

    def result(self):
        return invocation_result(self.outputs)


def _check_relation(actual, expected, message: str) -> None:
    if actual != expected:
        raise ValueError(message)


def _check_alias_contract(check: AliasCheck, left: ViewFacts, right: ViewFacts,
                          parameters: str) -> None:
    if check.noalias_violation((left.allocation, left.allocation_end),
                               (right.allocation, right.allocation_end)):
        raise ValueError(f"{parameters} violate an author noalias allocation constraint")
    if check.allocation_violation(left.allocation_identity, right.allocation_identity):
        raise ValueError(f"{parameters} violate an author named allocation-alias constraint")


def build_invocation_binders(
    interface: PublicInterface, *,
    observe_view: Callable[[object, ViewParameter, object], ViewFacts],
    allocate_output: Callable[[object, ViewParameter, tuple], tuple[object, ViewFacts]],
    check_relation: Callable[[object, object, str], None] = _check_relation,
    check_view_geometry: Callable[[object, ViewParameter, ViewFacts], None] | None = None,
    check_view_storage: Callable[[object, ViewParameter, ViewFacts], None] | None = None,
    check_alias_requirements: Callable[[AliasCheck, ViewFacts, ViewFacts], None] | None = None,
    check_dimensions: Callable[[object, dict[int, int]], None] | None = None,
    abstract: bool = False,
) -> tuple[Callable[[object, tuple], BoundPublicArguments], ...]:
    """Compile allocating and explicit-output binders from one public contract.

    Callbacks observe the selected family's objects and enforce its actual entry
    requirements. Generated code retains only static schema and callbacks, never
    tensor facts or an invocation owner. Abstract binding checks the same shape
    and stride relations through its supplied assertion callback, including entry
    geometry requirements. Only concrete binding observes addresses and checks
    storage or allocation relations.
    """
    def tuple_expression(entries: Sequence[str]) -> str:
        return "(" + ", ".join(entries) + ("," if entries else "") + ")"

    result = []
    for explicit_outputs in (False, True):
        relations = interface.binding_relations(explicit_outputs=explicit_outputs)
        supplied = relations.supplied
        bindings = {binding.parameter.position: binding for binding in relations.views}
        namespace = {
            "_observe": observe_view, "_allocate": allocate_output,
            "_check_relation": check_relation, "_check_view_geometry": check_view_geometry,
            "_check_view_storage": check_view_storage,
            "_check_alias_contract": _check_alias_contract,
            "_check_alias_requirements": check_alias_requirements,
            "_check_dimensions": check_dimensions, "_BoundPublicArguments": BoundPublicArguments,
        }
        for parameter in interface.parameters:
            namespace[f"_p{parameter.position}"] = parameter
        lines = [
            "def bind(owner, arguments):",
            f"    if len(arguments) != {len(supplied)}:",
            f"        raise TypeError(f'expected {len(supplied)} runtime arguments, got {{len(arguments)}}')",
        ]
        if relations.allocation_errors:
            namespace["_require_output_allocation"] = relations.require_output_allocation
            lines.append("    _require_output_allocation()")
            code = compile("\n".join(lines) + "\n", "<intent.public.invocation>", "exec", dont_inherit=True)
            exec(code, namespace)
            result.append(namespace["bind"])
            continue

        def reference(value: int | ViewAxis, field: str) -> str:
            return f"f{value.parameter}.{field}[{value.axis}]" if isinstance(value, ViewAxis) else repr(value)

        def check_view(parameter: ViewParameter) -> None:
            binding = bindings[parameter.position]
            for field, checks in (("shape", binding.shape_checks), ("strides", binding.stride_checks)):
                for axis, expected in checks:
                    message = f"{parameter.name}.{field}[{axis}] violates the declared interface relation"
                    if parameter.output and not explicit_outputs and field == "strides":
                        message += "; provide explicit output buffers with the required layout"
                    lines.append(f"    _check_relation(f{parameter.position}.{field}[{axis}], "
                                 f"{reference(expected, field)}, {message!r})")
            if check_view_geometry is not None:
                lines.append(f"    _check_view_geometry(owner, _p{parameter.position}, f{parameter.position})")
            if not abstract and check_view_storage is not None:
                lines.append(f"    _check_view_storage(owner, _p{parameter.position}, f{parameter.position})")

        for position, parameter in enumerate(supplied):
            index = parameter.position
            lines.append(f"    v{index} = arguments[{position}]")
            if isinstance(parameter, ViewParameter):
                lines.append(f"    f{index} = _observe(owner, _p{index}, v{index})")
                check_view(parameter)
        if not abstract and check_dimensions is not None:
            positions = {parameter.position for parameter in supplied}
            dimensions = ", ".join(f"{identity}: {reference(value, 'shape')}"
                                   for identity, value in relations.dimensions
                                   if value.parameter in positions)
            lines.append(f"    _check_dimensions(owner, {{{dimensions}}})")
        if not explicit_outputs:
            for parameter in interface.outputs:
                extents = [reference(extent, "shape") for extent in bindings[parameter.position].output_shape]
                index = parameter.position
                lines.append(f"    v{index}, f{index} = _allocate(owner, _p{index}, {tuple_expression(extents)})")
                check_view(parameter)
        if not abstract:
            for index, check in enumerate(interface.alias_checks):
                namespace[f"_a{index}"] = check
                arguments = f"_a{index}, f{check.left}, f{check.right}"
                if check.noalias or check.same_allocation:
                    parameters = f"{interface.parameters[check.left].name} and {interface.parameters[check.right].name}"
                    lines.append(f"    _check_alias_contract({arguments}, {parameters!r})")
                if check_alias_requirements is not None and check.writable:
                    lines.append(f"    _check_alias_requirements({arguments})")
        arguments = tuple_expression([f"v{parameter.position}" for parameter in interface.parameters])
        outputs = tuple_expression([f"v{parameter.position}" for parameter in interface.outputs])
        facts = tuple_expression([f"f{parameter.position}" if isinstance(parameter, ViewParameter) else "None"
                                  for parameter in interface.parameters])
        lines.append(f"    return _BoundPublicArguments({arguments}, {outputs}, {facts})")
        code = compile("\n".join(lines) + "\n", "<intent.public.invocation>", "exec", dont_inherit=True)
        exec(code, namespace)
        result.append(namespace["bind"])
    return tuple(result)
