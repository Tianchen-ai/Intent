"""Reference checks over the GPU host bindings shared by provider recipes."""
from ..contract import name_field, names_field


def optional_name(value, field):
    return None if value is None else name_field(value, field)


def require_references(expressions, interface):
    declared = set(interface.by_id) | {parameter.name for parameter in interface.configuration_space.parameters}
    for expression in expressions:
        missing = expression.references - declared
        if missing:
            raise ValueError(f"provider expression references undeclared bindings: {sorted(missing, key=str)}")


def view_binding(identity, interface):
    if type(identity) is not int or identity not in {view.id for view in interface.views}:
        raise ValueError("provider recipe must reference a declared GPU view")
    return interface.by_id[identity]


def native_names(names, available, field):
    names = names_field(names, field)
    missing = set(names) - set(available)
    if missing:
        raise ValueError(f"{field} references undeclared bindings: {sorted(missing)}")
    return names
