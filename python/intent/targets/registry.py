"""Provider declarations shared by target resolution and installation tools."""

from importlib import import_module
from .components import COMPONENTS

BACKENDS = {name: import_module(module).INSTALLATION for name, module in COMPONENTS.items()}


def backend(name: str) -> dict:
    from intent.targets.provider import provider
    adapter = provider(name)
    return {**BACKENDS[name], "target": adapter.target.__name__,
            "execution_model": adapter.execution_model, "device_kind": adapter.local.device_kind}


def make_target(name: str, options: dict, *, facts: dict | None = None):
    from intent.targets.provider import provider
    from intent.targets.specification import read_compilation_target

    if facts is not None:
        if options:
            raise ValueError("explicit compiler facts do not accept local runtime target options")
        return read_compilation_target(name, facts)
    return provider(name).target(**options)
