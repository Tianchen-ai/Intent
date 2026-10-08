"""Provider ownership without importing a device SDK or probing the host."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True, slots=True)
class EnvironmentCheck:
    name: str
    category: str
    action: Callable[[], object]


@dataclass(frozen=True, slots=True)
class Provider:
    family: str
    target: type
    read_facts: Callable[[str, dict, Any], Any]
    bind: Callable[[Any, Any], Any] | None
    environment_checks: Callable[[Any], tuple[EnvironmentCheck, ...]]
    environment_scope: str | None = None


def provider(name: str) -> Provider:
    # The installation declarations also own the fixed supported-provider list.
    # SDK imports belong inside binding or inspection, never adapter discovery.
    from intent.tools.backends import BACKENDS

    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; choose from {', '.join(BACKENDS)}")
    return import_module(f"intent.targets.{name}").PROVIDER
