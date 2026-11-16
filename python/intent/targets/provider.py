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


def unspecialized_call(target, definition, arguments, outputs):
    """A provider whose source ABI retains runtime shapes and strides."""
    return target, ()


@dataclass(frozen=True, slots=True)
class LocalExecution:
    """The local device and binding contract owned by one source provider."""

    device_kind: str
    indexed: bool
    options: frozenset[str]
    resolve: Callable[..., Any]
    execution_key: Callable[[Any], object]
    specialize: Callable[..., tuple[Any, object]] = unspecialized_call


@dataclass(frozen=True, slots=True)
class Provider:
    family: str
    target: type
    read_facts: Callable[[str, dict, Any], Any]
    bind: Callable[[Any, Any], Any] | None
    environment_checks: Callable[[Any], tuple[EnvironmentCheck, ...]]
    local: LocalExecution
    execution_model: str
    environment_scope: str | None = None


def provider(name: str) -> Provider:
    from .components import COMPONENTS

    if name not in COMPONENTS:
        raise ValueError(f"Unknown backend {name!r}; choose from {', '.join(COMPONENTS)}")
    return import_module(COMPONENTS[name]).PROVIDER
