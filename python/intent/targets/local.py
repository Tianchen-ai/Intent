"""Select an execution device from actual arguments, before native compilation."""
from __future__ import annotations

from dataclasses import dataclass

from .provider import provider


@dataclass(frozen=True)
class LocalDevice:
    kind: str
    index: int | None = None

    def __str__(self):
        return self.kind if self.index is None else f"{self.kind}:{self.index}"


def _device(value, execution):
    if isinstance(value, int) and not isinstance(value, bool):
        result = LocalDevice(execution.device_kind, value)
    else:
        parts = str(value).split(":")
        if len(parts) > 2 or not parts[0]:
            raise ValueError(f"invalid local device: {value!r}")
        result = LocalDevice(parts[0], int(parts[1]) if len(parts) == 2 else None)
    if result.kind != execution.device_kind or (result.index is not None and result.index < 0):
        raise ValueError(f"unsupported local device: {value!r}")
    if not execution.indexed and result.index is not None:
        raise ValueError(f"{result.kind} execution does not use a device ordinal")
    return result


def select_device(backend: str, arguments: tuple, explicit=None) -> LocalDevice:
    execution = provider(backend).local
    expected = execution.device_kind
    observed = set()
    for value in arguments:
        if not hasattr(value, "shape"):
            continue
        device = getattr(value, "device", "cpu")
        observed.add(_device(device, execution))
    if len(observed) > 1:
        raise ValueError(f"JIT arguments span different devices: {sorted(map(str, observed))}")
    selected = _device(explicit, execution) if explicit is not None else next(iter(observed), None)
    if selected is None:
        if execution.indexed:
            raise ValueError("JIT requires a device-bearing input or an explicit local device")
        selected = LocalDevice(expected)
    if selected.kind != expected:
        raise ValueError(f"backend {backend!r} executes on {expected}, but arguments select {selected}")
    if execution.indexed and selected.index is None:
        if observed and next(iter(observed)).index is not None:
            selected = next(iter(observed))
        else:
            raise ValueError(f"specify a concrete {expected} device ordinal")
    if observed and selected != next(iter(observed)):
        raise ValueError(f"explicit device {selected} conflicts with the input device {next(iter(observed))}")
    return selected


def resolve_local(backend: str, device: LocalDevice, *, workers=None, backend_options=None):
    execution = provider(backend).local
    if device.kind != execution.device_kind:
        raise ValueError(f"backend {backend!r} cannot execute on {device}")
    options = dict(backend_options or {})
    if set(options) - execution.options:
        raise ValueError(f"unsupported local {backend} options: {sorted(set(options) - execution.options)}")
    return execution.resolve(device, workers=workers, **options)
