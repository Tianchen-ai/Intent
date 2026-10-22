"""Select an execution device from actual arguments, before native compilation."""
from __future__ import annotations

from dataclasses import dataclass
import os

from .provider import provider
from .registry import make_target


@dataclass(frozen=True)
class LocalDevice:
    kind: str
    index: int | None = None

    def __str__(self):
        return self.kind if self.index is None else f"{self.kind}:{self.index}"


def _device(value, kind):
    if isinstance(value, int) and not isinstance(value, bool):
        result = LocalDevice(kind, value)
    else:
        parts = str(value).split(":")
        if len(parts) > 2 or not parts[0]:
            raise ValueError(f"invalid local device: {value!r}")
        result = LocalDevice(parts[0], int(parts[1]) if len(parts) == 2 else None)
    if result.kind not in {"cuda", "cpu", "mlu"} or (result.index is not None and result.index < 0):
        raise ValueError(f"unsupported local device: {value!r}")
    if result.kind == "cpu" and result.index is not None:
        raise ValueError("CPU selection uses the process affinity, not a device ordinal")
    return result


def select_device(backend: str, arguments: tuple, explicit=None) -> LocalDevice:
    family = provider(backend).family
    expected = {"gpu": "cuda", "cpu": "cpu", "dsa": "mlu"}[family]
    observed = set()
    for value in arguments:
        if not hasattr(value, "shape"):
            continue
        device = getattr(value, "device", "cpu")
        observed.add(_device(device, expected))
    if len(observed) > 1:
        raise ValueError(f"JIT arguments span different devices: {sorted(map(str, observed))}")
    selected = _device(explicit, expected) if explicit is not None else next(iter(observed), None)
    if selected is None:
        if expected != "cpu":
            raise ValueError("JIT requires a device-bearing input or an explicit local device")
        selected = LocalDevice("cpu")
    if selected.kind != expected:
        raise ValueError(f"backend {backend!r} executes on {expected}, but arguments select {selected}")
    if expected != "cpu" and selected.index is None:
        if observed and next(iter(observed)).index is not None:
            selected = next(iter(observed))
        else:
            raise ValueError(f"specify a concrete {expected} device ordinal")
    if observed and selected != next(iter(observed)):
        raise ValueError(f"explicit device {selected} conflicts with the input device {next(iter(observed))}")
    return selected


def execution_key(device: LocalDevice):
    return (device, tuple(sorted(os.sched_getaffinity(0))) if device.kind == "cpu" else ())


def resolve_local(backend: str, device: LocalDevice, *, workers=None, backend_options=None):
    options = dict(backend_options or {})
    allowed = {
        "triton": set(), "cutile": set(),
        "mojo": {"compiler", "build_threads", "private_bytes"},
        "weft": {"compiler", "cc", "cflags", "private_bytes"},
        "bangc": {"neuware", "compiler"},
    }[backend]
    if set(options) - allowed:
        raise ValueError(f"unsupported local {backend} options: {sorted(set(options) - allowed)}")
    if device.kind == "cpu":
        available = len(os.sched_getaffinity(0))
        if workers is None:
            workers = min(8, available)
        if type(workers) is not int or not 0 < workers <= available:
            raise ValueError(f"workers must be between 1 and the {available} CPUs allowed by affinity")
        options["workers"] = workers
    elif workers is not None:
        raise ValueError("workers is a CPU execution budget")
    if backend == "weft":
        from .weft import native_target
        return native_target(**options)
    if device.kind != "cpu":
        options["device"] = device.index
    return make_target(backend, options).resolve()
