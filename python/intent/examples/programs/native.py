"""Host staging over public provider buffers; source generation needs no SDK."""

from dataclasses import dataclass
from math import prod
import sys

import numpy as np

from intent.language.dtypes import DTYPES, dtype as intent_dtype
from intent.integrations.torch import torch_dtype

from .inputs import Array, empty, from_bytes


def is_tensor(value):
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def element_strides(value):
    return tuple(value.stride()) if is_tensor(value) else tuple(value.strides)


def empty_like(reference, shape, *, dtype):
    """The host workspace path used by compositions with ordinary artifacts."""
    if isinstance(reference, Array):
        return Array.metadata(shape, dtype) if reference.data is None else empty(shape, dtype)
    if is_tensor(reference):
        import torch
        return torch.empty(shape, dtype=torch_dtype(intent_dtype(dtype)), device=reference.device)
    raise TypeError("native compositions allocate workspace through their example kernel")


def fill(value, scalar):
    """Reset the actual workspace that the following kernels will consume."""
    if isinstance(value, Array):
        if value.data is None:
            raise RuntimeError("shape-only output has no writable contents")
        value.data.fill(scalar)
        return
    if is_tensor(value):
        value.fill_(scalar)
        return
    if callable(getattr(value, "copy_from_host", None)):
        host = empty(value.shape, value.dtype)
        host.data.fill(scalar)
        value.copy_from_host(host.data.view(np.uint8).reshape(-1))
        return
    raise TypeError("workspace reset requires an owned host or public native buffer")


@dataclass
class _Allocation:
    host: np.ndarray
    element: str
    native: object
    writable: bool = False


class HostBuffers:
    """One complete call's storage owners, including aliased views and InOut.

    Weft borrows CPU bytes. BANG C and CUDA upload each allocation once; native
    intermediates stay resident between composed kernels. Host-visible mutable
    state is copied back when the complete call finishes, within Context.call.
    """

    def __init__(self, backend, device, backend_options):
        from intent.targets.local import select_device
        from intent.targets.provider import provider
        selected = select_device(backend, (), device)
        self.storage = provider(backend).local.host_storage(selected, backend_options)
        self.allocations = {}
        self.owned = []
        self.resetters = {}
        self.produced = set()

    def begin(self):
        self.allocations.clear()
        self.resetters.clear()
        self.produced.clear()

    def workspace(self, reference, shape, *, dtype):
        value = self.storage.allocate(shape, dtype)
        self.owned.append(value)
        self.produced.add(id(value))
        return value

    def retain_outputs(self, value, *, owned=True):
        if isinstance(value, (tuple, list)):
            for item in value:
                self.retain_outputs(item, owned=owned)
            return
        self.produced.add(id(value))
        if owned and self.storage.accepts(value) and all(value is not owner for owner in self.owned):
            self.owned.append(value)

    def _preserve_record(self, record):
        key = ("staged", id(record))
        if key in self.resetters:
            return
        _, restore = self.storage.snapshot(record.native)
        self.resetters[key] = restore

    def _preserve_native(self, value):
        # Outputs and explicit workspace are initialized by the preceding
        # producer/reset in the same complete author call, not by old contents.
        if id(value) in self.produced:
            return
        key, restore = self.storage.snapshot(value)
        self.resetters.setdefault(("native", key), restore)

    def input(self, value, *, element, writable=False, preserve=False):
        if self.storage.accepts(value):
            # Provider handles retain their original ownership and are validated
            # by that provider's ordinary invocation binder.
            if preserve:
                self._preserve_native(value)
            return value
        if not isinstance(value, Array):
            raise TypeError("example views must be host Arrays or actual provider buffers")
        if value.dtype != element:
            raise TypeError(f"expected {element} input, got {value.dtype}")
        if value.data is None:
            raise RuntimeError("native execution cannot consume a shape-only source result")
        width = (intent_dtype(element).bits + 7) // 8
        strides = element_strides(value)
        contiguous = tuple(prod(value.shape[axis + 1:]) for axis in range(len(value.shape)))
        if self.storage.contiguous_views and strides != contiguous:
            raise NotImplementedError("this provider's public buffer requires contiguous element strides")
        owner = value.data
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        if not owner.flags.c_contiguous:
            raise NotImplementedError("example staging requires a contiguous owning host allocation")
        host = owner.view(np.uint8).reshape(-1)
        offset = value.data.ctypes.data - host.ctypes.data
        if offset % width or host.nbytes % width:
            raise ValueError("native staging requires whole storage elements")
        key = (host.ctypes.data, host.nbytes) if host.nbytes else ("empty", id(owner))
        record = self.allocations.get(key)
        if record is None:
            if self.storage.copies_host and host.nbytes:
                begin, end = host.ctypes.data, host.ctypes.data + host.nbytes
                for existing in self.allocations.values():
                    other = existing.host
                    if other.nbytes and begin < other.ctypes.data + other.nbytes and other.ctypes.data < end:
                        raise NotImplementedError("aliased views must share the same complete host allocation")
            native = self.storage.upload(host, element)
            self.owned.append(native)
            record = _Allocation(host, element, native)
            self.allocations[key] = record
        elif record.element != element:
            raise NotImplementedError("one staged allocation cannot have multiple logical storage dtypes")
        record.writable |= writable
        if preserve:
            self._preserve_record(record)
        return self.storage.view(record.native, value.shape, strides, offset // width, element)

    def finish(self):
        for record in self.allocations.values():
            if not record.writable or not record.host.nbytes or not self.storage.copies_host:
                continue
            data = self.storage.download(record.native)
            np.copyto(record.host, data)

    def close(self):
        for owner in self.owned:
            self.storage.release(owner)
        self.owned.clear()
        self.allocations.clear()
        self.resetters.clear()
        self.produced.clear()


def host_result(value, *, contents):
    """Return typed host results, or explicitly data-free metadata handles."""
    if isinstance(value, tuple):
        return tuple(host_result(item, contents=contents) for item in value)
    if isinstance(value, list):
        return [host_result(item, contents=contents) for item in value]
    if isinstance(value, dict):
        return {name: host_result(item, contents=contents) for name, item in value.items()}
    if isinstance(value, Array):
        return value if contents else Array.metadata(value.shape, value.dtype)
    if is_tensor(value):
        element = next((name for name, kind in DTYPES.items()
                        if name != "index" and torch_dtype(kind) == value.dtype), None)
        if element is None:
            raise NotImplementedError(f"host result cannot represent {value.dtype}")
        if not contents:
            return Array.metadata(tuple(value.shape), element)
        data = value.detach().cpu().contiguous().reshape(-1).view(sys.modules["torch"].uint8).numpy()
        return from_bytes(data, tuple(value.shape), element)
    if not callable(getattr(value, "to_host", None)):
        return value
    if not contents:
        return Array.metadata(value.shape, value.dtype)
    return from_bytes(value.to_host(), value.shape, value.dtype)
