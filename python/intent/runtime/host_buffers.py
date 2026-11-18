"""Explicit host transfers and ownership for an authored invocation sequence."""
from dataclasses import dataclass
import sys

from intent.language.dtypes import DTYPES
from .storage_types import numpy_dtype, numpy_element


def _host_view(value, element):
    import numpy as np
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        from intent.integrations.torch import torch_dtype
        if value.device.type != "cpu" or value.layout != torch.strided or value.is_conj() or value.is_neg():
            raise ValueError("explicit host staging requires ordinary CPU tensor storage")
        if element is None:
            element = next((name for name, dtype in DTYPES.items()
                            if name != "index" and torch_dtype(dtype) == value.dtype), None)
        if element is None or value.dtype != torch_dtype(DTYPES[element]):
            raise TypeError("host tensor storage dtype disagrees with the requested element type")
        backing = value.untyped_storage()
        host = torch.empty(0, dtype=torch.uint8).set_(backing, 0, (backing.nbytes(),), (1,)).numpy()
        return host, element, tuple(value.shape), tuple(value.stride()), value.storage_offset()
    if not isinstance(value, np.ndarray):
        raise TypeError("host staging requires a NumPy array or a CPU tensor")
    if element is None:
        element = numpy_element(value.dtype)
    if element is None or value.dtype != numpy_dtype(element):
        raise TypeError("host array storage dtype disagrees with the requested element type")
    owner = value
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    if not (owner.flags.c_contiguous or owner.flags.f_contiguous):
        raise NotImplementedError("host views require a contiguous owning allocation")
    if owner.base is not None:
        owner = np.frombuffer(owner.base, dtype=np.uint8)
    host = owner.ravel(order="K").view(np.uint8)
    width = value.itemsize
    offset = value.ctypes.data - host.ctypes.data
    if offset % width or host.nbytes % width or any(stride % width for stride in value.strides):
        raise ValueError("native staging requires whole storage elements")
    return host, element, value.shape, tuple(stride // width for stride in value.strides), offset // width


@dataclass
class _Allocation:
    host: object
    element: str
    native: object
    writable: bool = False


class HostBuffers:
    """An explicitly requested transfer scope, retaining shared storage and views.

    This is separate from JIT argument binding: ordinary calls never move inputs
    implicitly. Inputs, explicit workspace and outputs stay alive until close().
    finish() copies writable staged storage back; resetters restore trial state.
    """

    def __init__(self, backend, device, backend_options=None):
        from intent.targets.local import select_device
        from intent.targets.provider import provider
        selected = select_device(backend, (), device)
        self.storage = provider(backend).local.host_storage(selected, backend_options or {})
        self.allocations, self.resetters, self.produced = {}, {}, set()
        self.owned = []

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
        if isinstance(value, dict):
            value = tuple(value.values())
        if isinstance(value, (tuple, list)):
            for item in value:
                self.retain_outputs(item, owned=owned)
            return
        self.produced.add(id(value))
        if owned and self.storage.accepts(value) and all(value is not owner for owner in self.owned):
            self.owned.append(value)

    def _preserve_record(self, record):
        key = ("staged", id(record))
        if key not in self.resetters:
            _, self.resetters[key] = self.storage.snapshot(record.native)

    def _preserve_native(self, value):
        if id(value) not in self.produced:
            key, restore = self.storage.snapshot(value)
            self.resetters.setdefault(("native", key), restore)

    def input(self, value, *, element=None, writable=False, preserve=False):
        if self.storage.accepts(value):
            if preserve:
                self._preserve_native(value)
            return value
        host, element, shape, strides, offset = _host_view(value, element)
        key = (host.ctypes.data, host.nbytes) if host.nbytes else ("empty", id(value))
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
        return self.storage.view(record.native, shape, strides, offset, element)

    def reset(self):
        for restore in self.resetters.values():
            restore()

    def finish(self):
        import numpy as np
        for record in self.allocations.values():
            if record.writable and record.host.nbytes and self.storage.copies_host:
                np.copyto(record.host, self.storage.download(record.native))

    def close(self):
        for owner in self.owned:
            self.storage.release(owner)
        self.owned.clear()
        self.allocations.clear()
        self.resetters.clear()
        self.produced.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
