"""Host staging over public provider buffers; source generation needs no SDK."""

from dataclasses import dataclass
from math import prod
import sys

import numpy as np

from intent.language.dtypes import DTYPES, dtype as intent_dtype
from intent.runtime.torch import torch_dtype

from .inputs import Array, empty, from_bytes


def is_tensor(value):
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(value, torch.Tensor)


def element_strides(value):
    return tuple(value.stride()) if is_tensor(value) else tuple(value.strides)


def _native_classes():
    from intent.runtime.bangc import DeviceBuffer, DeviceView
    from intent.runtime.weft import Buffer
    return Buffer, DeviceBuffer, DeviceView


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
    Buffer, DeviceBuffer, _ = _native_classes()
    if isinstance(value, Buffer):
        if value.nbytes:
            from_bytes(value.storage, value.shape, value.dtype).data.fill(scalar)
        return
    if isinstance(value, DeviceBuffer):
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

    def __init__(self, target, device):
        from intent.targets.bangc import BangCTarget
        from intent.targets.weft import WeftTarget
        self.target, self.device = target, device
        self.kind = "bangc" if isinstance(target, BangCTarget) else (
            "weft" if isinstance(target, WeftTarget) else "torch")
        self.allocations = {}
        self.owned = []
        self.resetters = {}
        self.produced = set()

    def begin(self):
        self.allocations.clear()
        self.resetters.clear()
        self.produced.clear()

    def workspace(self, reference, shape, *, dtype):
        Buffer, DeviceBuffer, _ = _native_classes()
        if self.kind == "weft":
            value = Buffer.empty(tuple(shape), dtype)
        elif self.kind == "bangc":
            value = DeviceBuffer(tuple(shape), dtype, device=self.target.device,
                                 neuware=str(self.target.neuware))
            self.owned.append(value)
        else:
            import torch
            value = torch.empty(shape, dtype=torch_dtype(intent_dtype(dtype)), device=self.device)
        self.produced.add(id(value))
        return value

    def retain_outputs(self, value):
        if isinstance(value, (tuple, list)):
            for item in value:
                self.retain_outputs(item)
            return
        _, DeviceBuffer, _ = _native_classes()
        self.produced.add(id(value))
        if isinstance(value, DeviceBuffer) and all(value is not owner for owner in self.owned):
            self.owned.append(value)

    def _preserve_record(self, record):
        key = ("staged", id(record))
        if key in self.resetters:
            return
        if self.kind == "torch":
            saved = record.native.clone()
            self.resetters[key] = lambda: record.native.copy_(saved)
        else:
            saved = record.host.copy()
            if self.kind == "weft":
                self.resetters[key] = lambda: np.copyto(record.host, saved)
            else:
                self.resetters[key] = lambda: record.native.copy_from_host(saved)

    def _preserve_native(self, value):
        # Outputs and explicit workspace are initialized by the preceding
        # producer/reset in the same complete author call, not by old contents.
        if id(value) in self.produced:
            return
        Buffer, DeviceBuffer, DeviceView = _native_classes()
        if isinstance(value, Buffer):
            key = ("weft", value.allocation, value.allocation_end)
            if key not in self.resetters:
                owner = memoryview(value.storage.obj).cast("B")
                saved = bytes(owner)
                def restore():
                    owner[:] = saved
                self.resetters[key] = restore
        elif isinstance(value, (DeviceBuffer, DeviceView)):
            owner = value.owner if isinstance(value, DeviceView) else value
            key = ("bangc", id(owner))
            if key not in self.resetters:
                saved = owner.to_host()
                self.resetters[key] = lambda: owner.copy_from_host(saved)
        elif is_tensor(value):
            import torch
            storage = value.untyped_storage()
            key = ("torch", str(value.device), storage.data_ptr(), storage.nbytes())
            if key not in self.resetters:
                raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(
                    storage, 0, (storage.nbytes(),), (1,))
                saved = raw.clone()
                self.resetters[key] = lambda: raw.copy_(saved)

    def input(self, value, *, element, writable=False, preserve=False):
        Buffer, DeviceBuffer, DeviceView = _native_classes()
        if isinstance(value, (Buffer, DeviceBuffer, DeviceView)) or is_tensor(value):
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
        if self.kind == "weft" and strides != contiguous:
            raise NotImplementedError("the public Weft Buffer requires canonical contiguous element strides")
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
            if self.kind != "weft" and host.nbytes:
                begin, end = host.ctypes.data, host.ctypes.data + host.nbytes
                for existing in self.allocations.values():
                    other = existing.host
                    if other.nbytes and begin < other.ctypes.data + other.nbytes and other.ctypes.data < end:
                        raise NotImplementedError("aliased views must share the same complete host allocation")
            if self.kind == "bangc":
                from intent.runtime.bangc.buffer import ELEMENT_BYTES
                if element not in ELEMENT_BYTES:
                    raise NotImplementedError(f"BANG C buffers do not support storage dtype {element}")
                native = DeviceBuffer.from_host(host, shape=(host.nbytes // width,), dtype=element,
                    device=self.target.device, neuware=str(self.target.neuware))
                self.owned.append(native)
            elif self.kind == "weft":
                native = memoryview(host)
            else:
                import torch
                # The view reinterprets stored bytes, without converting BF16/FP8.
                native = (torch.frombuffer(host, dtype=torch.uint8) if host.nbytes else
                          torch.empty(0, dtype=torch.uint8)).to(self.device)
            record = _Allocation(host, element, native)
            self.allocations[key] = record
        elif record.element != element:
            raise NotImplementedError("one staged allocation cannot have multiple logical storage dtypes")
        record.writable |= writable
        if preserve:
            self._preserve_record(record)
        if self.kind == "bangc":
            return record.native.view(tuple(value.shape), strides, offset=offset // width)
        if self.kind == "weft":
            return Buffer(record.native[offset:offset + value.nbytes], shape=value.shape, dtype=element)
        return record.native.view(torch_dtype(intent_dtype(element))).as_strided(
            value.shape, strides, storage_offset=offset // width)

    def finish(self):
        for record in self.allocations.values():
            if not record.writable or not record.host.nbytes or self.kind == "weft":
                continue
            if self.kind == "bangc":
                data = np.frombuffer(record.native.to_host(), dtype=np.uint8)
            else:
                data = record.native.detach().cpu().numpy()
            np.copyto(record.host, data)

    def close(self):
        for owner in self.owned:
            owner.close()
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
    Buffer, DeviceBuffer, DeviceView = _native_classes()
    if not isinstance(value, (Buffer, DeviceBuffer, DeviceView)):
        return value
    if not contents:
        return Array.metadata(value.shape, value.dtype)
    storage = value.storage if isinstance(value, Buffer) else value.to_host()
    return from_bytes(storage, value.shape, value.dtype)
