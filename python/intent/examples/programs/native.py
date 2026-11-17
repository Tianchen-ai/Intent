"""Host staging over public provider buffers; source generation needs no SDK."""

from math import prod
import sys

import numpy as np

from intent.language.dtypes import DTYPES, dtype as intent_dtype
from intent.integrations.torch import torch_dtype
from intent.runtime.host_buffers import HostBuffers as _HostBuffers

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


class HostBuffers(_HostBuffers):
    """Adapt example Arrays to the public explicit host-transfer scope."""

    def input(self, value, *, element, writable=False, preserve=False):
        if isinstance(value, Array):
            if value.dtype != element:
                raise TypeError(f"expected {element} input, got {value.dtype}")
            if value.data is None:
                raise RuntimeError("native execution cannot consume a shape-only source result")
            value = value.data
        return super().input(value, element=element, writable=writable, preserve=preserve)


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
