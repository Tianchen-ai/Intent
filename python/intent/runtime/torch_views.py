from __future__ import annotations

from .interface import ViewParameter
from .invocation import ViewFacts
from .torch import torch_dtype


def _byte_span(pointer: int, shape: tuple, strides: tuple, size: int) -> tuple[int, int]:
    if any(extent == 0 for extent in shape):
        return pointer, pointer
    low = high = 0
    for extent, stride in zip(shape, strides):
        displacement = (extent - 1) * stride
        low += min(0, displacement)
        high += max(0, displacement)
    return pointer + low * size, pointer + (high + 1) * size


def view_byte_span(view) -> tuple[int, int]:
    return _byte_span(view.data_ptr(), tuple(view.shape), tuple(view.stride()), view.element_size())


def observe_view(parameter: ViewParameter, value, *, device, abstract: bool = False,
                 expected_dtype=None) -> ViewFacts:
    """Observe a tensor's public geometry; concrete binding also observes storage.

    Layout, alignment and writable overlap requirements belong to the compiled
    entry, independently of these observations and the author's interface.
    """
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{parameter.name} must be a torch.Tensor")
    expected = torch_dtype(parameter.dtype) if expected_dtype is None else expected_dtype
    dtype = value.dtype
    if dtype != expected:
        raise TypeError(f"{parameter.name} must have dtype {expected}, got {dtype}")
    actual_device = value.device
    if actual_device != device:
        raise ValueError(f"{parameter.name} must be on {device}, got {actual_device}")
    shape, strides = tuple(value.shape), tuple(value.stride())
    if len(shape) != len(parameter.shape):
        raise ValueError(f"{parameter.name} must have rank {len(parameter.shape)}, got {len(shape)}")
    if abstract:
        return ViewFacts(shape, strides, None, None, None, None, dtype, None, None, None)
    storage = value.untyped_storage()
    allocation = storage.data_ptr()
    byte_count = storage.nbytes()
    pointer = value.data_ptr()
    # Independently wrapped external storage can have distinct StorageImpls
    # over the same allocation. Empty allocations instead need a real handle:
    # their zero pointer does not identify shared ownership.
    identity = (actual_device, allocation) if byte_count else (actual_device, "empty", storage._cdata)
    begin, end = _byte_span(pointer, shape, strides, value.element_size())
    return ViewFacts(shape, strides, pointer, allocation, allocation + byte_count,
                     value.storage_offset(), dtype, begin, end, identity)


def allocate_output(parameter: ViewParameter, shape: tuple, *, device,
                    abstract: bool = False, expected_dtype=None) -> tuple[object, ViewFacts]:
    import torch

    dtype = torch_dtype(parameter.dtype) if expected_dtype is None else expected_dtype
    controlled = (not abstract and torch._C._len_torch_dispatch_stack() == 0
                  and torch._ops._len_torch_dispatch_stack_pre_dispatch() == 0
                  and torch._C._len_torch_function_stack() == 0)
    value = torch.empty(shape, dtype=dtype, device=device)
    if not controlled or type(value) is not torch.Tensor or any(extent == 0 for extent in shape):
        return value, observe_view(parameter, value, device=device, abstract=abstract,
                                   expected_dtype=dtype)
    # This allocation's geometry and storage belong to this call. Reuse the
    # facts established by empty, never a previous tensor's pointer or identity.
    strides = [0] * len(shape)
    elements = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = elements
        elements *= shape[axis]
    pointer = value.data_ptr()
    end = pointer + elements * ((parameter.dtype.bits + 7) // 8)
    facts = ViewFacts(shape, tuple(strides), pointer, pointer, end, 0, dtype,
                      pointer, end, (value.device, pointer))
    return value, facts


def check_abstract_relation(actual, expected, message: str) -> None:
    import torch

    torch._check(actual == expected, lambda: message)
