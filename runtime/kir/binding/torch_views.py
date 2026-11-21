def accepts(value):
    return any(
        cls.__module__.split(".")[0] == "torch" and cls.__name__ == "Tensor"
        for cls in type(value).__mro__
    )


def describe(a):
    import torch

    return (
        {torch.float32: "f32", torch.float16: "f16"}.get(a.dtype),
        tuple(a.shape),
        tuple(a.stride()),
        a.device,
        a.untyped_storage(),
        a.data_ptr(),
        True,
    )


def overlaps(a, b):
    return (
        not accepts(b)
        or a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    )


def allocate(shape, dtype, device):
    import torch

    return torch.empty(
        shape, dtype={"f32": torch.float32, "f16": torch.float16}[dtype], device=device
    )


def trial(a):
    import torch

    return torch.empty_strided(a.shape, a.stride(), dtype=a.dtype, device=a.device)
