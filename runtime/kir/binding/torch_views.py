def accepts(value):
    return any(
        cls.__module__.split(".")[0] == "torch" and cls.__name__ == "Tensor"
        for cls in type(value).__mro__
    )


def describe(a):
    import torch

    return (
        {
            torch.float32: "f32",
            torch.float16: "f16",
            torch.bfloat16: "bf16",
            torch.int32: "i32",
        }.get(a.dtype),
        tuple(a.shape),
        tuple(a.stride()),
        a.device,
        a.untyped_storage(),
        a.data_ptr(),
        True,
    )


def allocate(shape, dtype, device):
    import torch

    return torch.empty(
        shape,
        dtype={
            "f32": torch.float32,
            "f16": torch.float16,
            "bf16": torch.bfloat16,
            "i32": torch.int32,
        }[dtype],
        device=device,
    )


def shadow_group(views, start, end):
    import torch

    device = views[0].device
    backing = torch.empty(end - start + 256, dtype=torch.uint8, device=device)
    pad = (start - backing.data_ptr()) % 256
    clones, slices = [], []
    cursor = start
    for a in views:  # sorted by storage start by storage_groups
        base, size = storage_bounds(a)
        if base + size > cursor:
            count = base + size - cursor
            raw = torch.empty(0, dtype=torch.uint8, device=device).set_(
                a.untyped_storage(), cursor - base, (count,), (1,))
            slices.append((backing.narrow(0, pad + cursor - start, count), raw))
            cursor += count
        offset = pad + a.data_ptr() - start if a.numel() else pad
        if offset % a.element_size():
            raise ValueError("unaligned alias cannot participate in tuning")
        clones.append(torch.empty(0, dtype=a.dtype, device=device).set_(
            backing.untyped_storage(), offset // a.element_size(), a.shape, a.stride()))

    def restore():
        for destination, source in slices:
            destination.copy_(source)
    return clones, restore


def storage_bounds(v):
    root = v.untyped_storage()
    return root.data_ptr(), root.nbytes()


def host_shadow(v, backing, start):
    import torch

    if not backing.size:
        return torch.empty_strided(v.shape, v.stride(), dtype=v.dtype)
    offset = v.data_ptr() - start
    byte_tensor = torch.from_numpy(backing)
    if offset % v.element_size():
        raise ValueError("misaligned typed alias in trial group")
    return torch.empty(0, dtype=v.dtype).set_(
        byte_tensor.untyped_storage(), offset // v.element_size(), v.shape, v.stride()
    )

def project(value,shape,strides,offset):
    return value.as_strided(shape,strides,value.storage_offset()+offset)
