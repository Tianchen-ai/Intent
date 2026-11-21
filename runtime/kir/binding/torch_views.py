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


def overlaps(a, b):
    return (
        not accepts(b)
        or a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
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


def storage_key(value):
    return value.device, value.untyped_storage()._cdata


def shadow_group(views):
    import torch

    v = views[0]
    store = v.untyped_storage()
    raw = torch.empty(0, dtype=torch.uint8, device=v.device).set_(
        store, 0, (store.nbytes(),), (1,)
    )
    shadow = torch.empty_like(raw)
    clone = [
        torch.empty(0, device=a.device, dtype=a.dtype).set_(
            shadow.untyped_storage(), a.storage_offset(), a.shape, a.stride()
        )
        for a in views
    ]
    return clone, lambda: shadow.copy_(raw)


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
