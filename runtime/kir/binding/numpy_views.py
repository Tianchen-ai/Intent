def accepts(value):
    return any(
        cls.__module__.split(".")[0] == "numpy" and cls.__name__ == "ndarray"
        for cls in type(value).__mro__
    )


def describe(a):
    import numpy as np

    dtype = {
        np.dtype("float16"): "f16",
        np.dtype("float32"): "f32",
        np.dtype("int32"): "i32",
    }.get(a.dtype)
    if any(s % a.itemsize for s in a.strides):
        raise ValueError("nonintegral element stride")
    owner = a
    while getattr(owner, "base", None) is not None:
        owner = owner.base
    return (
        "bf16" if a.dtype.name == "bfloat16" else dtype,
        a.shape,
        tuple(s // a.itemsize for s in a.strides),
        "cpu",
        owner,
        a.ctypes.data,
        a.flags.writeable,
    )


def overlaps(a, b):
    import numpy as np

    if accepts(b):
        return np.may_share_memory(a, b)
    return (
        True  # Cross-container writable aliases require a common allocation contract.
    )


def allocate(shape, dtype, device):
    import numpy as np

    return np.empty(
        shape,
        dtype={"f32": "float32", "f16": "float16", "bf16": "bfloat16", "i32": "int32"}[
            dtype
        ],
    )


def storage_bounds(a):
    import numpy as np

    owner = describe(a)[4]
    raw = (
        np.ndarray((owner.nbytes,), dtype=np.uint8, buffer=owner)
        if isinstance(owner, np.ndarray)
        else np.frombuffer(owner, dtype=np.uint8)
    )
    return raw.ctypes.data, raw.size


def host_shadow(v, backing, start):
    import numpy as np

    return np.ndarray(
        v.shape,
        dtype=v.dtype,
        buffer=backing,
        offset=v.ctypes.data - start,
        strides=v.strides,
    )

def project(value,shape,strides,offset):
    # Slicing once retains the source owner, including for non-contiguous inputs.
    # Construct against its original allocation with the source's byte offset.
    base=value
    while isinstance(getattr(base,"base",None),type(value)):base=base.base
    import numpy as np
    return np.ndarray(shape,dtype=value.dtype,buffer=base,offset=value.ctypes.data-base.ctypes.data+offset*value.itemsize,
                      strides=tuple(x*value.itemsize for x in strides))
