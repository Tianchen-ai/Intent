def accepts(value):
    return any(
        cls.__module__.split(".")[0] == "numpy" and cls.__name__ == "ndarray"
        for cls in type(value).__mro__
    )


def describe(a):
    import numpy as np

    dtype = {np.dtype("float16"): "f16", np.dtype("float32"): "f32"}.get(a.dtype)
    if any(s % a.itemsize for s in a.strides):
        raise ValueError("nonintegral element stride")
    owner = a
    while getattr(owner, "base", None) is not None:
        owner = owner.base
    return (
        dtype,
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

    return np.empty(shape, dtype={"f32": "float32", "f16": "float16"}[dtype])


def trial(a):
    import numpy as np

    size = (
        a.itemsize + sum((n - 1) * s for n, s in zip(a.shape, a.strides))
        if a.size
        else 0
    )
    return np.ndarray(a.shape, a.dtype, buffer=bytearray(size), strides=a.strides)
