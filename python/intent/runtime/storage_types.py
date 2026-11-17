"""Host representations of public storage dtypes; no computation policy."""

_NUMPY_TYPES = {
    "bool": "bool", "u8": "uint8", "i8": "int8", "u16": "uint16", "i16": "int16",
    "u32": "uint32", "i32": "int32", "u64": "uint64", "i64": "int64",
    "f16": "float16", "f32": "float32", "f64": "float64",
}
_LOW_PRECISION_TYPES = {
    "bf16": "bfloat16", "f8e4m3fn": "float8_e4m3fn", "f8e5m2": "float8_e5m2",
}


def numpy_dtype(element):
    import numpy as np
    if element in _NUMPY_TYPES:
        return np.dtype(_NUMPY_TYPES[element])
    if element in _LOW_PRECISION_TYPES:
        import ml_dtypes
        return np.dtype(getattr(ml_dtypes, _LOW_PRECISION_TYPES[element]))
    raise NotImplementedError(f"host storage does not support dtype {element}")
