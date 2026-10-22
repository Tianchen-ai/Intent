"""Capture complete program results without changing their logical dtypes."""

import json
from pathlib import Path

import numpy as np

from .inputs import Array, numpy_dtype


def _storage_span(shape, strides):
    if any(size == 0 for size in shape):
        return 0, 0
    lower = sum(min(0, (size - 1) * stride) for size, stride in zip(shape, strides, strict=True))
    upper = sum(max(0, (size - 1) * stride) for size, stride in zip(shape, strides, strict=True))
    return upper - lower + 1, -lower


def capture_outputs(value, directory):
    """Copy a returned result tree now and write its typed storage to directory.

    The returned path names the manifest. Array snapshots own their bytes and
    remain valid after native buffers and their example context are closed.
    """
    arrays = []

    def capture(item):
        if isinstance(item, Array):
            if item.data is None:
                raise ValueError("cannot capture metadata-only output; execute the program first")
            dtype = numpy_dtype(item.dtype)
            shape = tuple(int(size) for size in item.shape)
            strides = tuple(int(stride) for stride in item.strides)
            count, offset = _storage_span(shape, strides)
            storage = np.zeros(count * dtype.itemsize, dtype=np.uint8)
            snapshot = np.ndarray(shape, dtype=dtype, buffer=storage,
                                  offset=offset * dtype.itemsize,
                                  strides=tuple(stride * dtype.itemsize for stride in strides))
            np.copyto(snapshot, item.data, casting="no")
            filename = f"array_{len(arrays):04d}.npy"
            arrays.append((filename, storage))
            return {"kind": "array", "file": filename, "dtype": item.dtype,
                    "shape": list(shape), "strides": list(strides), "offset": offset}
        if isinstance(item, tuple):
            return {"kind": "tuple", "items": [capture(element) for element in item]}
        if isinstance(item, list):
            return {"kind": "list", "items": [capture(element) for element in item]}
        if isinstance(item, dict):
            return {"kind": "dict", "items": [[capture(key), capture(element)]
                                               for key, element in item.items()]}
        if item is None:
            return {"kind": "none"}
        if isinstance(item, bool):
            return {"kind": "bool", "value": item}
        if isinstance(item, int):
            return {"kind": "int", "value": item}
        if isinstance(item, float):
            return {"kind": "float", "value": item.hex()}
        if isinstance(item, str):
            return {"kind": "str", "value": item}
        raise TypeError(f"unsupported program result type: {type(item).__name__}")

    manifest = capture(value)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for filename, storage in arrays:
        np.save(directory / filename, storage, allow_pickle=False)
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def read_outputs(directory):
    """Read a captured result tree into independently owned host storage."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    def restore(item):
        kind = item["kind"]
        if kind == "array":
            filename = item["file"]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("an output array file must be a local filename")
            storage = np.load(directory / filename, allow_pickle=False)
            if storage.dtype != np.dtype(np.uint8) or storage.ndim != 1:
                raise ValueError("output array storage must contain one-dimensional raw bytes")
            shape, strides = tuple(item["shape"]), tuple(item["strides"])
            dtype = numpy_dtype(item["dtype"])
            count, offset = _storage_span(shape, strides)
            if item["offset"] != offset or storage.nbytes != count * dtype.itemsize:
                raise ValueError("output array storage does not match its shape and strides")
            data = np.ndarray(shape, dtype=dtype, buffer=storage,
                              offset=offset * dtype.itemsize,
                              strides=tuple(stride * dtype.itemsize for stride in strides))
            return Array(data, item["dtype"], shape)
        if kind == "tuple":
            return tuple(restore(element) for element in item["items"])
        if kind == "list":
            return [restore(element) for element in item["items"]]
        if kind == "dict":
            return {restore(key): restore(element) for key, element in item["items"]}
        if kind == "none":
            return None
        if kind == "bool":
            return bool(item["value"])
        if kind == "int":
            return int(item["value"])
        if kind == "float":
            return float.fromhex(item["value"])
        if kind == "str":
            return item["value"]
        raise ValueError(f"unknown program result kind: {kind}")

    return restore(manifest)
