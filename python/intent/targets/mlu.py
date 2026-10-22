"""CNRT device facts. Querying a target does not allocate or execute a kernel."""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
import re

from .specification import DSACapabilities


@lru_cache(maxsize=None)
def _attribute_ids(neuware):
    # Read the installed SDK's enum rather than copying version-specific numbers.
    header = Path(neuware) / "include/cnrt.h"
    text = header.read_text(encoding="utf-8")
    return {name: int(value, 0) for name, value in re.findall(
        r"(cnrtAttr\w+)\s*=\s*(0x[0-9a-fA-F]+|[0-9]+)", text)}


def resolve_mlu_device(device, neuware):
    from intent.runtime.bangc.buffer import runtime
    driver = runtime(neuware)
    count = ctypes.c_uint()
    driver.invoke("cnrtGetDeviceCount", ctypes.byref(count))
    if not 0 <= device < count.value:
        raise ValueError(f"MLU device {device} is unavailable; CNRT reports {count.value} devices")
    query = driver.library.cnrtDeviceGetAttribute
    query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    query.restype = ctypes.c_int
    ids = _attribute_ids(str(Path(neuware).resolve()))

    def attribute(name):
        value = ctypes.c_int()
        if name not in ids:
            raise NotImplementedError(f"installed CNRT SDK does not declare {name}")
        status = query(ctypes.byref(value), ids[name], device)
        if status:
            raise RuntimeError(f"CNRT query {name} failed with status {status}")
        return value.value

    major = attribute("cnrtAttrComputeCapabilityMajor")
    minor = attribute("cnrtAttrComputeCapabilityMinor")
    if (major, minor) != (3, 0):
        raise NotImplementedError(f"BANG C lowering supports MLU370 (3.0), not detected capability {major}.{minor}")
    return DSACapabilities(
        attribute("cnrtAttrClusterCount"), attribute("cnrtAttrMcorePerCluster"),
        attribute("cnrtAttrNramSizePerMcore"), attribute("cnrtAttrWramSizePerMcore"),
        attribute("cnrtAttrSramSizePerMcore"))
