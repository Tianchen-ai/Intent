"""Local CNRT device facts for the BANG C exit."""

from runtime.kir.binding.completion import submit_device as submit
from math import prod
import ctypes as C
from functools import lru_cache
import torch
from runtime.bangc.launch.call import Artifact


@lru_cache(maxsize=None)
def facts(index):
    lib = C.CDLL("libcnrt.so")
    get = lib.cnrtDeviceGetAttribute
    get.argtypes = [C.POINTER(C.c_int), C.c_int, C.c_int]
    values = {}
    # Stable cnrtDeviceAttr_t identifiers; sizes are bytes per physical owner.
    for name, field in (
        ("isa", 0x20A),
        ("clusters", 0x205),
        ("cores", 0x206),
        ("nram_bytes", 0x302),
        ("wram_bytes", 0x303),
        ("sram_bytes", 0x306),
    ):
        value = C.c_int()
        status = get(C.byref(value), field, index)
        if status:
            raise RuntimeError(f"CNRT query {name} on device {index} returned {status}")
        values[name] = value.value
    return values



def external_dependencies():
    """Probe external Cambricon Neuware toolchain, runtime, reference library, and driver."""
    import glob
    import shutil
    import subprocess

    report = {}
    cncc = shutil.which("cncc")
    if cncc:
        try:
            ver = subprocess.check_output([cncc, "--version"], text=True, stderr=subprocess.STDOUT).splitlines()[0].strip()
        except Exception as exc:
            ver = f"error: {exc}"
        report["cncc"] = {"available": True, "path": cncc, "version": ver, "external": True}
    else:
        report["cncc"] = {"available": False, "reason": "cncc not found on PATH", "external": True}

    try:
        cnrt = C.CDLL("libcnrt.so")
        count = C.c_uint(0)
        rc = cnrt.cnrtGetDeviceCount(C.byref(count))
        devs = [facts(i) for i in range(count.value)] if rc == 0 and count.value > 0 else []
        report["cnrt"] = {
            "available": rc == 0,
            "device_count": int(count.value) if rc == 0 else 0,
            "devices": devs,
            "external": True,
        }
    except OSError as exc:
        report["cnrt"] = {"available": False, "reason": str(exc), "external": True}

    try:
        cnnl = C.CDLL("libcnnl.so")
        maj, min_, pat = C.c_int(0), C.c_int(0), C.c_int(0)
        if hasattr(cnnl, "cnnlGetLibVersion"):
            cnnl.cnnlGetLibVersion(C.byref(maj), C.byref(min_), C.byref(pat))
            ver = f"{maj.value}.{min_.value}.{pat.value}"
        else:
            ver = "loaded"
        report["cnnl"] = {"available": True, "version": ver, "role": "reference_library", "external": True}
    except OSError as exc:
        report["cnnl"] = {"available": False, "reason": str(exc), "role": "reference_library", "external": True}

    nodes = sorted(glob.glob("/dev/cambricon_dev*"))
    report["driver"] = {
        "available": bool(nodes) or bool(shutil.which("cnmon")),
        "devices": nodes,
        "cnmon": shutil.which("cnmon") or "",
        "external": True,
    }
    return report


class Target:
    submit = staticmethod(submit)

    def profile(self, device, buffers):
        if device.type != "mlu":
            raise ValueError("BANG C requires inputs on a local MLU")
        return device

    def options(self, device):
        return dict(facts(device.index),calibration="mlu370")

    def materialize(self, data):
        return Artifact(data)
