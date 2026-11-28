"""Host-side CUB DeviceHistogram candidate for a complete additive key reduction."""
import ctypes as C
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


def _nvcc():
    path = shutil.which("nvcc")
    if path:
        return Path(path)
    path = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"
    return path if path.is_file() else None


def cub_toolkit():
    nvcc = _nvcc()
    if nvcc is None:
        return None
    include = nvcc.resolve().parents[1] / "include"
    for root in (include, include / "cccl"):
        if (root / "cub/device/device_histogram.cuh").is_file():
            return nvcc, root
    return None


@lru_cache(maxsize=None)
def _bridge(toolkit, arch):
    nvcc, include = toolkit
    source = Path(__file__).resolve().parents[1] / "kernels/histogram.cu"
    with TemporaryDirectory() as directory:
        binary = Path(directory) / "histogram.so"
        subprocess.run([str(nvcc), "-std=c++17", "-O3", "--shared", "-Xcompiler", "-fPIC",
                        f"-arch=sm_{arch}", "-I", str(include), str(source),
                        "-o", str(binary)], check=True)
        lib = C.CDLL(str(binary))
    invoke = lib.intent_histogram
    invoke.argtypes = [C.c_void_p, C.POINTER(C.c_size_t), C.c_void_p, C.c_void_p,
                       C.c_int, C.c_int64, C.c_void_p]
    invoke.restype = C.c_int
    return invoke


class Histogram:
    def __init__(self, src, dst, toolkit, traffic):
        self.src, self.dst, self.toolkit = src, dst, toolkit
        self.traffic = traffic
        self.scratch = None

    def _invoke(self, pointer, size):
        status = self.invoke(pointer, C.byref(size), self.src.data_ptr(), self.dst.data_ptr(),
                             self.dst.numel(), self.src.numel(),
                             torch.cuda.current_stream(self.src.device).cuda_stream)
        if status:
            raise RuntimeError(f"CUB DeviceHistogram::HistogramEven returned CUDA error {status}")

    def compile(self):
        if self.scratch is None:
            with torch.cuda.device(self.src.device):
                major, minor = torch.cuda.get_device_capability(self.src.device)
                self.invoke = _bridge(self.toolkit, f"{major}{minor}")
                size = C.c_size_t()
                self._invoke(None, size)
                self.scratch = torch.empty(max(1, size.value), dtype=torch.uint8,
                                           device=self.src.device)

    def enqueue(self):
        self.compile()
        with torch.cuda.device(self.src.device):
            self._invoke(self.scratch.data_ptr(), C.c_size_t(self.scratch.numel()))

    def compilation_report(self):
        return {"library": "cub::DeviceHistogram::HistogramEven",
                "source": "NVIDIA CCCL cub/device/device_histogram.cuh",
                "selection": "partition count workspace versus resident SM counters",
                "workspace_bytes": self.scratch.numel() if self.scratch is not None else None,
                "estimated_counter_bytes": self.traffic}

    def tuning_report(self):
        return self.compilation_report()


def compose_histogram(calls, labels, sites, private):
    first, last = calls
    a, b = (step.artifact.data.get("key_summary", {}) for step in calls)
    if a.get("operation") != "count_keys" or b.get("operation") != "column_sum":
        return calls, labels
    name = sites[0][2][a["output"]]
    if name not in private or sites[1][2][b["input"]] != name:
        return calls, labels
    src, parts = first.values[a["input"]], first.values[a["output"]]
    dst = last.values[b["output"]]
    if (parts is not last.values[b["input"]] or src.dtype != torch.float32 or
            parts.dtype != torch.int32 or dst.dtype != torch.int32 or
            src.ndim != 1 or dst.ndim != 1 or parts.ndim != 2 or
            not src.is_contiguous() or not dst.is_contiguous() or
            src.untyped_storage().data_ptr() == dst.untyped_storage().data_ptr()):
        return calls, labels
    rows, bins = parts.shape
    if not (0 < bins < 2**31-1 and dst.numel() == bins and 0 < src.numel() < 2**31 and
            a["span"] > 0 and rows == (src.numel()+a["span"]-1)//a["span"]):
        return calls, labels
    # Partitioned global atomics and reduction write then reread one row per
    # partition. DeviceHistogram bounds concurrently resident partials by SMs.
    sms = torch.cuda.get_device_properties(src.device).multi_processor_count
    traffic = {"partitioned": 8*rows*bins, "device": 8*sms*bins}
    toolkit = cub_toolkit()
    if traffic["device"] >= traffic["partitioned"] or toolkit is None:
        return calls, labels
    return [Histogram(src, dst, toolkit, traffic)], ["cub::DeviceHistogram::HistogramEven"]
