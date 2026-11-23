from runtime.cpu.launch.completion import submit
from runtime.cpu.binding.profile import calibration
import os
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
from runtime.mojo.launch.native import Artifact


def _discover_onednn():
    override = os.environ.get("INTENT_ONEDNN_LIB", "").strip()
    candidates = [override] if override else [
        "/opt/rc-dependencies/onednn-3.13.4/lib/libdnnl.so",
        "/opt/rc-dependencies/onednn-3.13.4/lib/libdnnl.so.3",
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            real = os.path.realpath(cand)
            prefix = Path(real).parents[1]
            inc = prefix / "include"
            if (inc / "oneapi" / "dnnl" / "dnnl_ukernel.h").is_file():
                return real, str(inc), os.stat(real).st_mtime_ns
    return "", "", 0


@lru_cache(maxsize=16)
def discover(scope):
    binary = shutil.which(os.environ.get("INTENT_MOJO", "mojo"))
    if binary is None:
        raise FileNotFoundError("set INTENT_MOJO to the local Mojo compiler")
    report = subprocess.check_output(
        [binary, "build", "--print-effective-target"], text=True
    )
    fields = dict(
        line.strip().split(None, 1)
        for line in report.splitlines()
        if line.strip().startswith("--target-")
    )
    isa = set(fields["--target-features"].split(","))
    bits = 512 if "+avx512f" in isa else 256 if "+avx2" in isa else 0
    if not bits or not fields["--target-triple"].startswith("x86_64"):
        raise ValueError("unsupported Mojo CPU target")
    cpu_flags = {}
    for record in open("/proc/cpuinfo").read().split("\n\n"):
        info = dict(
            (k.strip(), v.strip())
            for k, v in (
                line.split(":", 1) for line in record.splitlines() if ":" in line
            )
        )
        if "processor" in info:
            cpu_flags[int(info["processor"])] = set(info.get("flags", "").split())
    if not all(("avx512f" if bits == 512 else "avx2") in cpu_flags[c] for c in scope):
        raise ValueError("effective Mojo ISA exceeds CPU set")
    cc = shutil.which("gcc")
    if not cc:
        raise FileNotFoundError("Mojo row host requires a C compiler with GNU OpenMP")
    gomp = subprocess.check_output(
        [cc, "-print-file-name=libgomp.so"], text=True
    ).strip()
    if not os.path.isabs(gomp) or not os.path.isfile(gomp):
        raise FileNotFoundError("GNU OpenMP runtime for Mojo host")

    def has_all(*flags):
        return all(all(f in cpu_flags[c] for f in flags) for c in scope)

    feats = dict(
        grouped_i8=("+avx512vnni" in isa and has_all("avx512_vnni")),
        avx512_bf16=has_all("avx512_bf16"),
        avx512_fp16=has_all("avx512_fp16"),
        amx_int8=has_all("amx_tile", "amx_int8"),
        amx_bf16=has_all("amx_tile", "amx_bf16"),
    )
    onednn_lib, onednn_inc, onednn_mtime = _discover_onednn()
    return (
        binary,
        fields["--target-cpu"],
        os.stat(binary).st_mtime_ns,
        cc,
        gomp,
        os.stat(cc).st_mtime_ns,
        os.stat(gomp).st_mtime_ns,
        bits,
        onednn_lib,
        onednn_inc,
        onednn_mtime,
    ), bits, feats


class Target:
    submit = staticmethod(submit)

    def profile(self, device, buffers):
        if str(device) != "cpu":
            raise ValueError("CPU backend requires CPU views")
        scope = tuple(sorted(os.sched_getaffinity(0)))
        workers = min(
            len(scope), int(os.environ.get("INTENT_CPU_THREADS", len(scope)))
        )
        if workers < 1:
            raise ValueError("worker budget must be positive")
        return scope, workers

    def options(self, key):
        scope, self.workers = key
        self.info, bits, feats = discover(scope)
        return dict(
            vector_bits=bits,
            workers=self.workers,
            calibration=calibration(scope, self.workers, bits),
            **feats,
        )

    def materialize(self, data):
        return Artifact(data, self.info, self.workers)
