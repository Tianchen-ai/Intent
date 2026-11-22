from runtime.cpu.binding.profile import calibration
import os
from functools import lru_cache
import shutil
import subprocess
from runtime.mojo.launch.native import Artifact


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
    isa = fields["--target-features"].split(",")
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
    return (
        binary,
        fields["--target-cpu"],
        os.stat(binary).st_mtime_ns,
        cc,
        gomp,
        os.stat(cc).st_mtime_ns,
        os.stat(gomp).st_mtime_ns,
    ), bits, "+avx512vnni" in isa and all("avx512_vnni" in cpu_flags[c] for c in scope)


class Target:
    def profile(self, device, buffers):
        if str(device) != "cpu":
            raise ValueError("CPU backend requires CPU views")
        scope = tuple(sorted(os.sched_getaffinity(0)))
        workers = min(len(scope), int(os.environ.get("INTENT_CPU_THREADS", len(scope))))
        if workers < 1:
            raise ValueError("worker budget must be positive")
        return scope, workers

    def options(self, key):
        scope, self.workers = key
        self.info, bits, dot4 = discover(scope)
        return dict(vector_bits=bits, workers=self.workers, grouped_i8=dot4,
                    calibration=calibration(scope,self.workers,bits))

    def materialize(self, data):
        return Artifact(data, self.info, self.workers)
