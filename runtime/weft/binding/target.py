import os
from functools import lru_cache
from pathlib import Path
from runtime.weft.launch.native import Artifact


@lru_cache(maxsize=16)
def discover(scope):
    from triton.cpu import target
    from triton.backends.weft.target import native_capabilities

    cpu = target("native")
    facts = native_capabilities()
    if set(facts["cpus"]) != set(scope):
        raise ValueError("Weft native discovery does not match the active CPU set")
    t = facts["target"]
    core_id = ""
    if cpu.arch == "riscv64":
        ids = set()
        for record in Path("/proc/cpuinfo").read_text().split("\n\n"):
            fields = dict(
                (k.strip(), v.strip())
                for k, v in (
                    line.split(":", 1) for line in record.splitlines() if ":" in line
                )
            )
            if "processor" in fields and int(fields["processor"]) in scope:
                ids.add((fields.get("mvendorid", ""), fields.get("marchid", "")))
        if len(ids) > 1:
            raise ValueError("choose a homogeneous CPU set for native Weft")
        if ids:
            core_id = "/".join(ids.pop())
    return (
        cpu,
        core_id,
        int(t["vector_bits"] if cpu.arch == "x86_64" else t["vlen_bits"]),
    )


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
        self.info, core, bits = discover(scope)
        return dict(
            vector_bits=bits, workers=self.workers, core=core, arch=self.info.arch
        )

    def materialize(self, data):
        return Artifact(data, self.info, self.workers)
