import os
from functools import lru_cache
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
    return cpu, int(t["vector_bits"] if cpu.arch == "x86_64" else t["vlen_bits"])


class Target:
    def specialize(self, device, values):
        if str(device) != "cpu":
            raise ValueError("CPU backend requires CPU views")
        scope = tuple(sorted(os.sched_getaffinity(0)))
        workers = min(len(scope), int(os.environ.get("INTENT_CPU_THREADS", len(scope))))
        if workers < 1:
            raise ValueError("worker budget must be positive")
        return (scope, workers), (scope,)

    def options(self, key, shape):
        scope, self.workers = key
        self.info, bits = discover(scope)
        return dict(vector_bits=bits, workers=self.workers)

    def materialize(self, data):
        return Artifact(data, self.info, self.workers)
