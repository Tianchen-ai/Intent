from runtime.cpu.launch.completion import submit
from runtime.cpu.binding.profile import calibration, private_cache_bytes
import os
from functools import lru_cache
from pathlib import Path
from runtime.weft.launch.native import Artifact


@lru_cache(maxsize=16)
def discover(scope):
    import weft_core
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
    weft_ver = str(facts.get("weft_version") or getattr(weft_core, "__version__", ""))
    isa = frozenset().union(
        *(
            facts.get(key) or ()
            for key in (
                "available_isa_extensions",
                "execution_isa_extensions",
                "isa_extensions",
            )
        )
    )
    brg = dict(facts.get("onednn_brgemm") or {})
    caps = dict(facts.get("capabilities") or {})
    bits = int(t["vector_bits"] if cpu.arch == "x86_64" else t["vlen_bits"])
    vec_regs = int(
        caps.get("vector_registers")
        or (32 if cpu.arch == "riscv64" else (32 if bits >= 512 else 16))
    )
    lmul_list = list(caps.get("legal_lmul_eighths") or ())
    max_lmul_eighths = int(max(lmul_list)) if lmul_list else (64 if cpu.arch == "riscv64" else 8)
    private_stack_bytes = int(caps.get("private_stack_bytes") or 65536)
    return (
        cpu,
        core_id,
        bits,
        vec_regs,
        max_lmul_eighths,
        private_stack_bytes,
        weft_ver,
        isa,
        brg,
    )



class Target:
    submit = staticmethod(submit)

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
        self.info, core, bits, vec_regs, max_lmul_eighths, private_stack_bytes, weft_ver, isa, brg = discover(scope)
        ver_nums = brg.get("version") or ()
        return dict(
            vector_bits=bits,
            vector_registers=vec_regs,
            max_lmul_eighths=max_lmul_eighths,
            private_stack_bytes=private_stack_bytes,
            workers=self.workers,
            core=core,
            arch=self.info.arch,
            zvfbfmin="zvfbfmin" in isa,
            grouped_i8=self.info.arch == "x86_64" and "+avx512vnni" in self.info.march,
            avx512_vnni="avx512vnni" in isa,
            avx512_fp16="avx512fp16" in isa,
            amx_bf16="amx-bf16" in isa,
            amx_int8="amx-int8" in isa,
            weft_version=weft_ver,
            weft_brgemm_available=bool(brg.get("available")),
            weft_brgemm_runtime=str(brg.get("runtime", "")),
            weft_brgemm_version=".".join(str(x) for x in ver_nums),
            weft_brgemm_f32=bool(brg.get("f32")),
            weft_brgemm_f16=bool(brg.get("f16")),
            weft_brgemm_bf16=bool(brg.get("bf16")),
            weft_brgemm_s8=bool(brg.get("s8")),
            weft_brgemm_u8s8=bool(brg.get("u8s8")),
            calibration=calibration(scope, self.workers, bits),
            l1_cache_bytes=private_cache_bytes(scope, 1),
            l2_cache_bytes=private_cache_bytes(scope),
        )

    def materialize(self, data):
        return Artifact(data, self.info, self.workers)
