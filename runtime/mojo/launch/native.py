import ctypes as ct
import os
from pathlib import Path
import subprocess
import threading
from time import perf_counter
from runtime.cpu.binding.frame import HostVariants
from runtime.cpu.cache.files import build_directory


def _bridge_source():
    here = Path(__file__).resolve()
    local = here.parents[1] / "kernels" / "brgemm_bridge.c"
    if local.is_file():
        return local
    raise FileNotFoundError(f"missing runtime kernel source {local}")


class Artifact:
    status_type = None

    def __init__(self, data, info, workers):
        if data["version"] != 2 or not data["rows"]:
            raise ValueError("invalid CPU row artifact")
        self.data, self.info, self.workers = data, info, workers
        if data.get("returns_status"):
            self.status_type = ct.c_int32
        self.winners = {}
        self.lock = threading.RLock()
        self.library = None
        self.compilations = []

    def accepts(self, shape):
        return True

    def bind(self, values, outputs, device, shape, key):
        scope, _ = key
        return HostVariants(self, values, outputs, scope, shape)

    def validate(self, scope):
        if tuple(sorted(os.sched_getaffinity(0))) != scope:
            raise ValueError("CPU affinity changed; prepare a new call")
        if (
            min(len(scope), int(os.environ.get("INTENT_CPU_THREADS", len(scope))))
            != self.workers
        ):
            raise ValueError("CPU budget changed; prepare a new call")

    def function(self):
        with self.lock:
            if self.library is None:
                start = perf_counter()
                identity = [self.info, self.data]
                path, hit = build_directory("mojo", identity, self.build)
                self.compilations = [
                    dict(
                        id="host",
                        milliseconds=(perf_counter() - start) * 1000,
                        cache_hit=hit,
                        rows=self.data["rows"],
                    )
                ]
                if self.data.get("onednn") and self.info[8]:
                    ct.CDLL(self.info[8], mode=ct.RTLD_GLOBAL)
                library = ct.CDLL(str(path / "host.so"))
                self.library = library
            return self.library.intent_entry

    def build(self, path):
        source = path / "row.mojo"
        source.write_text(self.data["source"])
        host = path / "entry.c"
        host.write_text(self.data["host"])
        obj = path / "entry.o"
        bits = self.info[7]
        extra_isa = self.info[11] if len(self.info) > 11 else ()
        isa_flags = (
            ["-mavx512f", "-mf16c", *extra_isa]
            if bits == 512
            else ["-mavx2", "-mf16c"]
        )
        subprocess.run(
            [
                self.info[3],
                "-O3",
                "-fopenmp",
                "-fPIC",
                *isa_flags,
                "-c",
                str(host),
                "-o",
                str(obj),
            ],
            check=True,
        )
        link_args = ["-Xlinker", str(obj), "-Xlinker", self.info[4]]
        if self.data.get("onednn"):
            onednn_lib, onednn_inc = self.info[8], self.info[9]
            if not onednn_lib or not onednn_inc:
                raise RuntimeError("oneDNN runtime library required by compiled artifact")
            bridge_obj = path / "brgemm_bridge.o"
            subprocess.run(
                [
                    self.info[3],
                    "-O3",
                    "-fPIC",
                    *isa_flags,
                    f"-I{onednn_inc}",
                    "-c",
                    str(_bridge_source()),
                    "-o",
                    str(bridge_obj),
                ],
                check=True,
            )
            onednn_dir = str(Path(onednn_lib).parent)
            link_args.extend(
                [
                    "-Xlinker",
                    str(bridge_obj),
                    "-Xlinker",
                    onednn_lib,
                    "-Xlinker",
                    "--disable-new-dtags",
                    "-Xlinker",
                    f"-rpath={onednn_dir}",
                    "-Xlinker",
                    "-lpthread",
                ]
            )
        subprocess.run(
            [
                self.info[0],
                "build",
                "--emit",
                "shared-lib",
                "--fp-mode", "contract=" + self.data.get("fp_contract", "fast"),
                str(source),
                *link_args,
                "-o",
                str(path / "host.so"),
            ],
            check=True,
        )
