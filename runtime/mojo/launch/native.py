import ctypes as ct
import os
import subprocess
import threading
from time import perf_counter
from runtime.kir.binding.frame import HostVariants
from runtime.kir.cache.files import build_directory


class Artifact:
    def __init__(self, data, info, workers):
        if data["version"] != 1 or not data["rows"]:
            raise ValueError("invalid CPU row artifact")
        self.data, self.info, self.workers = data, info, workers
        self.winners = {}
        self.lock = threading.RLock()
        self.library = None
        self.compilations = []

    def accepts(self, shape):
        return True

    def bind(self, values, outputs, device, scope):
        return HostVariants(self, values, outputs, scope)

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
                library = ct.CDLL(str(path / "host.so"))
                library.intent_initialize.argtypes = []
                library.intent_initialize.restype = None
                library.intent_initialize()
                self.library = library
            return self.library.intent_entry

    def build(self, path):
        source = path / "row.mojo"
        source.write_text(self.data["source"])
        subprocess.run(
            [
                self.info[0],
                "build",
                "--emit",
                "shared-lib",
                str(source),
                "-o",
                str(path / "host.so"),
            ],
            check=True,
        )
