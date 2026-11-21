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
                import triton
                from dataclasses import asdict
                from triton.backends.weft.toolchain import Toolchain

                toolchain = Toolchain.resolve({}, self.info).to_dict()
                identity = [
                    "typed-host-with-runtime",
                    asdict(self.info),
                    triton.__version__,
                    toolchain,
                    self.data,
                ]
                path, hit = build_directory("weft", identity, self.build)
                self.compilations = [
                    dict(
                        id="host",
                        milliseconds=(perf_counter() - start) * 1000,
                        cache_hit=hit,
                        rows=self.data["rows"],
                    )
                ]
                self.library = ct.CDLL(str(path / "host.so"))
            return self.library.intent_entry

    def build(self, path):
        import triton

        source = path / "task.weftir"
        source.write_text(self.data["task"])
        native = triton.compile(str(source), target=self.info)
        abi = native.metadata.single_instance_abi
        if abi["symbol"] != "row_task_instance" or abi["shape_parameters"] != [
            "T",
            "N",
        ]:
            raise ValueError("compiled task shape or entry ABI mismatch")
        actual = abi["arguments"]
        if len(actual) != len(self.data["abi"]):
            raise ValueError("task parameter count differs")
        ctypes = {"f16": "_Float16", "f32": "float"}
        for i, (arg, item) in enumerate(zip(actual, self.data["abi"])):
            rank = len(item["dims"])
            wanted = (
                (
                    ("const " if item["direction"] == "In" else "")
                    + ctypes[item["dtype"]]
                    + " *"
                )
                if rank
                else ctypes[item["dtype"]]
            )
            if (
                arg["name"] != "a" + str(i)
                or arg["c_type"].strip() != wanted
                or len(arg.get("shape", [])) != rank
                or (rank and not arg["strided"])
            ):
                raise ValueError(
                    f"task slot {i} disagrees with compiled declaration: {arg}"
                )
        if abi.get("used_extensions"):
            raise ValueError("row task requires an unhandled worker permission")
        native.export(path / "task")
        name = "task_" + native.hash + ".so"
        (path / "task/kernel.so").rename(path / "task" / name)
        (path / "task_bridge.h").write_text(
            "#include <stddef.h>\n" + abi["declaration"] + "\n"
        )
        (path / "host.c").write_text(self.data["host"])
        # Use the local C toolchain to type-check the bridge against the actual exported prototype.
        from triton.backends.weft.toolchain import Toolchain

        tool = Toolchain.resolve({}, self.info)
        cc = tool.cc
        subprocess.run(
            [
                cc,
                *tool.cflags,
                *tool.ldflags,
                "-std=gnu11",
                "-O3",
                "-fopenmp",
                "-fPIC",
                "-shared",
                "-Werror=incompatible-pointer-types",
                "host.c",
                "-Ltask",
                "-l:" + name,
                "-Wl,-rpath,$ORIGIN/task:$ORIGIN/task/lib",
                "-o",
                "host.so",
            ],
            cwd=path,
            check=True,
        )
