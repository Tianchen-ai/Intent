import ctypes as ct
import os
import subprocess
import threading
from time import perf_counter
from runtime.cpu.binding.frame import HostVariants, NativeBinding
from runtime.cpu.cache.files import build_directory


_WEFT_STATUS_NAMES = {
    1: "invalid-argument",
    2: "io-error",
    3: "abi-error",
    4: "target-error",
    5: "unsupported",
    6: "resource-error",
}


class WeftBinding(NativeBinding):
    def __init__(self, function, last_error, frame, extra, validate, status_type):
        super().__init__(function, frame, extra, validate, status_type)
        self._last_error = last_error

    def check_status(self, status, arguments=None):
        if status:
            code = int(status)
            actual = self.arguments if arguments is None else arguments
            geometry = tuple(x.value for x in actual[self.geometry_offset :])
            detail = ""
            if self._last_error is not None:
                raw = self._last_error()
                if raw:
                    detail = raw.decode("utf-8", errors="replace")
            label = _WEFT_STATUS_NAMES.get(code, "error")
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"{self.function.__name__} failed with native status {code} "
                f"(weft_status={label}), shape, grain, columns and workers={geometry}{suffix}"
            )


class WeftHostVariants(HostVariants):
    def bind_rows(self, fn, bound):
        last_error = self.artifact.last_error_fn()
        return [
            WeftBinding(
                fn,
                last_error,
                bound,
                (*bound.shape, r["grain"], r["columns"], self.artifact.workers),
                lambda: self.artifact.validate(self.scope),
                self.artifact.status_type,
            )
            for r in self.artifact.data["rows"]
        ]


class Artifact:
    status_type = ct.c_int32
    def __init__(self, data, info, workers):
        if data["version"] != 3 or not data["rows"]:
            raise ValueError("invalid CPU row artifact")
        self.data, self.info, self.workers = data, info, workers
        self.winners = {}
        self.lock = threading.RLock()
        self.library = None
        self._last_error = None
        self.compilations = []

    def accepts(self, shape):
        return True

    def bind(self, values, outputs, device, shape, key):
        scope, _ = key
        return WeftHostVariants(self, values, outputs, scope, shape)

    def validate(self, scope):
        if tuple(sorted(os.sched_getaffinity(0))) != scope:
            raise ValueError("CPU affinity changed; prepare a new call")
        if (
            min(len(scope), int(os.environ.get("INTENT_CPU_THREADS", len(scope))))
            != self.workers
        ):
            raise ValueError("CPU budget changed; prepare a new call")

    def last_error_fn(self):
        self.function()
        return self._last_error

    def function(self):
        with self.lock:
            if self.library is None:
                start = perf_counter()
                import triton
                from dataclasses import asdict
                from weft_core.compiler import identity as weft_compiler_identity
                from triton.backends.weft.toolchain import Toolchain

                toolchain = Toolchain.resolve({}, self.info).to_dict()
                identity = [
                    "typed-host-with-runtime-v2",
                    asdict(self.info),
                    triton.__version__,
                    weft_compiler_identity(),
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
                if hasattr(self.library, "intent_last_error"):
                    err_fn = self.library.intent_last_error
                    err_fn.argtypes = []
                    err_fn.restype = ct.c_char_p
                    self._last_error = err_fn
            return self.library.intent_entry

    def build(self, path):
        import triton

        source = path / "task.weftir"
        source.write_text(self.data["task"])
        native = triton.compile(str(source), target=self.info)
        abi = native.metadata.single_instance_abi
        if (
            abi["symbol"] != self.data.get("native_symbol","row_task_instance")
            or abi["shape_parameters"] != self.data["shape_parameters"]
        ):
            raise ValueError("compiled task shape or entry ABI mismatch")
        actual = abi["arguments"]
        task_slots = self.data.get("task_slots", self.data["abi"])
        if len(actual) != len(task_slots):
            raise ValueError("task parameter count differs")
        ctypes = {
            "f16": "_Float16",
            "f32": "float",
            "i8": "int8_t",
            "u8": "uint8_t",
            "i16": "int16_t",
            "f8e4m3fn": "uint8_t",
            "bf16": "uint16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u64": "uint64_t",
        }
        for i, (arg, item) in enumerate(zip(actual, task_slots)):
            rank = self.data["task_ranks"][i] if "task_ranks" in self.data else min(2,len(item["dims"]))
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
                or ("encoding" in item and arg["encoding"] != item["encoding"])
                or arg["c_type"].strip().removesuffix("restrict").rstrip() != wanted
                or len(arg.get("shape", [])) != rank
                or (rank and bool(arg["strided"]) != (i not in self.data.get("contiguous_slots",())))
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
            "#include <stddef.h>\n"
            + abi["declaration"]
            + "\nint weft_kernel_take_status(char *error, size_t error_size);\n"
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
