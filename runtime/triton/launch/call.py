from threading import RLock
from runtime.kir.binding.completion import PreparedLaunch
from time import perf_counter
import torch
import triton
from runtime.block.binding.grid import accepts, grid
from runtime.triton.cache.source import load

triton.set_allocator(
    lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda")
)


class Artifact:
    def __init__(self, data):
        if data["version"] != 2:
            raise ValueError("unsupported Intent Triton ABI version")
        self.data = data
        self.kernel = load(data["source"], data["entry"])
        pre = data.get("pre_stage")
        self.pre_kernel = load(data["source"], pre["entry"]) if pre else None
        self.compiled = {}
        self.winners = {}
        self.lock = RLock()

    def accepts(self, shape):
        return accepts(self.data["guard"], shape)

    def bind(self, values, outputs, device, shape, key):
        m, n = shape
        return Prepared(self, values, outputs, device, m, n)


_WS_DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}


class Prepared(PreparedLaunch):
    def __init__(self, artifact, values, outputs, device, m, n):
        self.artifact, self.values, self.outputs = artifact, values, outputs
        self.device, self.m, self.n = device, m, n
        self.pre_stage = artifact.data.get("pre_stage")
        self.library_gemm = bool(artifact.data.get("library_gemm"))
        self.workspace = None
        self.pre_flat = None
        if self.pre_stage:
            ws = self.pre_stage["workspace"]
            self.workspace = torch.empty(
                tuple(ws["shape"]), dtype=_WS_DTYPES[ws["dtype"]], device=device
            )
            self.pre_flat = (
                *(values[i] for i in self.pre_stage["arg_slots"]),
                self.workspace,
            )
        self.flat = self.flatten(values)
        if not artifact.accepts((m, n)):
            raise ValueError("width lies outside the compiled shape class")
        self.rows = artifact.data["rows"]
        if not self.rows:
            raise ValueError("no row satisfies the bound width")
        self.launchers = {}
        self.pre_launcher = None
        self.builds = []
        self.timings = {r["id"]: [] for r in self.rows}
        self.ready = False
        self.row = self.rows[0] if len(self.rows) == 1 else None

    def flatten(self, values):
        if self.artifact.data.get("matrix_views") or self.artifact.data.get("direct_views"):
            if self.workspace is not None:
                return (*values, self.workspace)
            return tuple(values)
        args = []
        for value in values:
            args.append(value)
            if isinstance(value, torch.Tensor):
                if isinstance(self.artifact.data["domain"]["inner"],list):
                    if value.ndim>1: args.extend(value.shape)
                elif value.ndim>2: args.extend(value.shape[:-1])
                args.extend(value.stride())
        return (*args, self.m, self.n)

    def _compile_key(self):
        return (
            self.device,
            self.m,
            self.n,
            tuple(
                (a.dtype, tuple(a.stride()), a.data_ptr() % 16 == 0)
                if isinstance(a, torch.Tensor)
                else (type(a), a if isinstance(a, int) else None)
                for a in self.flat
            ),
        )

    def compile(self):
        if self.ready:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("compile before graph capture")
        ckey = self._compile_key()
        with self.artifact.lock:
            cached = self.artifact.compiled.get(ckey)
        if cached is not None:
            self.rows, self.launchers, self.builds, self.pre_launcher = cached
            if len(self.rows) == 1:
                self.row = self.rows[0]
            self.ready = True
            return
        with torch.cuda.device(self.device), self.artifact.lock:
            cached = self.artifact.compiled.get(ckey)
            if cached is not None:
                self.rows, self.launchers, self.builds, self.pre_launcher = cached
                if len(self.rows) == 1:
                    self.row = self.rows[0]
                self.ready = True
                return
            if self.pre_stage is not None:
                pg = tuple(self.pre_stage["grid"])
                pre_start = perf_counter()
                pre_native = self.artifact.pre_kernel.warmup(
                    *self.pre_flat,
                    ROWS=self.pre_stage["rows"],
                    WIDTH=self.pre_stage["width"],
                    grid=pg,
                    enable_fp_fusion=self.pre_stage.get("enable_fp_fusion", False),
                    **self.pre_stage["native"],
                )
                self.pre_launcher = pre_native[pg]
                self.builds.append(
                    {
                        "id": "pre_stage",
                        "milliseconds": (perf_counter() - pre_start) * 1000,
                        "native_hash": pre_native.hash,
                    }
                )
            if self.library_gemm:
                self.artifact.compiled[ckey] = (
                    list(self.rows),
                    dict(self.launchers),
                    list(self.builds),
                    self.pre_launcher,
                )
                if len(self.rows) == 1:
                    self.row = self.rows[0]
                self.ready = True
                return
            failures = {}
            valid_rows = []
            for r in self.rows:
                start = perf_counter()
                try:
                    g = grid(self.artifact.data["grid"], (self.m, self.n), r)
                    native = self.artifact.kernel.warmup(
                        *self.flat,
                        ROWS=r["rows"],
                        WIDTH=r["width"],
                        grid=g,
                        enable_fp_fusion=self.artifact.data.get("enable_fp_fusion", False),
                        **r["native"],
                    )
                    self.launchers[r["id"]] = native[g]
                except triton.runtime.errors.OutOfResources as exc:
                    failures[r["id"]] = str(exc)
                    continue
                self.builds.append({"id": r["id"], "milliseconds": (perf_counter() - start) * 1000,
                                    "native_hash": native.hash})
                valid_rows.append(r)
            if not valid_rows:
                raise RuntimeError(f"Triton native compilation failed: {failures}")
            self.rows = valid_rows
            self.artifact.compiled[ckey] = (
                list(self.rows),
                dict(self.launchers),
                list(self.builds),
                self.pre_launcher,
            )
            if len(self.rows) == 1:
                self.row = self.rows[0]
        self.ready = True

    def compilation_report(self):
        self.compile()
        return {"candidate_count": len(self.rows), "rows": list(self.builds)}

    def _invoke(self, row, args):
        if self.pre_stage is not None:
            pre_args = (
                *(args[i] for i in self.pre_stage["arg_slots"]),
                self.workspace,
            )
            self.pre_launcher(
                *pre_args, self.pre_stage["rows"], self.pre_stage["width"]
            )
        if self.library_gemm:
            torch.matmul(
                args[self.pre_stage["lhs_slot"]],
                self.workspace,
                out=args[self.pre_stage["out_slot"]],
            )
            return
        self.launchers[row["id"]](*args, row["rows"], row["width"])

    def enqueue(self):
        with torch.cuda.device(self.device):
            self.compile()
            if not all(grid(self.artifact.data["grid"], (self.m, self.n), self.rows[0])):
                return
            if self.row is None:
                key = (
                    self.device,
                    self.m,
                    self.n,
                    tuple(
                        tuple(v.stride())
                        for v in self.values
                        if isinstance(v, torch.Tensor)
                    ),
                )
                with self.artifact.lock:
                    if key not in self.artifact.winners:
                        if torch.cuda.is_current_stream_capturing():
                            raise RuntimeError("select a row before graph capture")
                        from runtime.kir.binding.trials import (
                            TrialBindings,
                            cuda_event_trials,
                            cuda_trials,
                        )
                        from statistics import median

                        shadow = TrialBindings(self.values, self.artifact.data["abi"])
                        trial = self.flatten(shadow.values)
                        stream = torch.cuda.current_stream(self.device)
                        times = {}
                        for r in self.rows:

                            def invoke():
                                self._invoke(r, trial)

                            if shadow.restores:
                                times[r["id"]] = cuda_trials(
                                    shadow,
                                    stream,
                                    invoke,
                                )
                            else:
                                times[r["id"]] = cuda_event_trials(
                                    stream,
                                    invoke,
                                )
                        best = min(self.rows, key=lambda r: median(times[r["id"]]))
                        self.artifact.winners[key] = best, times
                    self.row, self.timings = self.artifact.winners[key]
            self._invoke(self.row, self.flat)
            stream = torch.cuda.current_stream(self.device)
            for value in self.values:
                if isinstance(value, torch.Tensor):
                    value.record_stream(stream)
            if self.workspace is not None:
                self.workspace.record_stream(stream)

    def bind_row(self, row_id):
        from copy import copy
        self.compile()
        row = copy(self)
        row._forced = True
        row.row=next((r for r in self.rows if r["id"]==row_id), None)
        if row.row is None:
            raise ValueError("unknown Triton row")
        return row

    def tuning_report(self):
        from runtime.kir.binding.selection import selection

        return selection(
            self, self.row["id"] if self.row else None, self.timings
        )

    def result(self):
        return self.outputs[0] if len(self.outputs) == 1 else tuple(self.outputs)
