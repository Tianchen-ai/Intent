from threading import RLock
from time import perf_counter
import torch
import triton
from runtime.block.binding.grid import accepts, grid
from runtime.triton.cache.source import load


class Artifact:
    def __init__(self, data):
        if data["version"] != 2:
            raise ValueError("unsupported Intent Triton ABI version")
        self.data = data
        self.kernel = load(data["source"], data["entry"])
        self.winners = {}
        self.lock = RLock()

    def accepts(self, shape):
        return accepts(self.data["guard"], shape)

    def bind(self, values, outputs, device, shape, key):
        m, n = shape
        return Prepared(self, values, outputs, device, m, n)


class Prepared:
    def __init__(self, artifact, values, outputs, device, m, n):
        self.artifact, self.values, self.outputs = artifact, values, outputs
        self.device, self.m, self.n = device, m, n
        self.flat = self.flatten(values)
        if not artifact.accepts((m, n)):
            raise ValueError("width lies outside the compiled shape class")
        self.rows = artifact.data["rows"]
        if not self.rows:
            raise ValueError("no row satisfies the bound width")
        self.launchers = {}
        self.builds = []
        self.timings = {r["id"]: [] for r in self.rows}
        self.ready = False
        self.row = self.rows[0] if len(self.rows) == 1 else None

    def flatten(self, values):
        args = []
        for value in values:
            args.append(value)
            if isinstance(value, torch.Tensor):
                if isinstance(self.artifact.data["domain"]["inner"],list):
                    if value.ndim>1: args.extend(value.shape)
                elif value.ndim>2: args.extend(value.shape[:-1])
                args.extend(value.stride())
        return (*args, self.m, self.n)

    def compile(self):
        if self.ready:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("compile before graph capture")
        with torch.cuda.device(self.device):
            for r in self.rows:
                start = perf_counter()
                native = self.artifact.kernel.warmup(
                    *self.flat,
                    ROWS=r["rows"],
                    WIDTH=r["width"],
                    grid=grid(self.artifact.data["grid"], (self.m, self.n), r),
                    enable_fp_fusion=False,
                    **r["native"],
                )
                self.launchers[r["id"]] = native[
                    grid(self.artifact.data["grid"], (self.m, self.n), r)
                ]
                self.builds.append({"id": r["id"], "milliseconds": (perf_counter() - start) * 1000,
                                    "native_hash": native.hash})
        self.ready = True

    def compilation_report(self):
        self.compile()
        return {"candidate_count": len(self.rows), "rows": list(self.builds)}

    def _invoke(self, row, args):
        self.launchers[row["id"]](*args, row["rows"], row["width"])

    def launch(self):
        with torch.cuda.device(self.device):
            self.compile()
            if self.m == 0 or self.n == 0:
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
                            cuda_trials,
                        )
                        from statistics import median

                        shadow = TrialBindings(self.values, self.artifact.data["abi"])
                        trial = self.flatten(shadow.values)
                        times = {}
                        for r in self.rows:

                            def invoke():
                                self._invoke(r, trial)

                            if shadow.restores:
                                times[r["id"]] = cuda_trials(
                                    shadow,
                                    torch.cuda.current_stream(self.device),
                                    invoke,
                                )
                            else:
                                times[r["id"]] = [
                                    1000
                                    * triton.testing.do_bench(invoke, warmup=3, rep=10)
                                ]
                        best = min(self.rows, key=lambda r: median(times[r["id"]]))
                        self.artifact.winners[key] = best, times
                    self.row, self.timings = self.artifact.winners[key]
            self._invoke(self.row, self.flat)
            stream = torch.cuda.current_stream(self.device)
            for value in self.values:
                if isinstance(value, torch.Tensor):
                    value.record_stream(stream)

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
