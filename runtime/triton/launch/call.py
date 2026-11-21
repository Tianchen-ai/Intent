from threading import RLock
import torch
import triton
from runtime.block.binding.grid import grid
from runtime.triton.cache.source import load


class Artifact:
    def __init__(self, data):
        if data["version"] != 1:
            raise ValueError("unsupported Intent Triton ABI version")
        self.data = data
        self.kernel = load(data["source"], data["entry"])
        self.winners = {}
        self.lock = RLock()

    def bind(self, values, outputs, device, m, n):
        return Prepared(self, values, outputs, device, m, n)


class Prepared:
    def __init__(self, artifact, values, outputs, device, m, n):
        self.artifact, self.values, self.outputs = artifact, values, outputs
        self.device, self.m, self.n = device, m, n
        self.flat = self.flatten(values)
        g = artifact.data["guard"]
        if n < g["min_width"] or (g["max_width"] is not None and n > g["max_width"]):
            raise ValueError("width lies outside the compiled shape class")
        self.rows = artifact.data["rows"]
        if not self.rows:
            raise ValueError("no row satisfies the bound width")
        self.launchers = {}
        self.ready = False
        self.row = self.rows[0] if len(self.rows) == 1 else None

    def flatten(self, values):
        args = []
        for value in values:
            args.append(value)
            if isinstance(value, torch.Tensor):
                args.extend(value.stride())
        return (*args, self.m, self.n)

    def compile(self):
        if self.ready:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("compile before graph capture")
        with torch.cuda.device(self.device):
            for r in self.rows:
                native = self.artifact.kernel.warmup(
                    *self.flat,
                    ROWS=r["rows"],
                    WIDTH=r["width"],
                    grid=grid(self.m, r),
                    enable_fp_fusion=False,
                    **r["native"],
                )
                self.launchers[r["id"]] = native[grid(self.m, r)]
        self.ready = True

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
                        shadow = [
                            (
                                torch.empty_strided(
                                    v.shape, v.stride(), dtype=v.dtype, device=v.device
                                )
                                if any(v is y for y in self.outputs)
                                else v
                            )
                            for v in self.values
                        ]
                        trial = self.flatten(shadow)
                        self.artifact.winners[key] = min(
                            self.rows,
                            key=lambda r: triton.testing.do_bench(
                                lambda: self._invoke(r, trial), warmup=3, rep=10
                            ),
                        )
                    self.row = self.artifact.winners[key]
            self._invoke(self.row, self.flat)
            stream = torch.cuda.current_stream(self.device)
            for value in self.values:
                if isinstance(value, torch.Tensor):
                    value.record_stream(stream)

    def result(self):
        return self.outputs[0] if len(self.outputs) == 1 else tuple(self.outputs)
