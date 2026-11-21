from threading import RLock
from importlib.metadata import version
import torch
import cuda.tile as ct
from cuda.tile.tune import exhaustive_search
from runtime.cutile.cache.source import load
from runtime.cutile.launch.dispatcher import BoundKernel
from runtime.block.binding.grid import accepts, grid


class Artifact:
    def __init__(self, data):
        if version("cuda-tile").split(".")[:2] != ["1", "6"]:
            raise RuntimeError("Intent cuTile requires SDK 1.6.x")
        if data["version"] != 1:
            raise ValueError("unsupported cuTile ABI")
        self.data = data
        self.function = load(data["source"], data["entry"])._pyfunc
        self.lock = RLock()
        self.variants = {}
        self.winners = {}

    def accepts(self, shape):
        return accepts(self.data["guard"], shape)

    def bind(self, values, outputs, device, m, n):
        return Call(self, values, outputs, device, (m, n))


class Call:
    def __init__(self, artifact, values, outputs, device, shape):
        self.artifact, self.values, self.outputs = artifact, tuple(values), outputs
        self.device, self.shape = device, shape
        self.candidates = {}
        self.compilation = []
        self.selected = None
        self.failures = {}
        self.timings = {}

    def args(self, row, values):
        return (*values, row["rows"], row["width"])

    def grid(self, row):
        return grid(self.artifact.data["grid"], self.shape, row)

    def compile(self):
        if self.candidates:
            return
        a = self.artifact
        with torch.cuda.device(self.device), a.lock:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("compile cuTile before graph capture")
            arch = torch.cuda.get_device_capability(self.device)
            for r in a.data["rows"]:
                id = r["id"]
                if id not in a.variants:
                    a.variants[id] = BoundKernel(
                        a.function, hints=r["native"], arch=arch
                    )
                dispatcher = a.variants[id]
                try:
                    record = dispatcher.prepare_native(self.args(r, self.values))
                except Exception as exc:
                    self.failures[id] = str(exc)
                    continue
                self.compilation.append(dict(row=id, **record))
                self.candidates[id] = (r, dispatcher)
            if not self.candidates:
                raise RuntimeError(f"cuTile native compilation failed: {self.failures}")
            self.compiles_before_launch = sum(
                d.compiles for _, d in self.candidates.values()
            )

    def launch(self):
        a = self.artifact
        with torch.cuda.device(self.device):
            self.compile()
            if not all(self.shape):
                return
            stream = torch.cuda.current_stream(self.device)
            if self.selected is None:
                layout = tuple(
                    (v.dtype, tuple(v.stride()), v.data_ptr() % 256)
                    for v in self.values
                    if isinstance(v, torch.Tensor)
                )
                key = (self.shape, layout, tuple(self.candidates))
                with a.lock:
                    winner = a.winners.get(key)
                    if winner is None:
                        if torch.cuda.is_current_stream_capturing():
                            raise RuntimeError("warm up cuTile before capture")
                        if len(self.candidates) == 1:
                            winner = next(iter(self.candidates))
                        else:
                            from runtime.kir.binding.trials import (
                                TrialBindings,
                                cuda_trials,
                            )
                            from statistics import median

                            trials = TrialBindings(self.values, a.data["abi"])
                            if trials.restores:
                                for i, (row, kernel) in self.candidates.items():
                                    self.timings[i] = cuda_trials(
                                        trials,
                                        stream,
                                        lambda: ct.launch(
                                            stream,
                                            self.grid(row),
                                            kernel,
                                            self.args(row, trials.values),
                                        ),
                                    )
                                winner = min(
                                    self.candidates,
                                    key=lambda i: median(self.timings[i]),
                                )
                            else:
                                result = exhaustive_search(
                                    tuple(self.candidates),
                                    stream,
                                    lambda i: self.grid(self.candidates[i][0]),
                                    lambda i: self.candidates[i][1],
                                    lambda i: self.args(
                                        self.candidates[i][0], trials.values
                                    ),
                                    quiet=True,
                                )
                                self.failures.update(
                                    (i, msg) for i, _, msg in result.failures
                                )
                                winner = result.best.config
                                self.timings = {
                                    m.config: [m.mean_us] for m in result.successes
                                }
                        a.winners[key] = winner, self.timings
                    winner, self.timings = a.winners[key]
                    self.selected = self.candidates[winner]
            r, k = self.selected
            ct.launch(stream, self.grid(r), k, self.args(r, self.values))
            for v in self.values:
                if isinstance(v, torch.Tensor):
                    v.record_stream(stream)

    def tuning_report(self):
        from runtime.kir.binding.selection import selection

        times = {
            r["id"]: self.timings.get(r["id"], []) for r in self.artifact.data["rows"]
        }
        return selection(
            self.artifact.data, self.selected[0]["id"] if self.selected else None, times
        )

    def result(self):
        return self.outputs[0] if len(self.outputs) == 1 else tuple(self.outputs)

    def compilation_report(self):
        return dict(
            candidate_count=len(self.artifact.data["rows"]),
            rows=self.compilation,
            failures=dict(self.failures),
            compiles_before_launch=self.compiles_before_launch,
            native_compiles=sum(d.compiles for _, d in self.candidates.values()),
            dispatch_reuses=sum(k.reuses for k in self.artifact.variants.values()),
        )
