from threading import RLock
from statistics import median
import torch
from runtime.block.binding.grid import accepts, grid
from runtime.bangc.cache.native import Native


class Artifact:
    def __init__(self, data):
        if data["version"] != 2 or not data["rows"]:
            raise ValueError("unsupported BANG C artifact")
        self.data = data
        self.variants = {}
        self.winners = {}
        self.lock = RLock()

    def accepts(self, shape):
        return accepts(self.data["guard"], shape)

    def bind(self, values, outputs, device, m, n):
        return Call(self, values, outputs, device, (m, n))


class Call:
    def __init__(self, artifact, values, outputs, device, shape):
        self.artifact, self.values, self.outputs, self.device, self.shape = (
            artifact,
            tuple(values),
            outputs,
            device,
            shape,
        )
        self.records = []
        self.timings = {}
        self.selected = None

    def compile(self):
        a = self.artifact
        with a.lock:
            if self.records:
                return
            report = []
            for row in a.data["rows"]:
                i = row["id"]
                hit = i in a.variants
                if not hit:
                    a.variants[i] = Native(a.data["source"], row, a.data["abi"])
                report.append(
                    dict(
                        row=i,
                        seconds=0 if hit else a.variants[i].seconds,
                        cache_hit=hit,
                    )
                )
            self.records = report

    def compilation_report(self):
        return dict(candidate_count=len(self.artifact.data["rows"]), rows=self.records)

    def _args(self, values):
        args = []
        for v in values:
            if isinstance(v, torch.Tensor):
                args.extend((v.data_ptr(), *v.shape, *v.stride()))
            else:
                args.append(v)
        return args

    def _enqueue(self, row, args, stream):
        a = self.artifact
        groups = grid(a.data["grid"], self.shape, row)[0]
        status = a.variants[row["id"]].launch(
            *args, *self.shape, groups, row["tasks"], stream.mlu_stream
        )
        if status:
            raise RuntimeError(
                f"MLU row {row['id']} enqueue on {self.device} failed with CNRT status {status}"
            )

    def launch(self):
        self.compile()
        if not all(self.shape):
            return
        a = self.artifact
        with torch.mlu.device(self.device):
            stream = torch.mlu.current_stream(self.device)
            if self.selected is None:
                key = (
                    self.shape,
                    tuple(
                        (v.dtype, tuple(v.stride()))
                        for v in self.values
                        if isinstance(v, torch.Tensor)
                    ),
                )
                with a.lock:
                    if key not in a.winners:
                        rows = a.data["rows"]
                        durations = {row["id"]: [] for row in rows}
                        if len(rows) > 1:
                            from runtime.kir.binding.trials import TrialBindings

                            trials = TrialBindings(self.values, a.data["abi"])
                            args = self._args(trials.values)
                            for row in rows:
                                trials.reset()
                                self._enqueue(row, args, stream)
                            for _ in range(3):
                                for row in rows:
                                    before, after = torch.mlu.Event(
                                        enable_timing=True
                                    ), torch.mlu.Event(enable_timing=True)
                                    trials.reset()
                                    before.record(stream)
                                    self._enqueue(row, args, stream)
                                    after.record(stream)
                                    after.synchronize()
                                    durations[row["id"]].append(
                                        before.hardware_time(after)
                                    )
                            winner = min(rows, key=lambda r: median(durations[r["id"]]))
                        else:
                            winner = rows[0]
                        a.winners[key] = winner, durations
                    self.selected, self.timings = a.winners[key]
            self._enqueue(self.selected, self._args(self.values), stream)
            for v in self.values:
                if isinstance(v, torch.Tensor):
                    v.record_stream(stream)

    def tuning_report(self):
        from runtime.kir.binding.selection import selection

        return selection(
            self.artifact.data,
            self.selected["id"] if self.selected else None,
            self.timings,
        )

    def result(self):
        return self.outputs[0] if len(self.outputs) == 1 else self.outputs
