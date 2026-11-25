from threading import RLock
from runtime.kir.binding.completion import PreparedLaunch
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
        if data["version"] not in (1, 2, 3):
            raise ValueError("unsupported cuTile ABI")
        self.data = data
        self.function = load(data["source"], data["entry"])._pyfunc
        pre = data.get("pre_stage")
        self.pre_function = (
            load(data["source"], pre["entry"])._pyfunc if pre else None
        )
        self.lock = RLock()
        self.variants = {}
        self.pre_variant = None
        self.compiled = {}
        self.winners = {}

    def accepts(self, shape):
        return accepts(self.data["guard"], shape)

    def bind(self, values, outputs, device, shape, key):
        m, n = shape
        return Call(self, values, outputs, device, (m, n))


_WS_DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}


class Call(PreparedLaunch):
    def __init__(self, artifact, values, outputs, device, shape):
        self.artifact, self.values, self.outputs = artifact, tuple(values), outputs
        self.device, self.shape = device, shape
        self.pre_stage = artifact.data.get("pre_stage")
        self.library_gemm = bool(artifact.data.get("library_gemm"))
        self.workspace = None
        self.pre_dispatcher = None
        if self.pre_stage:
            ws = self.pre_stage["workspace"]
            self.workspace = torch.empty(
                tuple(ws["shape"]), dtype=_WS_DTYPES[ws["dtype"]], device=device
            )
        self.candidates = {}
        self.compilation = []
        self.selected = None
        self.failures = {}
        self.timings = {}
        self.native_values = self._views(self.values)

    def _views(self, values):
        if not self.artifact.data.get("native_views"):
            return values
        result = list(values)
        for slot, shape in self.artifact.data["native_views"].items():
            result[int(slot)] = result[int(slot)].view(tuple(shape))
        return tuple(result)

    def pre_args(self, values):
        return (
            *(values[i] for i in self.pre_stage["arg_slots"]),
            self.workspace,
            self.pre_stage["rows"],
            self.pre_stage["width"],
        )

    def args(self, row, values):
        if self.artifact.data.get("matrix_views") or self.artifact.data.get("direct_views"):
            values = self.native_values if values is self.values else self._views(values)
            if self.workspace is not None:
                return (*values, self.workspace, row["rows"], row["width"])
            return (*values,row["rows"],row["width"])
        slot=self.artifact.data.get("outer_shape_slot", -1)
        outer=values[slot].shape[:-1] if slot>=0 else ()
        columns=((-(-self.shape[1]//row["width"])),) if self.artifact.data.get("column_tiles_parameter") else ()
        return (*values, *outer, *columns, row["rows"], row["width"])

    def grid(self, row):
        return grid(self.artifact.data["grid"], self.shape, row)

    def _compile_key(self):
        base = self.args(self.artifact.data["rows"][0], self.values)[:-2]
        sptrs = [
            a.untyped_storage().data_ptr()
            for a in base
            if isinstance(a, torch.Tensor)
        ]
        shared = len(set(sptrs)) != len(sptrs)
        seen = {}
        items = []
        for a in base:
            if isinstance(a, torch.Tensor):
                alias = None
                if shared:
                    sp = a.untyped_storage().data_ptr()
                    alias = (seen.setdefault(sp, len(seen)), a.storage_offset())
                items.append(
                    (
                        a.dtype,
                        tuple(a.shape),
                        tuple(a.stride()),
                        a.data_ptr() % 16 == 0,
                        alias,
                    )
                )
            else:
                items.append((type(a), a if isinstance(a, (int, bool)) else None))
        return (self.device, self.shape, tuple(items))

    def compile(self):
        if self.candidates:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("compile cuTile before graph capture")
        a = self.artifact
        ckey = self._compile_key()
        with a.lock:
            cached = a.compiled.get(ckey)
            if cached is not None:
                self.candidates, comp, self.failures, self.pre_dispatcher = cached
                self.compilation = [
                    dict(c, compiler_ms=0.0, dispatcher_hit=True) for c in comp
                ]
                self.compiles_before_launch = sum(
                    d.compiles for _, d in self.candidates.values() if d is not None
                ) + (self.pre_dispatcher.compiles if self.pre_dispatcher is not None else 0)
                return
        with torch.cuda.device(self.device), a.lock:
            cached = a.compiled.get(ckey)
            if cached is not None:
                self.candidates, comp, self.failures, self.pre_dispatcher = cached
                self.compilation = [
                    dict(c, compiler_ms=0.0, dispatcher_hit=True) for c in comp
                ]
                self.compiles_before_launch = sum(
                    d.compiles for _, d in self.candidates.values() if d is not None
                ) + (self.pre_dispatcher.compiles if self.pre_dispatcher is not None else 0)
                return
            arch = torch.cuda.get_device_capability(self.device)
            if self.pre_stage is not None:
                if a.pre_variant is None:
                    a.pre_variant = BoundKernel(
                        a.pre_function, hints=self.pre_stage["native"], arch=arch
                    )
                self.pre_dispatcher = a.pre_variant
                pre_record = self.pre_dispatcher.prepare_native(
                    self.pre_args(self.values)
                )
                self.compilation.append(dict(row="pre_stage", **pre_record))
            if self.library_gemm:
                for r in a.data["rows"]:
                    self.candidates[r["id"]] = (r, None)
                self.compiles_before_launch = (
                    self.pre_dispatcher.compiles if self.pre_dispatcher is not None else 0
                )
                a.compiled[ckey] = (
                    dict(self.candidates),
                    list(self.compilation),
                    dict(self.failures),
                    self.pre_dispatcher,
                )
                return
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
                d.compiles for _, d in self.candidates.values() if d is not None
            ) + (self.pre_dispatcher.compiles if self.pre_dispatcher is not None else 0)
            a.compiled[ckey] = (
                dict(self.candidates),
                list(self.compilation),
                dict(self.failures),
                self.pre_dispatcher,
            )

    def _launch_once(self, stream, row, kernel, values):
        if self.pre_stage is not None:
            ct.launch(
                stream,
                tuple(self.pre_stage["grid"]),
                self.pre_dispatcher,
                self.pre_args(values),
            )
        if self.library_gemm:
            torch.matmul(
                values[self.pre_stage["lhs_slot"]],
                self.workspace,
                out=values[self.pre_stage["out_slot"]],
            )
            return
        ct.launch(stream, self.grid(row), kernel, self.args(row, values))

    def enqueue(self):
        a = self.artifact
        with torch.cuda.device(self.device):
            self.compile()
            if not all(self.grid(a.data["rows"][0])):
                return
            stream = torch.cuda.current_stream(self.device)
            if self.selected is None:
                layout = tuple(
                    (v.dtype, tuple(v.shape), tuple(v.stride()), v.data_ptr() % 16 == 0)
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
                            if trials.restores or self.pre_stage is not None:
                                for i, (row, kernel) in self.candidates.items():
                                    self.timings[i] = cuda_trials(
                                        trials,
                                        stream,
                                        lambda: self._launch_once(
                                            stream, row, kernel, trials.values
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
            self._launch_once(stream, r, k, self.values)
            for v in self.values:
                if isinstance(v, torch.Tensor):
                    v.record_stream(stream)
            if self.workspace is not None:
                self.workspace.record_stream(stream)

    def bind_row(self, row_id):
        from copy import copy
        self.compile()
        row = copy(self)
        row._forced = True
        row.selected=self.candidates[row_id]
        return row

    def tuning_report(self):
        from runtime.kir.binding.selection import selection

        times = {
            r["id"]: self.timings.get(r["id"], []) for r in self.artifact.data["rows"]
        }
        return selection(
            self, self.selected[0]["id"] if self.selected else None, times
        )

    def result(self):
        return self.outputs[0] if len(self.outputs) == 1 else tuple(self.outputs)

    def compilation_report(self):
        return dict(
            candidate_count=len(self.artifact.data["rows"]),
            rows=self.compilation,
            failures=dict(self.failures),
            compiles_before_launch=self.compiles_before_launch,
            native_compiles=sum(
                d.compiles for _, d in self.candidates.values() if d is not None
            )
            + (self.pre_dispatcher.compiles if self.pre_dispatcher is not None else 0),
            dispatch_reuses=sum(k.reuses for k in self.artifact.variants.values())
            + (
                self.artifact.pre_variant.reuses
                if self.artifact.pre_variant is not None
                else 0
            ),
        )
