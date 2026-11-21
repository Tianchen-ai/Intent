import ast
import json
from threading import RLock
from .signature import signature


class Kernel:
    def __init__(self, compiler, definition, backend):
        self.compiler = compiler
        self.source = definition.source
        self.root = None
        self.args = signature(ast.parse(definition.source).body[0])
        self.exit = compiler.runtime(backend, "launch.call")
        self.backend = backend
        self.artifacts = {}
        self.devices = {}
        self.lock = RLock()

    def prepare(self, *inputs, outputs=None):
        import torch

        ins = [a for a in self.args if a["direction"] != "Out"]
        if len(ins) != len(inputs):
            raise TypeError(f"expected {len(ins)} inputs")
        shapes, bound = {}, {}
        device = None
        for arg, value in zip(ins, inputs):
            bound[arg["name"]] = value
            if arg["direction"] == "Scalar":
                if not isinstance(value, (int, float)):
                    raise TypeError(f"expected host scalar for {arg['name']}")
                bound[arg["name"]] = float(value)
                continue
            if not isinstance(value, torch.Tensor) or not value.is_cuda:
                raise ValueError("Triton input requires a local CUDA tensor")
            if device is not None and value.device != device:
                raise ValueError("arguments span multiple devices")
            device = value.device
            self._view(arg, value, shapes, device)
        out_args = [a for a in self.args if a["direction"] == "Out"]
        supplied = (
            ()
            if outputs is None
            else ((outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs))
        )
        if outputs is not None and len(supplied) != len(out_args):
            raise ValueError("output count differs from signature")
        result = []
        for i, arg in enumerate(out_args):
            shape = [shapes[n] for n in arg["dims"]]
            y = (
                supplied[i]
                if supplied
                else torch.empty(shape, device=device, dtype=self.dtype(arg))
            )
            self._view(arg, y, shapes, device)
            if y.numel() and (
                not self.non_overlapping(y)
                or any(
                    isinstance(v, torch.Tensor) and torch._C._overlaps(y, v)
                    for v in bound.values()
                )
            ):
                raise ValueError(
                    "this row implementation needs distinct writable storage"
                )
            bound[arg["name"]] = y
            result.append(y)
        vals = tuple(bound[a["name"]] for a in self.args)
        matrix = next(v for v in vals if isinstance(v, torch.Tensor) and v.ndim == 2)
        m, n = matrix.shape
        limit = (1 << max(0, n - 1).bit_length()) if n <= 8192 else 0
        key = (device.index, limit)
        with self.lock:
            if key not in self.artifacts:
                if device not in self.devices:
                    self.devices[device] = torch.cuda.get_device_properties(device)
                properties = self.devices[device]
                options = dict(
                    sm=10 * properties.major + properties.minor,
                    width_bound=limit,
                    max_threads=properties.max_threads_per_block,
                    warp_size=properties.warp_size,
                )
                with self.compiler.derive(
                    {"exits": {self.backend: {"options": options}}}
                ) as target:
                    if self.root is None:
                        self.root = self.compiler.root_ir(self.source)
                    artifact = json.loads(target.compile(self.root, exit=self.backend))
                if artifact["abi"] != self.args:
                    raise ValueError("compiled ABI differs from author signature")
                self.artifacts[key] = self.exit.Artifact(artifact)
            artifact = self.artifacts[key]
        return artifact.bind(vals, result, device, m, n)

    @staticmethod
    def non_overlapping(view):
        # For a 2-D view the shortest collision solves i*s0 == j*s1.
        from math import gcd

        dims = [(n, s) for n, s in zip(view.shape, view.stride()) if n > 1]
        if any(s == 0 for _, s in dims):
            return False
        if len(dims) < 2:
            return True
        (m, a), (n, b) = dims
        g = gcd(a, b)
        return m <= b // g or n <= a // g

    @staticmethod
    def dtype(arg):
        import torch

        return {"f16": torch.float16, "f32": torch.float32}[arg["dtype"]]

    def _view(self, arg, value, shapes, device):
        if (
            value.device != device
            or value.dtype != self.dtype(arg)
            or value.ndim != len(arg["dims"])
        ):
            raise ValueError(f"view signature mismatch for {arg['name']}")
        for name, extent in zip(arg["dims"], value.shape):
            if shapes.setdefault(name, extent) != extent:
                raise ValueError(f"inconsistent dimension {name}")

    def run(self, *inputs):
        call = self.prepare(*inputs)
        call.launch()
        return call.result()

    def __call__(self, *args):
        if len(args) != len(self.args):
            raise TypeError("expected all arguments in signature order")
        ins, outs = [], []
        for arg, v in zip(self.args, args):
            (outs if arg["direction"] == "Out" else ins).append(v)
        self.prepare(*ins, outputs=outs).launch()
