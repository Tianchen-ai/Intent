"""Bind declared directions, dimensions, device and storage for one call."""


class Signature:
    def __init__(self, arguments):
        self.args = arguments

    def bind(self, inputs, outputs):
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
            if not isinstance(value, torch.Tensor):
                raise ValueError("expected a tensor argument")
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
                    isinstance(v, torch.Tensor)
                    and y.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
                    for v in bound.values()
                )
            ):
                raise ValueError(
                    "this row implementation needs distinct writable storage"
                )
            bound[arg["name"]] = y
            result.append(y)
        vals = tuple(bound[a["name"]] for a in self.args)
        return vals, result, device

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
