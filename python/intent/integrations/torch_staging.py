"""Explicit host transfer for examples and applications using Torch storage."""
from intent.language.dtypes import dtype
from .torch import torch_dtype


class TorchStorage:
    copies_host = True
    contiguous_views = False

    def __init__(self, device):
        import torch
        self.torch = torch
        self.device = torch.device(str(device))

    def accepts(self, value):
        return isinstance(value, self.torch.Tensor)

    def allocate(self, shape, element):
        return self.torch.empty(shape, dtype=torch_dtype(dtype(element)), device=self.device)

    def upload(self, host, element):
        raw = self.torch.frombuffer(host, dtype=self.torch.uint8) if host.nbytes else self.torch.empty(0, dtype=self.torch.uint8)
        return raw.to(self.device)

    def view(self, allocation, shape, strides, offset, element):
        return allocation.view(torch_dtype(dtype(element))).as_strided(shape, strides, storage_offset=offset)

    def download(self, allocation):
        return allocation.detach().cpu().numpy()

    def snapshot(self, value):
        storage = value.untyped_storage()
        key = str(value.device), storage.data_ptr(), storage.nbytes()
        raw = self.torch.empty(0, dtype=self.torch.uint8, device=value.device).set_(storage, 0, (storage.nbytes(),), (1,))
        saved = raw.clone()
        return key, lambda: raw.copy_(saved)

    def release(self, value):
        pass  # Tensor owners retain their allocator and stream lifetimes.

    def prepare_fill(self, value, scalar):
        from intent.runtime.native_call import NativeMemoryFill
        from intent.runtime.sequence import PreparedInitialization
        if not self.accepts(value) or value.device != self.device:
            raise ValueError("initialization must use the selected device's tensor")
        native = None
        if value.device.type == "cpu":
            if not value.is_contiguous():
                raise NotImplementedError("native CPU initialization requires contiguous storage")
            encoded = bytes(self.torch.tensor([scalar], dtype=value.dtype).view(self.torch.uint8).tolist())
            native = NativeMemoryFill(value.data_ptr(), value.numel(), encoded, (value,), lambda: None)
        return PreparedInitialization(lambda: value.fill_(scalar), (value,), native)
