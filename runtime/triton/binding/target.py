import torch
from runtime.triton.launch.call import Artifact


class Target:
    def __init__(self):
        self.devices = {}

    def specialize(self, device, values):
        if device is None or device.type != "cuda":
            raise ValueError("Triton requires local CUDA buffers")
        matrix = next(v for v in values if isinstance(v, torch.Tensor) and v.ndim == 2)
        m, n = matrix.shape
        limit = (1 << max(0, n - 1).bit_length()) if n <= 8192 else 0
        return (device, limit), (m, n)

    def options(self, key):
        device, width = key
        if device not in self.devices:
            self.devices[device] = torch.cuda.get_device_properties(device)
        p = self.devices[device]
        return dict(
            sm=p.major * 10 + p.minor,
            warp_size=p.warp_size,
            max_threads=p.max_threads_per_block,
            width_bound=width,
        )

    @staticmethod
    def materialize(data):
        return Artifact(data)
