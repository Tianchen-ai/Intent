import torch
from runtime.triton.launch.call import Artifact


class Target:
    def __init__(self):
        self.devices = {}

    def profile(self, device, buffers):
        if getattr(device, "type", None) != "cuda":
            raise ValueError("Triton requires local CUDA buffers")
        return device

    def options(self, device):
        if device not in self.devices:
            self.devices[device] = torch.cuda.get_device_properties(device)
        p = self.devices[device]
        return dict(
            calibration={"NVIDIA H100 80GB HBM3":"sm90","NVIDIA GeForce RTX 5090 D":"sm120"}.get(torch.cuda.get_device_name(device),""), sm_count=p.multi_processor_count,
            sm=p.major * 10 + p.minor,
            warp_size=p.warp_size,
            max_threads=p.max_threads_per_block,
        )

    @staticmethod
    def materialize(data):
        return Artifact(data)
