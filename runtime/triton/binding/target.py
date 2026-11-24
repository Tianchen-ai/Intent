from runtime.kir.binding.completion import submit_device as submit
import torch
from runtime.triton.launch.call import Artifact



class Target:
    submit = staticmethod(submit)

    def __init__(self):
        self.devices = {}

    def profile(self, device, buffers):
        if getattr(device, "type", None) != "cuda":
            raise ValueError("Triton requires local CUDA buffers")
        tensor_map = True
        seen_ptrs = set()
        for view in buffers:
            if not isinstance(view, torch.Tensor):
                continue
            ptr = view.data_ptr()
            if ptr in seen_ptrs:
                tensor_map = False
            seen_ptrs.add(ptr)
            tensor_map &= ptr % 16 == 0
        return device, tensor_map

    def options(self, key):
        if isinstance(key, tuple):
            device, tensor_map = key
        else:
            device, tensor_map = key, False
        if device not in self.devices:
            self.devices[device] = torch.cuda.get_device_properties(device)
        p = self.devices[device]
        return dict(
            registers=p.regs_per_multiprocessor,
            shared_per_block=p.shared_memory_per_block,
            dynamic_shared_per_block=getattr(p,"shared_memory_per_block_optin",p.shared_memory_per_block),
            calibration={"NVIDIA H100 80GB HBM3":"sm90","NVIDIA GeForce RTX 5090 D":"sm120"}.get(torch.cuda.get_device_name(device),""), sm_count=p.multi_processor_count,
            sm=p.major * 10 + p.minor,
            warp_size=p.warp_size,
            max_threads=p.max_threads_per_block,
            allow_tma=tensor_map,
        )

    @staticmethod
    def materialize(data):
        return Artifact(data)
