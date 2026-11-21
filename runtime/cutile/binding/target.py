from math import prod
import torch
from runtime.cutile.launch.call import Artifact


class Target:
    def profile(self, device, buffers):
        if getattr(device, "type", None) != "cuda":
            raise ValueError("cuTile requires local CUDA arrays")
        limit = (1 << 31) - 1
        narrow = True
        tensor_map = True
        for view in buffers:
            if not isinstance(view, torch.Tensor):
                continue
            shape, strides = view.shape, view.stride()
            tensor_map &= view.data_ptr() % 16 == 0 and strides[-1] == 1 and all(
                stride * view.element_size() % 16 == 0 for stride in strides[:-1])
            narrow &= (prod(shape[:-1])+1<=limit and shape[-1]+4095<=limit
                       and all(0<=s<=limit for s in strides)
                       and sum(max(0,n-1)*s for n,s in zip(shape,strides))<=limit)
        return device, 32 if narrow else 64, tensor_map

    def options(self, key):
        device, bits, tensor_map = key
        arch = torch.cuda.get_device_capability(device)
        return dict(
            calibration={"NVIDIA H100 80GB HBM3":"sm90","NVIDIA GeForce RTX 5090 D":"sm120"}.get(torch.cuda.get_device_name(device),""), sm_count=torch.cuda.get_device_properties(device).multi_processor_count,
            sm=10 * arch[0] + arch[1],
            index_bits=bits,
            allow_tma=tensor_map,
        )

    def materialize(self, data):
        return Artifact(data)
