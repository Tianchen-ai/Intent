import torch
from runtime.cutile.launch.call import Artifact


class Target:
    def specialize(self, device, values):
        if getattr(device, "type", None) != "cuda":
            raise ValueError("cuTile requires local CUDA arrays")
        x = next(v for v in values if isinstance(v, torch.Tensor) and v.ndim == 2)
        # Bounds include the padded coordinates visited by the last tile.
        limit = (1 << 31) - 1
        narrow = x.shape[0] + 1 <= limit and x.shape[1] + 4095 <= limit
        tensor_map = True
        for v in values:
            if isinstance(v, torch.Tensor):
                tensor_map &= (
                    v.data_ptr() % 16 == 0
                    and v.stride()[-1] == 1
                    and all((s * v.element_size()) % 16 == 0 for s in v.stride()[:-1])
                )
                narrow &= (
                    all(0 <= s <= limit for s in v.stride())
                    and sum(max(0, n - 1) * s for n, s in zip(v.shape, v.stride()))
                    <= limit
                )
        return (device, 32 if narrow else 64, tensor_map), tuple(x.shape)

    def options(self, key, shape):
        device, bits, tensor_map = key
        arch = torch.cuda.get_device_capability(device)
        return dict(
            sm=10 * arch[0] + arch[1],
            observed_width=shape[1],
            index_bits=bits,
            allow_tma=tensor_map,
        )

    def materialize(self, data):
        return Artifact(data)
