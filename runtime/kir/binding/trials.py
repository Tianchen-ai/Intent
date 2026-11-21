"""Trial bindings preserve all views of each writable allocation."""

from .views import adapter


def storage_groups(items):
    """Connected components of overlapping storage intervals on one device."""
    groups = []
    for start, end, i in sorted(items):
        if groups and start < groups[-1][1]:
            base, limit, indices = groups[-1]
            indices.append(i)
            groups[-1] = (base, max(limit, end), indices)
        else:
            groups.append((start, end, [i]))
    return groups


class TrialBindings:
    def __init__(self, values, abi):
        groups = {}
        host = []
        self.values = list(values)
        self.restores = []
        for i, (value, slot) in enumerate(zip(values, abi)):
            if slot["direction"] == "Scalar":
                continue
            impl = adapter(value)
            device = impl.describe(value)[3]
            start, size = impl.storage_bounds(value)
            if str(device) == "cpu":
                host.append((start, start + size, i))
                continue
            groups.setdefault((impl, device), []).append((start, start + size, i))
        for start, end, indices in storage_groups(host):
            if all(abi[i]["direction"] == "In" for i in indices):
                continue
            import ctypes
            import numpy as np

            raw = np.ctypeslib.as_array(
                (ctypes.c_uint8 * (end - start)).from_address(start)
            )
            backing = np.empty(end - start + 256, dtype=np.uint8)
            off = (start - backing.ctypes.data) % 256
            shadow = backing[off : off + end - start]
            for i in indices:
                self.values[i] = adapter(values[i]).host_shadow(
                    values[i], shadow, start
                )
            if any(abi[i]["direction"] != "Out" for i in indices):
                self.restores.append(lambda dst=shadow, src=raw: np.copyto(dst, src))
        for (impl, device), items in groups.items():
            for start, end, indices in storage_groups(items):
                if all(abi[i]["direction"] == "In" for i in indices):
                    continue
                shadow, restore = impl.shadow_group([values[i] for i in indices], start, end)
                for i, view in zip(indices, shadow):
                    self.values[i] = view
                if any(abi[i]["direction"] != "Out" for i in indices):
                    self.restores.append(restore)
        self.values = tuple(self.values)

    def reset(self):
        for restore in self.restores:
            restore()


def cuda_trials(trials, stream, submit):
    """Time each stateful invocation after restoring its complete allocation groups."""
    import torch

    times = []
    for _ in range(4):
        trials.reset()
        before = torch.cuda.Event(enable_timing=True)
        after = torch.cuda.Event(enable_timing=True)
        before.record(stream)
        submit()
        after.record(stream)
        after.synchronize()
        times.append(1000.0 * before.elapsed_time(after))
    return times[1:]
