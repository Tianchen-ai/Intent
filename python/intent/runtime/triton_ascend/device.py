from contextlib import nullcontext
from ..diagnostics import unavailable_resources


def device_context(device):
    import torch
    return nullcontext() if torch.npu.current_device() == device.index else torch.npu.device(device)


class AscendDevice:
    name = "triton_ascend"
    compile_options = {"enable_fp_fusion": True, "default_dot_input_precision": "ieee"}

    @staticmethod
    def allocate_descriptor(size, alignment, stream):
        raise NotImplementedError("Triton-Ascend does not use CUDA tensor descriptors")

    @staticmethod
    def benchmark(hooks, function, quantiles):
        import torch
        from triton.runtime import driver
        function()
        cache = driver.active.get_empty_cache_for_benchmark()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        samples = []
        try:
            hooks._measurement = cache, start, end
            for _ in range(33):
                function()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            values = torch.tensor(samples[1:], dtype=torch.float64, device="cpu")
            return values.quantile(torch.tensor(quantiles, dtype=torch.float64, device="cpu")).tolist()
        finally:
            hooks._measurement = None

    @staticmethod
    def resources(kernel):
        return unavailable_resources("triton_ascend", "CANN owns storage allocation; this launcher does not export physical capacity usage")
