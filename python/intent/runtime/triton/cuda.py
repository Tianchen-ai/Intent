"""CUDA execution facilities for the common Triton program binder."""
from ..diagnostics import resource


class CUDADevice:
    name = "triton"
    compile_options = {"enable_fp_fusion": True, "enable_reflect_ftz": False}

    @staticmethod
    def allocate_descriptor(size, alignment, stream):
        import torch
        buffer = torch.empty(size, dtype=torch.int8, device="cuda")
        if buffer.data_ptr() % alignment:
            raise RuntimeError("Triton descriptor allocator returned a misaligned buffer")
        return buffer

    @staticmethod
    def benchmark(hooks, function, quantiles):
        import torch
        from triton.runtime import driver
        function()
        cache = driver.active.get_empty_cache_for_benchmark()
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        try:
            hooks._measurement = cache, start, end
            with torch.cuda.stream(stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    function()
                samples = []
                for _ in range(33):
                    graph.replay()
                    end.synchronize()
                    samples.append(start.elapsed_time(end))
                stream.synchronize()
            values = torch.tensor(samples[1:], dtype=torch.float64, device="cpu")
            return values.quantile(torch.tensor(quantiles, dtype=torch.float64, device="cpu")).tolist()
        finally:
            hooks._measurement = None
            torch.cuda.current_stream().wait_stream(stream)

    @staticmethod
    def resources(kernel):
        metadata = kernel.metadata
        fields = (
            ("registers_per_thread", getattr(kernel, "n_regs", None), "registers", "CompiledKernel.n_regs", "native_loading"),
            ("local_memory_words_per_thread", getattr(kernel, "n_spills", None), "32-bit words",
             "CompiledKernel.n_spills (CUDA LOCAL_SIZE_BYTES / 4)", "native_loading"),
            ("shared_memory_bytes", getattr(metadata, "shared", None), "bytes", "CompiledKernel.metadata.shared", "native_compilation"),
            ("max_threads_per_block", getattr(kernel, "n_max_threads", None), "threads", "CompiledKernel.n_max_threads", "native_loading"),
        )
        return tuple(resource(name, value, unit, "triton." + source, stage,
                              unavailable="The selected Triton compiled kernel does not expose this field")
                     for name, value, unit, source, stage in fields)
