"""Public setup descriptions; compilation still uses the existing Target API."""

BACKENDS = {
    "triton": {
        "modules": ("torch", "triton"),
        "requirements": ("triton==3.6.0", "numpy==1.26.4"), "torch": "2.10.0+cu130",
        "torch_index": "https://download.pytorch.org/whl/cu130",
        "python_max": (3, 12), "toolchain": "NVIDIA driver and CUDA-capable PyTorch",
    },
    "cutile": {
        "modules": ("torch", "cuda.tile", "cuda.tile.tune"),
        "requirements": ("triton==3.6.0", "numpy==1.26.4",
                         "cuda-toolkit[tileiras,nvvm,nvcc]==13.3.1", "cuda-tile==1.6.0"),
        "torch": "2.10.0+cu128",
        "torch_index": "https://download.pytorch.org/whl/cu128",
        "python_max": (3, 12), "toolchain": "NVIDIA driver and the CUDA tile compiler",
    },
    "mojo": {
        "modules": ("torch",),
        "requirements": (), "torch": "2.10.0+cpu",
        "torch_index": "https://download.pytorch.org/whl/cpu",
        "python_max": (3, 12), "toolchain": "An explicitly installed Mojo compiler; Linux x86-64 AVX2/AVX512",
    },
    "weft": {
        "modules": (),
        "requirements": (), "torch": None, "torch_index": None,
        "python_max": (3, 12), "toolchain": "An Intent compiler built with Weft; public generation requires explicit vector_bits and workers",
    },
    "bangc": {
        "modules": (),
        "requirements": (), "torch": None, "torch_index": None,
        "python_max": (3, 12), "toolchain": "An explicitly installed NeuWare SDK and MLU370 runtime",
    },
}


def backend(name: str) -> dict:
    from intent.targets.provider import provider
    adapter = provider(name)
    return {**BACKENDS[name], "target": adapter.target.__name__}


def make_target(name: str, options: dict, *, facts: dict | None = None):
    from intent.targets.provider import provider
    from intent.targets.specification import read_compilation_target

    if facts is not None:
        if options:
            raise ValueError("explicit compiler facts do not accept local runtime target options")
        return read_compilation_target(name, facts)
    return provider(name).target(**options)
