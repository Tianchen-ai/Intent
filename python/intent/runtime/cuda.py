"""CUDA invocation context shared by the NVIDIA source providers."""
from contextlib import nullcontext


def device_context(device):
    import torch
    return nullcontext() if torch.cuda.current_device() == device.index else torch.cuda.device(device)
