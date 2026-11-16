"""Reusable execution constraints, independent of compiler-provider selection."""
from __future__ import annotations

import os


def indexed_key(device):
    return device


def torch_storage(device, options):
    from intent.integrations.torch_staging import TorchStorage
    return TorchStorage(device)


def cpu_key(device):
    return device, tuple(sorted(os.sched_getaffinity(0)))


def cpu_workers(workers):
    available = len(os.sched_getaffinity(0))
    if workers is None:
        workers = min(8, available)
    if type(workers) is not int or not 0 < workers <= available:
        raise ValueError(f"workers must be between 1 and the {available} CPUs allowed by affinity")
    return workers


def indexed_target(target, device, *, workers, **options):
    if workers is not None:
        raise ValueError("workers is a CPU execution budget")
    return target(device=device.index, **options).resolve()
