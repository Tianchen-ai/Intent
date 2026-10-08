from __future__ import annotations

from ..specification import GPUCapabilities


def resolve_gpu_device(device: int) -> GPUCapabilities:
    from cuda.bindings import driver
    import torch

    if not torch.cuda.is_available() or device >= torch.cuda.device_count():
        raise RuntimeError("requested CUDA device is unavailable")

    def require_success(result: tuple[object, ...], operation: str) -> object:
        status, *values = result
        if status != driver.CUresult.CUDA_SUCCESS:
            _, name = driver.cuGetErrorName(status)
            raise RuntimeError(f"{operation} failed: {name.decode()}")
        return values[0] if values else None

    require_success(driver.cuInit(0), "cuInit")
    cuda_device = require_success(driver.cuDeviceGet(device), "cuDeviceGet")

    def attribute(name: object) -> int:
        value = require_success(
            driver.cuDeviceGetAttribute(name, cuda_device),
            f"cuDeviceGetAttribute({name.name})",
        )
        return int(value)

    major = attribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
    )
    minor = attribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    )
    return GPUCapabilities(
        compute_units=attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
        ),
        shared_memory_per_unit=attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        ),
        max_dynamic_shared_memory_per_block=attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
        ),
        registers_per_unit=attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR
        ),
        max_threads_per_block=attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
        ),
        compute_capability_major=major,
        compute_capability_minor=minor,
        single_to_double_precision_perf_ratio=attribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO
        ),
        matrix_units=major >= 7,
        dynamic_vector_width=False,
    )
