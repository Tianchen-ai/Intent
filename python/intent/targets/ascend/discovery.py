from .specification import AscendCapabilities


def resolve_device(device):
    import torch
    import torch_npu
    from triton.runtime import driver
    if not torch.npu.is_available():
        raise RuntimeError("Triton-Ascend requires a local NPU and a working CANN/torch-npu installation")
    with torch.npu.device(device):
        target = driver.active.get_current_target()
        if target.backend != "npu":
            raise RuntimeError("Triton-Ascend did not resolve an NPU target")
        properties = torch.npu.get_device_properties(device)
        return AscendCapabilities(str(target.arch), int(properties.cube_core_num), int(properties.vector_core_num))
