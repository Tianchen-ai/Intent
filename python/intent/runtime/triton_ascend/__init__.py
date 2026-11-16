from functools import partial
from ..block.program import materialize_block_program
from ..triton.program import TritonProgram, _collect_triton_ir
from .device import AscendDevice, device_context


def materialize(source, module_text, entry_name, device, contract, *, source_path=None):
    import torch
    import torch_npu
    selected = torch.device("npu", device)
    return materialize_block_program(provider_name="triton_ascend",
        provider_type=partial(TritonProgram, device=AscendDevice()),
        source=source, module_text=module_text, entry_name=entry_name, contract=contract,
        device=selected, device_context=partial(device_context, selected),
        source_path=source_path, backend_ir_collector=_collect_triton_ir)
