"""Data-dependent indexing and multi-kernel routing author programs (pure Python/NumPy)."""

import numpy as np

from . import inputs
from .constants import BLOCK_SIZE, COLUMNS, EXPERTS, JAGGED_BATCH, NONZEROS, NONZEROS_PER_ROW, ROUTES
from .specs import KernelSpec, MoEAlignment


def embedding(context):
    kernel = context.compile(KernelSpec("embedding"))
    table = inputs.normal((1024, 128), "bf16")
    indices = inputs.integers(0, 1024, (4096,), "i64")
    return context.call(kernel, table, indices)


def csr_spmv(context):
    kernel = context.compile(KernelSpec("csr_spmv"))
    offsets = inputs.arange(0, NONZEROS + 1, NONZEROS_PER_ROW, dtype="i32")
    columns = inputs.integers(0, COLUMNS, (NONZEROS,), "i32")
    values = inputs.normal((NONZEROS,), "f32")
    vector = inputs.normal((COLUMNS,), "f32")
    return context.call(kernel, offsets, columns, values, vector)


def jagged_mean(context):
    kernel = context.compile(KernelSpec("jagged_mean"))
    lengths = np.arange(JAGGED_BATCH, dtype=np.int32) % 128 + 1
    offsets = inputs.array(np.concatenate((np.zeros(1, dtype=np.int32),
                                           np.cumsum(lengths, dtype=np.int32))), "i32")
    values = inputs.normal((33024, 128), "f32")
    return context.call(kernel, values, offsets)


def nonzero(context):
    kernel = context.compile(KernelSpec("nonzero"))
    values = inputs.normal((64, 4096), "f32")
    values.data[:, ::3] = 0.0
    indices, counts = context.call(kernel, values)
    return {"indices": indices, "counts": counts}


def moe_alignment(context):
    count = context.compile(KernelSpec("moe_count_routes"))
    prefix = context.compile(KernelSpec("moe_prefix_routes"))
    scatter = context.compile(KernelSpec("moe_scatter_routes"))
    mark = context.compile(KernelSpec("moe_mark_expert_blocks"))
    ids = inputs.array((np.arange(ROUTES, dtype=np.int32) % EXPERTS).reshape(4096, 2), "i32")
    program = MoEAlignment(count, prefix, scatter, mark)
    sorted_routes, expert_blocks, total_padded = context.call(program, ids)
    return {"route_ids": sorted_routes, "expert_blocks": expert_blocks,
            "total_padded": total_padded, "block_size": BLOCK_SIZE}
