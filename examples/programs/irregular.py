"""Data-dependent indices and explicit multi-kernel routing."""

import numpy as np

from . import inputs

from kernels.backward.embedding import embedding_forward_lookup_bf16
from kernels.compaction.nonzero import compact_nonzero_rows
from kernels.ragged.jagged_mean import BATCH as JAGGED_BATCH, jagged_mean as jagged_definition
from kernels.routing.moe_align import (
    EXPERTS, ROUTES, PADDED_ROUTES, BLOCK_SIZE,
    moe_count_routes, moe_prefix_routes, moe_scatter_routes, moe_mark_expert_blocks,
)
from kernels.sparse.csr_spmv import (
    ROWS, COLUMNS, NONZEROS, NONZEROS_PER_ROW, csr_spmv as csr_definition,
)
from .composition import MoEAlignment


def embedding(context):
    kernel = context.compile(embedding_forward_lookup_bf16)
    table = inputs.normal((1024, 128), "bf16")
    indices = inputs.integers(0, 1024, (4096,), "i64")
    return context.call(kernel, table, indices)


def csr_spmv(context):
    kernel = context.compile(csr_definition)
    offsets = inputs.arange(0, NONZEROS + 1, NONZEROS_PER_ROW, dtype="i32")
    columns = inputs.integers(0, COLUMNS, (NONZEROS,), "i32")
    values = inputs.normal((NONZEROS,), "f32")
    vector = inputs.normal((COLUMNS,), "f32")
    return context.call(kernel, offsets, columns, values, vector)


def jagged_mean(context):
    kernel = context.compile(jagged_definition)
    lengths = np.arange(JAGGED_BATCH, dtype=np.int32) % 128 + 1
    offsets = inputs.array(np.concatenate((np.zeros(1, dtype=np.int32),
                                           np.cumsum(lengths, dtype=np.int32))), "i32")
    values = inputs.normal((33024, 128), "f32")
    return context.call(kernel, values, offsets)


def nonzero(context):
    kernel = context.compile(compact_nonzero_rows)
    values = inputs.normal((64, 4096), "f32")
    values.data[:, ::3] = 0.0
    indices, counts = context.call(kernel, values)
    return {"indices": indices, "counts": counts}


def moe_alignment(context):
    count = context.compile(moe_count_routes)
    prefix = context.compile(moe_prefix_routes)
    scatter = context.compile(moe_scatter_routes)
    mark = context.compile(moe_mark_expert_blocks)
    ids = inputs.array((np.arange(ROUTES, dtype=np.int32) % EXPERTS).reshape(4096, 2), "i32")
    program = MoEAlignment(count, prefix, scatter, mark)
    sorted_routes, expert_blocks, total_padded = context.call(program, ids)
    return {"route_ids": sorted_routes, "expert_blocks": expert_blocks,
            "total_padded": total_padded, "block_size": BLOCK_SIZE}
