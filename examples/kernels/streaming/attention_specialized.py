import intent
import intent.language as I

from kernels.streaming.attention import empty_attention_summary
from kernels.streaming.attention import merge_attention_summaries
from kernels.streaming.attention import normalize_attention_summary
from kernels.streaming.attention import reduce_score_maximum
from kernels.streaming.attention import summarize_attention_chunk_bf16
from kernels.streaming.attention import summarize_masked_attention_chunk


SINK_BATCH = 1
SINK_SEQUENCE = 4096
SINK_QUERY_HEADS = 32
SINK_KV_HEADS = 8
SINK_HEAD_DIMENSION = 128
GEMMA_BATCH = 2
GEMMA_SEQUENCE = 4096
GEMMA_QUERY_HEADS = 32
GEMMA_KV_HEADS = 8
GEMMA_HEAD_DIMENSION = 128
GEMMA_WINDOW = 1024
GEMMA_SOFT_CAP = 50.0
SWA_BATCH = 2
SWA_SEQUENCE = 4096
SWA_QUERY_HEADS = 32
SWA_KV_HEADS = 8
SWA_HEAD_DIMENSION = 128
SWA_WINDOW = 1024
BLOCK_CAUSAL_BATCH = 2
BLOCK_CAUSAL_SEQUENCE = 4096
BLOCK_CAUSAL_HEADS = 16
BLOCK_CAUSAL_DIMENSION = 128
BLOCK_CAUSAL_GROUP = 64
NATIVE_SPARSE_FORWARD_BATCH = 2
NATIVE_SPARSE_FORWARD_SEQUENCE = 4096
NATIVE_SPARSE_DECODE_BATCH = 8
NATIVE_SPARSE_DECODE_SEQUENCE = 8192
NATIVE_SPARSE_QUERY_HEADS = 32
NATIVE_SPARSE_KV_HEADS = 4
NATIVE_SPARSE_DIMENSION = 128
NATIVE_SPARSE_SELECTED_BLOCKS = 64
NATIVE_SPARSE_BLOCK = 64


@intent.fn
def window_attention_values(
    key_chunk,
    key_coordinates,
    queries,
    query_coordinates,
    scale,
    window,
    inclusive_lower,
    soft_cap,
):
    scores = I.matmul(
        queries,
        key_chunk,
        transpose_rhs=True,
        acc_dtype=I.f32,
    )
    if soft_cap > 0.0:
        scaled_scores = I.fdiv(
            scores * scale, soft_cap, approximate=True, flush_to_zero=True
        )
        scores = I.tanh(scaled_scores, approximate=True) * (
            soft_cap * I.LOG2E
        )
    else:
        scores = scores * (scale * I.LOG2E)
    valid = key_coordinates[None, :] <= query_coordinates[:, None]
    if window > 0:
        if inclusive_lower:
            lower = key_coordinates[None, :] >= query_coordinates[:, None] - window
        else:
            lower = key_coordinates[None, :] > query_coordinates[:, None] - window
        valid = valid & lower
    scores = I.select(valid, scores, -I.inf)
    chunk_valid = I.reduce.any(valid, axis=1)
    maximum = I.select(
        chunk_valid,
        reduce_score_maximum(scores, axis=1),
        0.0,
    )
    probability = I.select(
        valid,
        I.exp2(scores - maximum[:, None], approximate=True, flush_to_zero=True),
        0.0,
    )
    return I.record(
        valid=chunk_valid,
        maximum=maximum,
        denominator=I.reduce.sum(probability, axis=1),
        probability=probability,
    )


@intent.fn
def summarize_window_attention_f16(
    key_chunk,
    value_chunk,
    key_coordinates,
    queries,
    query_coordinates,
    scale,
    window,
    inclusive_lower,
    soft_cap,
):
    values = window_attention_values(
        key_chunk,
        key_coordinates,
        queries,
        query_coordinates,
        scale,
        window,
        inclusive_lower,
        soft_cap,
    )
    return I.record(
        valid=values.valid,
        maximum=values.maximum,
        denominator=values.denominator,
        accumulator=I.select(
            values.valid[:, None],
            I.matmul(
                I.cast(values.probability, I.f16),
                value_chunk,
                acc_dtype=I.f32,
            ),
            0.0,
        ),
    )


@intent.fn
def summarize_window_attention_bf16(
    key_chunk,
    value_chunk,
    key_coordinates,
    queries,
    query_coordinates,
    scale,
    window,
    inclusive_lower,
    soft_cap,
):
    values = window_attention_values(
        key_chunk,
        key_coordinates,
        queries,
        query_coordinates,
        scale,
        window,
        inclusive_lower,
        soft_cap,
    )
    return I.record(
        valid=values.valid,
        maximum=values.maximum,
        denominator=values.denominator,
        accumulator=I.select(
            values.valid[:, None],
            I.matmul(
                I.cast(values.probability, I.bf16),
                value_chunk,
                acc_dtype=I.f32,
            ),
            0.0,
        ),
    )


@intent.fn
def summarize_block_causal_chunk(
    key_chunk,
    value_chunk,
    key_coordinates,
    queries,
    query_coordinates,
    scale,
    half,
    block,
    sequence_start,
):
    scores = I.matmul(
        queries,
        key_chunk,
        transpose_rhs=True,
        acc_dtype=I.f32,
    ) * (scale * I.LOG2E)
    query_index = query_coordinates - sequence_start
    key_index = key_coordinates - sequence_start
    query_clean = query_index >= half
    key_clean = key_index >= half
    query_local = I.select(query_clean, query_index - half, query_index)
    key_local = I.select(key_clean, key_index - half, key_index)
    query_block = query_local // block
    key_block = key_local // block
    valid = (
        (query_block[:, None] == key_block[None, :])
        & (query_clean[:, None] == False)
        & (key_clean[None, :] == False)
    ) | (
        (query_block[:, None] > key_block[None, :])
        & (query_clean[:, None] == False)
        & key_clean[None, :]
    ) | (
        (query_block[:, None] >= key_block[None, :])
        & query_clean[:, None]
        & key_clean[None, :]
    )
    scores = I.select(valid, scores, -I.inf)
    chunk_valid = I.reduce.any(valid, axis=1)
    maximum = I.select(
        chunk_valid,
        reduce_score_maximum(scores, axis=1),
        0.0,
    )
    probability = I.select(
        valid,
        I.exp2(scores - maximum[:, None], approximate=True, flush_to_zero=True),
        0.0,
    )
    return I.record(
        valid=chunk_valid,
        maximum=maximum,
        denominator=I.reduce.sum(probability, axis=1),
        accumulator=I.matmul(
            I.cast(probability, I.f16),
            value_chunk,
            acc_dtype=I.f32,
        ),
    )


@intent.fn
def add_sink_to_summary(summary, sink_log2):
    maximum = I.select(
        summary.valid,
        I.maximum_num(summary.maximum, sink_log2),
        sink_log2,
    )
    data_scale = I.select(
        summary.valid,
        I.exp2(summary.maximum - maximum, approximate=True, flush_to_zero=True),
        0.0,
    )
    sink_scale = I.exp2(sink_log2 - maximum, approximate=True, flush_to_zero=True)
    return I.record(
        valid=I.full(summary.valid.shape, fill=True, dtype=I.bool),
        maximum=maximum,
        denominator=data_scale * summary.denominator + sink_scale,
        accumulator=data_scale[:, None] * summary.accumulator,
    )


@intent.kernel
def attention_sink_prefill(
    q: I.In[I.bf16, ("B", "HQ", "Q", "D")],
    k: I.In[I.bf16, ("B", "HK", "K", "D")],
    v: I.In[I.bf16, ("B", "HK", "K", "DV")],
    sinks: I.In[I.bf16, ("HQ",)],
    output: I.Out[I.bf16, ("B", "HQ", "Q", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
):
    B, HQ, Q, _ = q.shape
    K = k.shape[2]
    DV = v.shape[3]
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            summary = I.region_fold(
                source=(
                    k[batch, key_head, key_axis, :],
                    v[batch, key_head, key_axis, :],
                    I.indices(key_axis),
                ),
                axis=0,
                summarize=summarize_attention_chunk_bf16,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(Q, DV),
                operands=(
                    q[batch, query_head, query_axis, :],
                    I.indices(query_axis),
                    scale,
                    True,
                    True,
                ),
            )
            summary = add_sink_to_summary(
                summary,
                I.cast(sinks[query_head], I.f32) * I.LOG2E,
            )
            output[batch, query_head, query_axis, :] = I.cast(
                normalize_attention_summary(summary),
                I.bf16,
            )


@intent.kernel
def attention_sink_decode_partials(
    q: I.In[I.bf16, ("B", "HQ", "D")],
    k: I.In[I.bf16, ("B", "K", "HK", "D")],
    v: I.In[I.bf16, ("B", "K", "HK", "DV")],
    sinks: I.In[I.bf16, ("HQ",)],
    start_q: I.In[I.i32, (1,)],
    partial_lse: I.Out[I.f32, ("B", "HQ", "P")],
    partial_output: I.Out[I.bf16, ("B", "HQ", "P", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    WINDOW: I.Constexpr[int],
    P: I.Constexpr[int],
):
    B, HQ, D = q.shape
    K = k.shape[1]
    HK = k.shape[2]
    DV = v.shape[3]
    key_axis = I.domain(0, K)
    parts = I.domain(0, P)
    local_heads = I.domain(0, HEAD_GROUP)
    width = (K + P - 1) // P
    query_coordinate = I.cast(start_q[0], I.index)
    for batch in I.parallel(I.domain(0, B)):
        for key_head in I.parallel(I.domain(0, HK)):
            query_heads = key_head * HEAD_GROUP + I.indices(local_heads)
            query = I.gather(q, index=(batch, query_heads, slice(None)))
            sink_log2 = I.cast(I.gather(sinks, index=(query_heads,)), I.f32) * I.LOG2E
            for part in I.parallel(parts):
                begin = I.minimum(part * width, K)
                end = I.minimum((part + 1) * width, K)
                end = I.maximum(0, I.minimum(end, query_coordinate + 1))
                if WINDOW > 0:
                    begin = I.maximum(begin, query_coordinate - WINDOW + 1)
                begin = I.minimum(begin, end)
                keys = key_axis[begin:end]
                summary = I.region_fold(
                    source=(
                        k[batch, keys, key_head, :],
                        v[batch, keys, key_head, :],
                        I.indices(keys),
                    ),
                    axis=0,
                    summarize=summarize_window_attention_bf16,
                    combine=merge_attention_summaries,
                    identity=empty_attention_summary(local_heads, DV),
                    operands=(
                        query,
                        I.full((HEAD_GROUP,), fill=query_coordinate, dtype=I.index),
                        scale,
                        WINDOW,
                        False,
                        0.0,
                    ),
                )
                if part == 0:
                    summary = add_sink_to_summary(summary, sink_log2)
                safe_denominator = I.select(summary.valid, summary.denominator, 1.0)
                I.scatter_unique(
                    partial_lse,
                    index=(batch, query_heads, part),
                    value=I.select(
                        summary.valid,
                        summary.maximum + I.log(safe_denominator) * I.LOG2E,
                        -I.inf,
                    ),
                )
                I.scatter_unique(
                    partial_output,
                    index=(batch, query_heads, part, slice(None)),
                    value=I.cast(normalize_attention_summary(summary), I.bf16),
                )


@intent.kernel
def gemma_gqa_prefill(
    q: I.In[I.bf16, ("B", "HQ", "Q", "D")],
    k: I.In[I.bf16, ("B", "HK", "K", "D")],
    v: I.In[I.bf16, ("B", "HK", "K", "DV")],
    output: I.Out[I.bf16, ("B", "HQ", "Q", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    WINDOW: I.Constexpr[int],
    SOFT_CAP: I.Constexpr[float],
):
    B, HQ, Q, _ = q.shape
    K = k.shape[2]
    DV = v.shape[3]
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            summary = I.region_fold(
                source=(
                    k[batch, key_head, key_axis, :],
                    v[batch, key_head, key_axis, :],
                    I.indices(key_axis),
                ),
                axis=0,
                summarize=summarize_window_attention_bf16,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(Q, DV),
                operands=(
                    q[batch, query_head, query_axis, :],
                    I.indices(query_axis),
                    scale,
                    WINDOW,
                    True,
                    SOFT_CAP,
                ),
            )
            output[batch, query_head, query_axis, :] = I.cast(
                normalize_attention_summary(summary),
                I.bf16,
            )


@intent.kernel
def gemma_gqa_decode_partials(
    q: I.In[I.bf16, ("B", "HQ", 1, "D")],
    k: I.In[I.bf16, ("B", "HK", "K", "D")],
    v: I.In[I.bf16, ("B", "HK", "K", "DV")],
    partial_lse: I.Out[I.f32, ("B", "HQ", "P")],
    partial_output: I.Out[I.bf16, ("B", "HQ", "P", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    WINDOW: I.Constexpr[int],
    SOFT_CAP: I.Constexpr[float],
    P: I.Constexpr[int],
):
    B, HQ, _, D = q.shape
    HK = k.shape[1]
    K = k.shape[2]
    DV = v.shape[3]
    key_axis = I.domain(0, K)
    parts = I.domain(0, P)
    local_heads = I.domain(0, HEAD_GROUP)
    width = (K + P - 1) // P
    for batch in I.parallel(I.domain(0, B)):
        for key_head in I.parallel(I.domain(0, HK)):
            query_heads = key_head * HEAD_GROUP + I.indices(local_heads)
            I.assume_in_bounds(query_heads, q, axis=1)
            I.assume_in_bounds(query_heads, partial_lse, axis=1)
            I.assume_in_bounds(query_heads, partial_output, axis=1)
            query = I.gather(q, index=(batch, query_heads, 0, slice(None)))
            for part in I.parallel(parts):
                begin = I.minimum(part * width, K)
                end = I.minimum((part + 1) * width, K)
                if WINDOW > 0:
                    begin = I.minimum(end, I.maximum(begin, K - 1 - WINDOW))
                keys = key_axis[begin:end]
                summary = I.region_fold(
                    source=(
                        k[batch, key_head, keys, :],
                        v[batch, key_head, keys, :],
                        I.indices(keys),
                    ),
                    axis=0,
                    summarize=summarize_window_attention_bf16,
                    combine=merge_attention_summaries,
                    identity=empty_attention_summary(local_heads, DV),
                    operands=(
                        query,
                        I.full((HEAD_GROUP,), fill=K - 1, dtype=I.index),
                        scale,
                        WINDOW,
                        True,
                        SOFT_CAP,
                    ),
                )
                safe_denominator = I.select(summary.valid, summary.denominator, 1.0)
                I.scatter_unique(
                    partial_lse,
                    index=(batch, query_heads, part),
                    value=I.select(
                        summary.valid,
                        summary.maximum + I.log(safe_denominator) * I.LOG2E,
                        -I.inf,
                    ),
                )
                I.scatter_unique(
                    partial_output,
                    index=(batch, query_heads, part, slice(None)),
                    value=I.cast(normalize_attention_summary(summary), I.bf16),
                )


@intent.kernel
def sliding_window_gqa_prefill(
    q: I.In[I.f16, ("B", "HQ", "Q", "D")],
    k: I.In[I.f16, ("B", "HK", "K", "D")],
    v: I.In[I.f16, ("B", "HK", "K", "DV")],
    output: I.Out[I.f16, ("B", "HQ", "Q", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    WINDOW: I.Constexpr[int],
):
    B, HQ, Q, _ = q.shape
    K = k.shape[2]
    DV = v.shape[3]
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            summary = I.region_fold(
                source=(
                    k[batch, key_head, key_axis, :],
                    v[batch, key_head, key_axis, :],
                    I.indices(key_axis),
                ),
                axis=0,
                summarize=summarize_window_attention_f16,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(Q, DV),
                operands=(
                    q[batch, query_head, query_axis, :],
                    I.indices(query_axis),
                    scale,
                    WINDOW,
                    False,
                    0.0,
                ),
            )
            output[batch, query_head, query_axis, :] = I.cast(
                normalize_attention_summary(summary),
                I.f16,
            )


@intent.kernel
def block_causal_attention_fwd(
    q: I.In[I.f16, ("B", "Q", "H", "D")],
    k: I.In[I.f16, ("B", "K", "H", "D")],
    v: I.In[I.f16, ("B", "K", "H", "DV")],
    output: I.Out[I.f16, ("B", "Q", "H", "DV")],
    scale: I.f32,
    BLOCK: I.Constexpr[int],
):
    B, Q, H, _ = q.shape
    K = k.shape[1]
    DV = v.shape[3]
    half = Q // 2
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            summary = I.region_fold(
                source=(
                    k[batch, key_axis, head, :],
                    v[batch, key_axis, head, :],
                    I.indices(key_axis),
                ),
                axis=0,
                summarize=summarize_block_causal_chunk,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(Q, DV),
                operands=(
                    q[batch, query_axis, head, :],
                    I.indices(query_axis),
                    scale,
                    half,
                    BLOCK,
                    0,
                ),
            )
            output[batch, query_axis, head, :] = I.cast(
                normalize_attention_summary(summary),
                I.f16,
            )


@intent.kernel
def native_sparse_attention_fwd(
    q: I.In[I.f16, ("B", "Q", "HQ", "D")],
    k: I.In[I.f16, ("B", "K", "HK", "D")],
    v: I.In[I.f16, ("B", "K", "HK", "DV")],
    selected_blocks: I.In[I.i32, ("B", "Q", "HK", "S")],
    output: I.Out[I.f16, ("B", "Q", "HQ", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    BLOCK: I.Constexpr[int],
):
    B, Q, HQ, D = q.shape
    K = k.shape[1]
    DV = v.shape[3]
    slots = selected_blocks.shape[3]
    slot_axis = I.domain(0, slots)
    block_members = I.domain(0, BLOCK)
    token_count = slots * BLOCK
    for batch in I.parallel(I.domain(0, B)):
        for query_index in I.parallel(I.domain(0, Q)):
            for query_head in I.parallel(I.domain(0, HQ)):
                key_head = query_head // HEAD_GROUP
                block_id = I.cast(
                    selected_blocks[batch, query_index, key_head, slot_axis],
                    I.index,
                )
                key_indices = (
                    I.select(block_id >= 0, block_id, 0)[:, None] * BLOCK
                    + I.indices(block_members)[None, :]
                )
                active = (
                    (block_id[:, None] >= 0)
                    & (key_indices < K)
                    & (key_indices <= query_index)
                )
                safe_key_indices = I.select(active, key_indices, 0)
                keys = I.reshape(
                    I.gather(
                        k,
                        index=(batch, safe_key_indices, key_head, slice(None)),
                    ),
                    (token_count, D),
                )
                values = I.reshape(
                    I.gather(
                        v,
                        index=(batch, safe_key_indices, key_head, slice(None)),
                    ),
                    (token_count, DV),
                )
                summary = summarize_masked_attention_chunk(
                    keys,
                    values,
                    I.reshape(key_indices, (token_count,)),
                    I.reshape(active, (token_count,)),
                    I.reshape(q[batch, query_index, query_head, :], (1, D)),
                    I.full((1,), fill=query_index, dtype=I.index),
                    scale,
                )
                output[batch, query_index, query_head, :] = I.reshape(
                    I.cast(normalize_attention_summary(summary), I.f16),
                    (DV,),
                )


@intent.kernel
def varlen_block_causal_attention_fwd(
    q: I.In[I.f16, ("U", "H", "D")],
    k: I.In[I.f16, ("U", "H", "D")],
    v: I.In[I.f16, ("U", "H", "DV")],
    sequence_offsets: I.In[I.i32, ("B_PLUS_1",)],
    output: I.Out[I.f16, ("U", "H", "DV")],
    scale: I.f32,
    BLOCK: I.Constexpr[int],
):
    U, H, D = q.shape
    B = sequence_offsets.shape[0] - 1
    DV = v.shape[2]
    sequences = I.ragged(
        outer=I.domain(0, B),
        members=I.domain(0, U),
        offsets=sequence_offsets,
    )
    for sequence in I.parallel(sequences.outer):
        members = sequences[sequence]
        sequence_start = I.cast(sequence_offsets[sequence], I.index)
        sequence_length = I.cast(
            sequence_offsets[sequence + 1] - sequence_offsets[sequence],
            I.index,
        )
        half = sequence_length // 2
        for head in I.parallel(I.domain(0, H)):
            summary = I.region_fold(
                source=(
                    k[members, head, :],
                    v[members, head, :],
                    I.indices(members),
                ),
                axis=0,
                summarize=summarize_block_causal_chunk,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(members, DV),
                operands=(
                    q[members, head, :],
                    I.indices(members),
                    scale,
                    half,
                    BLOCK,
                    sequence_start,
                ),
            )
            output[members, head, :] = I.cast(
                normalize_attention_summary(summary),
                I.f16,
            )
